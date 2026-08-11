"""Martis et al. (2024) MUA-to-SNN reproduction utilities.

The module intentionally keeps paper-faithful choices explicit.  The Sabes
``spikes`` matrix stores unsorted events in slot 0 and sorted units in the
remaining slots.  A channel-level MUA event stream is reconstructed by taking
the union of every valid slot on that physical electrode and then making a
binary 1 ms bin, which also matches the activity density of the authors'
published example data.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import snntorch as snn
import torch
from scipy.ndimage import uniform_filter1d
from scipy.signal import lfilter
from snntorch import surrogate
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


@dataclass(frozen=True)
class TaskWindow:
    """A chronological reach/task interval on the 1 kHz time grid."""

    task_index: int
    start: int
    end: int

    @property
    def steps(self) -> int:
        return self.end - self.start


@dataclass
class PreparedSession:
    """Preprocessed channel-level MUA and behavioral targets."""

    session_name: str
    time_sec: np.ndarray
    mua_binary: np.ndarray
    velocity: np.ndarray
    cursor_position: np.ndarray
    target_position_250hz: np.ndarray
    channel_names: list[str]
    task_windows: list[TaskWindow]
    original_sample_count: int
    original_sample_rate_hz: float
    mua_event_count: int
    mua_mode: str
    moving_average_mode: str


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters reported by Martis et al. (2024)."""

    hidden_sizes: tuple[int, int, int] = (64, 128, 64)
    threshold: float = 0.1
    beta_init: float = 0.9
    learning_rate: float = 1e-3
    batch_size: int = 10
    epochs: int = 100
    seed: int = 42


@dataclass
class TrainResult:
    history: pd.DataFrame
    best_epoch: int
    best_validation_loss: float
    best_state_dict: dict[str, torch.Tensor]


def _decode_matlab_char(dataset: h5py.Dataset) -> str:
    values = np.asarray(dataset).ravel(order="F")
    return "".join(chr(int(value)) for value in values if int(value) != 0).strip()


def _read_channel_names(handle: h5py.File) -> list[str]:
    references = handle["chan_names"]
    return [
        _decode_matlab_char(handle[references[0, index]])
        for index in range(references.shape[1])
    ]


def _is_spike_vector(dataset: h5py.Dataset) -> bool:
    # MATLAB empty cells in these files are uint64 [0, 0] sentinels.
    return dataset.dtype.kind == "f" and dataset.size > 0


def _moving_average(values: np.ndarray, order: int, mode: str) -> np.ndarray:
    """Apply the paper's moving-average operation with an explicit phase rule."""

    if mode == "centered":
        # MATLAB movmean-like offline smoothing. Boundary values are extended.
        return uniform_filter1d(values, size=order, axis=0, mode="nearest")
    if mode == "causal":
        kernel = np.ones(order, dtype=np.float64) / order
        return lfilter(kernel, [1.0], values, axis=0)
    raise ValueError("moving_average_mode must be 'centered' or 'causal'")


def _read_channel_events(
    handle: h5py.File,
    channel_index: int,
    start_sec: float,
    end_sec: float,
    mode: str,
) -> np.ndarray:
    refs = handle["spikes"]
    if mode == "all_threshold_crossings":
        slots: Iterable[int] = range(refs.shape[0])
    elif mode == "unsorted_only":
        slots = (0,)
    else:
        raise ValueError(
            "mua_mode must be 'all_threshold_crossings' or 'unsorted_only'"
        )

    parts: list[np.ndarray] = []
    for slot in slots:
        dataset = handle[refs[slot, channel_index]]
        if not _is_spike_vector(dataset):
            continue
        values = np.asarray(dataset, dtype=np.float64).ravel()
        values = values[(values >= start_sec) & (values <= end_sec)]
        if values.size:
            parts.append(values)

    if not parts:
        return np.empty(0, dtype=np.float64)
    # Different unit slots are disjoint in principle, but unique also protects
    # the binary MUA stream from duplicate timestamps.
    return np.unique(np.concatenate(parts))


def _task_windows_from_target_changes(
    original_time: np.ndarray,
    original_target_position: np.ndarray,
    time_1khz: np.ndarray,
) -> list[TaskWindow]:
    """Recover complete reaches from target transitions.

    The MAT files do not contain a separate trial-id vector.  A new target is
    the task boundary.  Excluding the partial interval before the first target
    transition and the trailing interval after the final transition yields the
    reach counts printed in Martis et al. Table I (583 for 20170127_03).
    """

    changed = np.r_[
        True,
        np.any(original_target_position[1:] != original_target_position[:-1], axis=1),
    ]
    change_indices = np.flatnonzero(changed)
    if change_indices.size < 3:
        raise ValueError("Too few target changes to construct task windows")

    starts_sec = original_time[change_indices[1:-1]]
    ends_sec = original_time[change_indices[2:]]

    windows: list[TaskWindow] = []
    for task_index, (start_sec, end_sec) in enumerate(zip(starts_sec, ends_sec)):
        start = int(np.searchsorted(time_1khz, start_sec, side="left"))
        end = int(np.searchsorted(time_1khz, end_sec, side="left"))
        if end > start:
            windows.append(TaskWindow(task_index=task_index, start=start, end=end))
    return windows


def prepare_sabes_session(
    mat_path: Path,
    *,
    bin_width_sec: float = 0.001,
    mua_mode: str = "all_threshold_crossings",
    moving_average_mode: str = "centered",
) -> PreparedSession:
    """Load one Sabes MAT file and construct paper-style 1 kHz tensors."""

    if not mat_path.exists():
        raise FileNotFoundError(mat_path)
    if not np.isclose(bin_width_sec, 0.001):
        raise ValueError("This reproduction is defined for 1 ms bins")

    with h5py.File(mat_path, "r") as handle:
        original_time = np.asarray(handle["t"], dtype=np.float64).squeeze()
        cursor_position = np.asarray(handle["cursor_pos"], dtype=np.float64).T
        target_position = np.asarray(handle["target_pos"], dtype=np.float64).T
        channel_names = _read_channel_names(handle)

        if len(channel_names) != 96:
            raise ValueError(f"Expected 96 channels, found {len(channel_names)}")
        if not all(name.startswith("M1 ") for name in channel_names):
            raise ValueError("This reproduction expects the 96-channel M1 sessions")

        start_sec = float(original_time[0])
        end_sec = float(original_time[-1])
        n_steps = int(np.floor((end_sec - start_sec) / bin_width_sec)) + 1
        time_1khz = start_sec + np.arange(n_steps, dtype=np.float64) * bin_width_sec
        mua_binary = np.zeros((n_steps, len(channel_names)), dtype=np.bool_)

        for channel_index in range(len(channel_names)):
            events = _read_channel_events(
                handle,
                channel_index,
                start_sec,
                end_sec,
                mua_mode,
            )
            if events.size == 0:
                continue
            indices = np.floor((events - start_sec) / bin_width_sec).astype(np.int64)
            indices = indices[(indices >= 0) & (indices < n_steps)]
            mua_binary[np.unique(indices), channel_index] = True

    position_1khz = np.column_stack(
        [
            np.interp(time_1khz, original_time, cursor_position[:, axis])
            for axis in range(2)
        ]
    )
    smoothed_position = _moving_average(position_1khz, 32, moving_average_mode)
    velocity = np.diff(
        smoothed_position,
        axis=0,
        prepend=smoothed_position[:1],
    ) / bin_width_sec
    velocity = _moving_average(velocity, 8, moving_average_mode).astype(np.float32)

    task_windows = _task_windows_from_target_changes(
        original_time,
        target_position,
        time_1khz,
    )
    sample_rate = (original_time.size - 1) / (original_time[-1] - original_time[0])

    return PreparedSession(
        session_name=mat_path.stem,
        time_sec=time_1khz,
        mua_binary=mua_binary,
        velocity=velocity,
        cursor_position=position_1khz.astype(np.float32),
        target_position_250hz=target_position,
        channel_names=channel_names,
        task_windows=task_windows,
        original_sample_count=int(original_time.size),
        original_sample_rate_hz=float(sample_rate),
        mua_event_count=int(mua_binary.sum()),
        mua_mode=mua_mode,
        moving_average_mode=moving_average_mode,
    )


def session_summary(session: PreparedSession) -> pd.DataFrame:
    durations = np.array([window.steps for window in session.task_windows])
    duration_sec = session.time_sec[-1] - session.time_sec[0]
    row = {
        "session": session.session_name,
        "duration_min": duration_sec / 60,
        "original_rate_hz": session.original_sample_rate_hz,
        "model_rate_hz": 1000.0,
        "channels": session.mua_binary.shape[1],
        "complete_tasks": len(session.task_windows),
        "task_ms_median": float(np.median(durations)),
        "task_ms_min": int(durations.min()),
        "task_ms_max": int(durations.max()),
        "binary_events": session.mua_event_count,
        "input_density": float(session.mua_binary.mean()),
        "mean_rate_per_channel_hz": float(session.mua_binary.mean() * 1000),
    }
    return pd.DataFrame([row])


def split_task_windows(
    windows: Sequence[TaskWindow],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> tuple[list[TaskWindow], list[TaskWindow], list[TaskWindow]]:
    """Chronological split matching the authors' public notebook."""

    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("Split ratios must sum to 1")
    n_total = len(windows)
    n_train = round(ratios[0] * n_total)
    n_validation = round(ratios[1] * n_total)
    train = list(windows[:n_train])
    validation = list(windows[n_train : n_train + n_validation])
    test = list(windows[n_train + n_validation :])
    return train, validation, test


def filter_task_windows(
    windows: Sequence[TaskWindow],
    *,
    min_steps: int = 200,
    max_steps: int = 4000,
) -> tuple[list[TaskWindow], list[TaskWindow]]:
    """Separate genuine reach windows from acquisition pauses."""

    kept = [window for window in windows if min_steps <= window.steps <= max_steps]
    rejected = [window for window in windows if window not in kept]
    return kept, rejected


class ReachWindowDataset(Dataset):
    """Lazy task-window views; the full 70 MB MUA matrix is not copied."""

    def __init__(
        self,
        session: PreparedSession,
        windows: Sequence[TaskWindow],
        *,
        truncate_steps: int | None = None,
    ) -> None:
        self.session = session
        self.windows = list(windows)
        self.truncate_steps = truncate_steps

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = self.windows[index]
        end = window.end
        if self.truncate_steps is not None:
            end = min(end, window.start + self.truncate_steps)
        features = torch.from_numpy(
            self.session.mua_binary[window.start:end].astype(np.float32)
        )
        targets = torch.from_numpy(self.session.velocity[window.start:end])
        return features, targets


def pad_reach_batch(
    batch: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return time-major tensors and a valid-timestep mask."""

    max_steps = max(features.shape[0] for features, _ in batch)
    batch_size = len(batch)
    input_size = batch[0][0].shape[1]
    output_size = batch[0][1].shape[1]

    features = torch.zeros(max_steps, batch_size, input_size, dtype=torch.float32)
    targets = torch.zeros(max_steps, batch_size, output_size, dtype=torch.float32)
    mask = torch.zeros(max_steps, batch_size, 1, dtype=torch.float32)

    for index, (item_features, item_targets) in enumerate(batch):
        steps = item_features.shape[0]
        features[:steps, index] = item_features
        targets[:steps, index] = item_targets
        mask[:steps, index, 0] = 1.0
    return features, targets, mask


class MartisSNN(nn.Module):
    """96-64-128-64-2 dense LIF regressor from Martis et al."""

    def __init__(
        self,
        input_size: int = 96,
        output_size: int = 2,
        hidden_sizes: tuple[int, int, int] = (64, 128, 64),
        threshold: float = 0.1,
        beta_init: float = 0.9,
        optimized_forward: bool = False,
    ) -> None:
        super().__init__()
        self.optimized_forward = optimized_forward
        h1, h2, h3 = hidden_sizes
        spike_gradient = surrogate.fast_sigmoid()

        self.fc1 = nn.Linear(input_size, h1, bias=False)
        self.lif1 = snn.Leaky(
            beta=torch.full((h1,), beta_init),
            threshold=threshold,
            learn_beta=True,
            learn_threshold=False,
            spike_grad=spike_gradient,
            reset_mechanism="subtract",
        )
        self.fc2 = nn.Linear(h1, h2, bias=False)
        self.lif2 = snn.Leaky(
            beta=torch.full((h2,), beta_init),
            threshold=threshold,
            learn_beta=True,
            learn_threshold=False,
            spike_grad=spike_gradient,
            reset_mechanism="subtract",
        )
        self.fc3 = nn.Linear(h2, h3, bias=False)
        self.lif3 = snn.Leaky(
            beta=torch.full((h3,), beta_init),
            threshold=threshold,
            learn_beta=True,
            learn_threshold=False,
            spike_grad=spike_gradient,
            reset_mechanism="subtract",
        )
        self.fc_out = nn.Linear(h3, output_size, bias=False)
        # The paper uses the output membrane voltage directly for regression.
        self.leaky_out = snn.Leaky(
            beta=torch.full((output_size,), beta_init),
            threshold=1.0,
            learn_beta=True,
            learn_threshold=False,
            spike_grad=spike_gradient,
            reset_mechanism="none",
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Run a time-major [T, B, 96] input with state reset at the start."""

        if self.optimized_forward:
            return self._forward_functional(features)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem_out = self.leaky_out.init_leaky()
        outputs: list[torch.Tensor] = []

        for step in range(features.shape[0]):
            current1 = self.fc1(features[step])
            spikes1, mem1 = self.lif1(current1, mem1)
            current2 = self.fc2(spikes1)
            spikes2, mem2 = self.lif2(current2, mem2)
            current3 = self.fc3(spikes2)
            spikes3, mem3 = self.lif3(current3, mem3)
            current_out = self.fc_out(spikes3)
            _, mem_out = self.leaky_out(current_out, mem_out)
            outputs.append(mem_out)

        return torch.stack(outputs)

    def _forward_functional(self, features: torch.Tensor) -> torch.Tensor:
        """Equivalent LIF equations without mutating snnTorch module buffers.

        snnTorch's default ``reset_delay=True`` computes a detached reset from
        the previous membrane value, applies it during the next state update,
        and then emits a spike from the updated membrane.  Keeping membrane
        tensors local preserves those equations and the surrogate gradient,
        while avoiding shape checks and Python-side buffer assignments at
        every timestep. This form also reduces Python-side overhead.
        """

        batch_size = features.shape[1]
        mem1 = features.new_zeros((batch_size, self.fc1.out_features))
        mem2 = features.new_zeros((batch_size, self.fc2.out_features))
        mem3 = features.new_zeros((batch_size, self.fc3.out_features))
        mem_out = features.new_zeros((batch_size, self.fc_out.out_features))
        outputs: list[torch.Tensor] = []

        beta1 = self.lif1.beta.clamp(0, 1)
        beta2 = self.lif2.beta.clamp(0, 1)
        beta3 = self.lif3.beta.clamp(0, 1)
        beta_out = self.leaky_out.beta.clamp(0, 1)
        threshold1 = self.lif1.threshold
        threshold2 = self.lif2.threshold
        threshold3 = self.lif3.threshold

        for step in range(features.shape[0]):
            reset1 = self.lif1.spike_grad(mem1 - threshold1).detach()
            mem1 = beta1 * mem1 + self.fc1(features[step]) - reset1 * threshold1
            spikes1 = self.lif1.spike_grad(mem1 - threshold1)

            reset2 = self.lif2.spike_grad(mem2 - threshold2).detach()
            mem2 = beta2 * mem2 + self.fc2(spikes1) - reset2 * threshold2
            spikes2 = self.lif2.spike_grad(mem2 - threshold2)

            reset3 = self.lif3.spike_grad(mem3 - threshold3).detach()
            mem3 = beta3 * mem3 + self.fc3(spikes2) - reset3 * threshold3
            spikes3 = self.lif3.spike_grad(mem3 - threshold3)

            # Output reset_mechanism="none": pure leaky integration.  The
            # output spike is unused in the original regression forward pass.
            mem_out = beta_out * mem_out + self.fc_out(spikes3)
            outputs.append(mem_out)

        return torch.stack(outputs)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    expanded_mask = mask.expand_as(prediction)
    squared_error = (prediction - target).square() * expanded_mask
    return squared_error.sum() / expanded_mask.sum().clamp_min(1.0)


def make_dataloaders(
    session: PreparedSession,
    train_windows: Sequence[TaskWindow],
    validation_windows: Sequence[TaskWindow],
    config: TrainConfig,
    *,
    truncate_steps: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = ReachWindowDataset(
        session,
        train_windows,
        truncate_steps=truncate_steps,
    )
    validation_dataset = ReachWindowDataset(
        session,
        validation_windows,
        truncate_steps=truncate_steps,
    )
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,  # matches the chronological public training notebook
        num_workers=0,
        collate_fn=pad_reach_batch,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=pad_reach_batch,
    )
    return train_loader, validation_loader


def set_reproducible_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(
    model: MartisSNN,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: TrainConfig,
    device: torch.device,
    *,
    progress_dir: Path | None = None,
) -> TrainResult:
    set_reproducible_seed(config.seed)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history_rows: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state = deepcopy(model.state_dict())

    progress = tqdm(range(1, config.epochs + 1), desc="Martis SNN training")
    for epoch in progress:
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        for features, target, mask in train_loader:
            features = features.to(device)
            target = target.to(device)
            mask = mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(features)
            loss = masked_mse(prediction, target, mask)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach().cpu())
            train_batches += 1

        model.eval()
        validation_loss_sum = 0.0
        validation_batches = 0
        with torch.no_grad():
            for features, target, mask in validation_loader:
                features = features.to(device)
                target = target.to(device)
                mask = mask.to(device)
                prediction = model(features)
                loss = masked_mse(prediction, target, mask)
                validation_loss_sum += float(loss.detach().cpu())
                validation_batches += 1

        train_loss = train_loss_sum / max(train_batches, 1)
        validation_loss = validation_loss_sum / max(validation_batches, 1)
        history_rows.append(
            {
                "epoch": epoch,
                "train_mse": train_loss,
                "validation_mse": validation_loss,
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            if progress_dir is not None:
                progress_dir.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, progress_dir / "best_model_in_progress.pt")
        if progress_dir is not None:
            pd.DataFrame(history_rows).to_csv(
                progress_dir / "training_history_in_progress.csv", index=False
            )
        progress.set_postfix(train=f"{train_loss:.3g}", val=f"{validation_loss:.3g}")

    return TrainResult(
        history=pd.DataFrame(history_rows),
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        best_state_dict=best_state,
    )


def predict_continuous(
    model: MartisSNN,
    session: PreparedSession,
    test_windows: Sequence[TaskWindow],
    device: torch.device,
    *,
    max_steps: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the chronological test recording without resetting per reach."""

    if not test_windows:
        raise ValueError("No test windows")
    start = test_windows[0].start
    end = test_windows[-1].end
    if max_steps is not None:
        end = min(end, start + max_steps)
    features = torch.from_numpy(session.mua_binary[start:end].astype(np.float32))
    features = features.unsqueeze(1).to(device)
    model.eval()
    with torch.no_grad():
        prediction = model(features).squeeze(1).cpu().numpy()
    target = session.velocity[start:end]
    time_sec = session.time_sec[start:end] - session.time_sec[start]
    return time_sec, target, prediction


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for axis, name in enumerate(("vx", "vy")):
        y = target[:, axis].astype(np.float64)
        y_hat = prediction[:, axis].astype(np.float64)
        residual_sum = np.square(y - y_hat).sum()
        total_sum = np.square(y - y.mean()).sum()
        r2 = 1.0 - residual_sum / total_sum if total_sum > 0 else np.nan
        if np.std(y) > 0 and np.std(y_hat) > 0:
            cc = float(np.corrcoef(y, y_hat)[0, 1])
        else:
            cc = np.nan
        rmse = float(np.sqrt(np.mean(np.square(y - y_hat))))
        rows.append({"axis": name, "R2": float(r2), "CC": cc, "RMSE": rmse})
    result = pd.DataFrame(rows)
    result.loc[len(result)] = {
        "axis": "mean",
        "R2": result["R2"].mean(),
        "CC": result["CC"].mean(),
        "RMSE": result["RMSE"].mean(),
    }
    return result


def config_as_dict(config: TrainConfig) -> dict:
    return asdict(config)
