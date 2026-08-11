"""Integer fixed-point inference for the Martis-style SNN baseline.

The paper reports binary spike inputs, signed 8-bit weights, 32-bit membrane
potentials, and 13-bit decay values.  It does not publish trained observer
statistics or an exact Q-format for every value.  This module therefore keeps
the reported widths fixed and makes the missing scale choices explicit.

This is an arithmetic emulation for reproducibility, not a cycle-accurate FPGA
simulator.  Matrix products, decay products, membrane states, thresholds, and
resets stay in integer tensors until the final two-dimensional output is
converted back to real units for regression metrics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

WEIGHT_BITS = 8
WEIGHT_QMIN = -(2 ** (WEIGHT_BITS - 1))
WEIGHT_QMAX = 2 ** (WEIGHT_BITS - 1) - 1
DECAY_BITS = 13
DECAY_FRACTIONAL_BITS = 13
DECAY_QMIN = 0
DECAY_QMAX = 2**DECAY_BITS - 1
POTENTIAL_BITS = 32
POTENTIAL_QMIN = -(2 ** (POTENTIAL_BITS - 1))
POTENTIAL_QMAX = 2 ** (POTENTIAL_BITS - 1) - 1


def _round_half_away_from_zero(values: torch.Tensor) -> torch.Tensor:
    absolute = torch.floor(values.abs() + 0.5)
    return torch.where(values < 0, -absolute, absolute)


def _round_shift_signed(values: torch.Tensor, shift: int) -> torch.Tensor:
    """Divide signed int64 values by 2**shift with nearest rounding."""

    if values.dtype != torch.int64:
        raise TypeError("_round_shift_signed expects an int64 tensor")
    if shift < 1:
        raise ValueError("shift must be positive")
    rounded_absolute = (values.abs() + 2 ** (shift - 1)) >> shift
    return torch.where(values < 0, -rounded_absolute, rounded_absolute)


def _saturate_potential(values: torch.Tensor) -> tuple[torch.Tensor, int]:
    saturation_count = int(((values < POTENTIAL_QMIN) | (values > POTENTIAL_QMAX)).sum().item())
    return values.clamp(POTENTIAL_QMIN, POTENTIAL_QMAX).to(torch.int32), saturation_count


@dataclass(frozen=True)
class FixedPointLayer:
    name: str
    weight_int8: torch.Tensor
    weight_scale: float
    weight_fractional_bits: int | None
    beta_u13: torch.Tensor
    threshold_int32: torch.Tensor | None
    weight_saturation_count: int
    beta_saturation_count: int
    threshold_saturation_count: int
    weight_min_fp32: float
    weight_max_fp32: float
    weight_mse: float
    weight_max_abs_error: float


def _weight_scale(weight: torch.Tensor, scale_mode: str) -> tuple[float, int | None]:
    max_abs = float(weight.abs().max().item())
    if max_abs == 0.0:
        return 1.0, 0 if scale_mode == "pow2" else None
    if scale_mode == "maxabs":
        return max_abs / WEIGHT_QMAX, None
    if scale_mode == "pow2":
        fractional_bits = math.floor(math.log2(WEIGHT_QMAX / max_abs))
        return 2.0 ** (-fractional_bits), fractional_bits
    raise ValueError("weight_scale_mode must be 'pow2' or 'maxabs'")


def _quantize_layer(
    name: str,
    linear: torch.nn.Linear,
    neuron: Any,
    *,
    scale_mode: str,
    has_threshold: bool,
) -> FixedPointLayer:
    if linear.bias is not None:
        raise ValueError(f"{name} must not contain a bias")
    weight = linear.weight.detach().cpu().to(torch.float64)
    scale, fractional_bits = _weight_scale(weight, scale_mode)
    scaled_weight = weight / scale
    rounded_weight = _round_half_away_from_zero(scaled_weight)
    weight_saturation_count = int(
        ((rounded_weight < WEIGHT_QMIN) | (rounded_weight > WEIGHT_QMAX)).sum().item()
    )
    weight_int8 = rounded_weight.clamp(WEIGHT_QMIN, WEIGHT_QMAX).to(torch.int8)
    reconstructed_weight = weight_int8.to(torch.float64) * scale
    weight_error = reconstructed_weight - weight

    beta = torch.as_tensor(neuron.beta).detach().cpu().to(torch.float64).reshape(-1)
    scaled_beta = _round_half_away_from_zero(beta * (2**DECAY_FRACTIONAL_BITS))
    beta_saturation_count = int(
        ((scaled_beta < DECAY_QMIN) | (scaled_beta > DECAY_QMAX)).sum().item()
    )
    beta_u13 = scaled_beta.clamp(DECAY_QMIN, DECAY_QMAX).to(torch.int32)
    if beta_u13.numel() not in {1, linear.out_features}:
        raise ValueError(f"Unexpected beta shape for {name}: {tuple(beta.shape)}")
    if beta_u13.numel() == 1:
        beta_u13 = beta_u13.expand(linear.out_features).clone()

    threshold_int32: torch.Tensor | None = None
    threshold_saturation_count = 0
    if has_threshold:
        threshold = torch.as_tensor(neuron.threshold).detach().cpu().to(torch.float64).reshape(-1)
        scaled_threshold = _round_half_away_from_zero(threshold / scale)
        threshold_saturation_count = int(
            ((scaled_threshold < POTENTIAL_QMIN) | (scaled_threshold > POTENTIAL_QMAX)).sum().item()
        )
        threshold_int32 = scaled_threshold.clamp(POTENTIAL_QMIN, POTENTIAL_QMAX).to(torch.int32)
        if threshold_int32.numel() == 1:
            threshold_int32 = threshold_int32.expand(linear.out_features).clone()
        if threshold_int32.numel() != linear.out_features:
            raise ValueError(f"Unexpected threshold shape for {name}: {tuple(threshold.shape)}")
        if bool((threshold_int32 <= 0).any()):
            raise ValueError(f"Quantized threshold must be positive for {name}")

    return FixedPointLayer(
        name=name,
        weight_int8=weight_int8.contiguous(),
        weight_scale=float(scale),
        weight_fractional_bits=fractional_bits,
        beta_u13=beta_u13.contiguous(),
        threshold_int32=threshold_int32,
        weight_saturation_count=weight_saturation_count,
        beta_saturation_count=beta_saturation_count,
        threshold_saturation_count=threshold_saturation_count,
        weight_min_fp32=float(weight.min().item()),
        weight_max_fp32=float(weight.max().item()),
        weight_mse=float(torch.mean(weight_error.square()).item()),
        weight_max_abs_error=float(weight_error.abs().max().item()),
    )


class FixedPointSNN:
    """CPU integer inference using the precision widths reported in the paper."""

    def __init__(self, layers: tuple[FixedPointLayer, ...], scale_mode: str) -> None:
        if len(layers) != 4:
            raise ValueError("Expected three hidden layers and one output layer")
        self.layers = layers
        self.scale_mode = scale_mode
        self.last_diagnostics: dict[str, dict[str, int]] = {}

    @classmethod
    def from_float_model(
        cls,
        model: Any,
        *,
        weight_scale_mode: str = "pow2",
    ) -> FixedPointSNN:
        definitions = (
            ("fc1", model.fc1, model.lif1, True),
            ("fc2", model.fc2, model.lif2, True),
            ("fc3", model.fc3, model.lif3, True),
            ("fc_out", model.fc_out, model.leaky_out, False),
        )
        layers = tuple(
            _quantize_layer(
                name,
                linear,
                neuron,
                scale_mode=weight_scale_mode,
                has_threshold=has_threshold,
            )
            for name, linear, neuron, has_threshold in definitions
        )
        return cls(layers, weight_scale_mode)

    @staticmethod
    def _integer_current(spikes: torch.Tensor, layer: FixedPointLayer) -> torch.Tensor:
        # int32 matmul gives an integer accumulator; the largest fan-in here is
        # only 128, so a synaptic sum cannot overflow int32.
        return spikes.to(torch.int32) @ layer.weight_int8.to(torch.int32).T

    @staticmethod
    def _decay(mem: torch.Tensor, beta_u13: torch.Tensor) -> torch.Tensor:
        product = mem.to(torch.int64) * beta_u13.to(torch.int64)
        return _round_shift_signed(product, DECAY_FRACTIONAL_BITS)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape [time, batch, channels]")
        if features.shape[2] != self.layers[0].weight_int8.shape[1]:
            raise ValueError("Input channel count does not match fc1")
        if not bool(torch.all((features == 0) | (features == 1))):
            raise ValueError("Fixed-point SNN input must contain binary spikes")

        batch_size = features.shape[1]
        membranes = [
            torch.zeros(
                (batch_size, layer.weight_int8.shape[0]),
                dtype=torch.int32,
            )
            for layer in self.layers
        ]
        diagnostics = {
            layer.name: {
                "potential_saturation_count": 0,
                "max_abs_potential_integer": 0,
            }
            for layer in self.layers
        }
        outputs: list[torch.Tensor] = []

        for step in range(features.shape[0]):
            spikes = features[step].to(torch.int32)
            for layer_index, layer in enumerate(self.layers):
                previous_membrane = membranes[layer_index]
                current = self._integer_current(spikes, layer).to(torch.int64)
                candidate = self._decay(previous_membrane, layer.beta_u13) + current

                if layer.threshold_int32 is not None:
                    # snnTorch's default reset_delay=True: a spike detected from
                    # the previous state subtracts the threshold on this update.
                    delayed_reset = previous_membrane > layer.threshold_int32
                    candidate = candidate - (
                        delayed_reset.to(torch.int64) * layer.threshold_int32.to(torch.int64)
                    )

                membrane, saturation_count = _saturate_potential(candidate)
                membranes[layer_index] = membrane
                layer_diagnostics = diagnostics[layer.name]
                layer_diagnostics["potential_saturation_count"] += saturation_count
                max_abs = int(membrane.to(torch.int64).abs().max().item())
                layer_diagnostics["max_abs_potential_integer"] = max(
                    layer_diagnostics["max_abs_potential_integer"], max_abs
                )

                if layer.threshold_int32 is not None:
                    spikes = (membrane > layer.threshold_int32).to(torch.int32)
                else:
                    outputs.append(membrane.to(torch.float32) * layer.weight_scale)

        self.last_diagnostics = diagnostics
        return torch.stack(outputs)

    def quantization_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for layer in self.layers:
            threshold_min = (
                int(layer.threshold_int32.min().item())
                if layer.threshold_int32 is not None
                else None
            )
            threshold_max = (
                int(layer.threshold_int32.max().item())
                if layer.threshold_int32 is not None
                else None
            )
            rows.append(
                {
                    "layer": layer.name,
                    "shape": "x".join(str(value) for value in layer.weight_int8.shape),
                    "weight_bits": WEIGHT_BITS,
                    "weight_scale_mode": self.scale_mode,
                    "weight_scale": layer.weight_scale,
                    "weight_fractional_bits": layer.weight_fractional_bits,
                    "weight_min_fp32": layer.weight_min_fp32,
                    "weight_max_fp32": layer.weight_max_fp32,
                    "weight_mse": layer.weight_mse,
                    "weight_max_abs_error": layer.weight_max_abs_error,
                    "weight_saturation_count": layer.weight_saturation_count,
                    "decay_bits": DECAY_BITS,
                    "decay_fractional_bits": DECAY_FRACTIONAL_BITS,
                    "decay_integer_min": int(layer.beta_u13.min().item()),
                    "decay_integer_max": int(layer.beta_u13.max().item()),
                    "decay_saturation_count": layer.beta_saturation_count,
                    "potential_bits": POTENTIAL_BITS,
                    "threshold_integer_min": threshold_min,
                    "threshold_integer_max": threshold_max,
                    "threshold_saturation_count": layer.threshold_saturation_count,
                }
            )
        return rows

    def artifact_arrays(self) -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {}
        for layer in self.layers:
            arrays[f"{layer.name}_weight_int8"] = layer.weight_int8.numpy()
            arrays[f"{layer.name}_weight_scale"] = np.asarray(layer.weight_scale, dtype=np.float64)
            if layer.weight_fractional_bits is not None:
                arrays[f"{layer.name}_weight_fractional_bits"] = np.asarray(
                    layer.weight_fractional_bits, dtype=np.int32
                )
            # NumPy has no 13-bit scalar dtype. uint16 is the smallest storage
            # container; valid values are explicitly limited to [0, 8191].
            arrays[f"{layer.name}_decay_u13"] = layer.beta_u13.numpy().astype(np.uint16)
            if layer.threshold_int32 is not None:
                arrays[f"{layer.name}_threshold_int32"] = layer.threshold_int32.numpy()
        return arrays

    def metadata(self) -> dict[str, Any]:
        return {
            "input": {
                "semantic": "binary spike event",
                "logical_bits": 1,
                "integer_values": [0, 1],
            },
            "weight": {
                "bits": WEIGHT_BITS,
                "signed": True,
                "integer_range": [WEIGHT_QMIN, WEIGHT_QMAX],
                "scale_granularity": "one scale per Linear layer",
                "scale_mode": self.scale_mode,
            },
            "decay": {
                "bits": DECAY_BITS,
                "signed": False,
                "format_assumption": "UQ0.13",
                "fractional_bits": DECAY_FRACTIONAL_BITS,
                "integer_range": [DECAY_QMIN, DECAY_QMAX],
            },
            "potential": {
                "bits": POTENTIAL_BITS,
                "signed": True,
                "integer_range": [POTENTIAL_QMIN, POTENTIAL_QMAX],
                "scale": "same real-unit scale as the corresponding layer weights",
            },
            "threshold": {
                "bits": POTENTIAL_BITS,
                "scale": "same as the corresponding membrane potential",
            },
            "arithmetic": {
                "synaptic_accumulator": "signed int32",
                "decay_product": "signed int64 intermediate, rounded then int32 saturated",
                "membrane_update": "signed int32 with saturation",
                "rounding": "nearest, half away from zero",
                "hidden_reset": "subtract with snnTorch reset_delay=True semantics",
                "output_conversion": "final int32 membrane multiplied by layer scale",
            },
        }
