# 폴더 이동 및 Git 업데이트 안내

## 만들어지는 이동용 폴더

`github_release/cross-session-snn_portable_20260813/`

이 폴더는 현재 저장소의 독립 복사본입니다. 다음을 포함합니다.

- `.git`: 기존 commit history와 origin 원격 주소
- 코드: `src/`, `scripts/`
- 설정: `configs/`
- 문서: `README.md`, `HANDOFF_KO.md`, `docs/`
- 원시 SABES `.mat` 데이터
- 완료된 결과와 현재 중단된 체크포인트
- 결과 이미지와 CSV
- 논문 PDF의 로컬 참고 복사본(존재하는 경우)

다음은 제외합니다.

- `.venv/`: 약 1.5 GB이며 새 위치에서 다시 만드는 것이 안전함
- `.ruff_cache/`와 Python cache

## 새 위치로 옮기는 방법

`cross-session-snn_portable_20260813` 폴더 전체를 외장 디스크나 새 경로로 복사합니다. 폴더 내부 일부만 선택해서 옮기면 데이터나 체크포인트가 빠질 수 있습니다.

복사 후 PowerShell에서 다음을 실행합니다.

```powershell
Set-Location '새로운경로\cross-session-snn_portable_20260813'
powershell -ExecutionPolicy Bypass -File scripts\verify_portable_bundle.ps1
```

`uv`가 설치돼 있다면 환경을 다시 만듭니다.

```powershell
uv sync --dev
.\.venv\Scripts\python.exe -c "import torch, numpy, pandas, snntorch; print('environment ok')"
```

## 중단된 실험 재개

세 seed를 다시 병렬 실행하려면:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\resume_transfer_adaptation_search.ps1
```

직접 한 seed만 실행하려면:

```powershell
.\.venv\Scripts\python.exe scripts\train_transfer_adaptation_search.py --seed 42 --cpu-threads 4 --resume
```

실험이 모두 끝나면:

```powershell
.\.venv\Scripts\python.exe scripts\summarize_transfer_adaptation_search.py
```

## Git에 올라가는 것과 올라가지 않는 것

`.gitignore`에 따라 다음은 기본적으로 Git에 올라갑니다.

- Python 코드
- YAML 설정
- Markdown 문서
- `pyproject.toml`, `uv.lock`

다음은 로컬 이동용 폴더 안에는 있지만 Git에는 올라가지 않습니다.

- `data/*`와 `.mat`
- `outputs/`
- `*.pt`, `*.npz`, `*.npy`
- 논문 PDF
- `.venv/`

이 구분은 GitHub의 파일 크기 제한과 데이터 라이선스·개인 문서 문제를 피하기 위한 것입니다.

## 권장 Git 업데이트 명령

먼저 상태를 확인합니다.

```powershell
git status --short
git diff --check
```

코드·설정·문서만 명시적으로 추가합니다.

```powershell
git add HANDOFF_KO.md README.md pyproject.toml uv.lock
git add src scripts configs docs
git status --short
```

대용량 파일이 staged되지 않았는지 확인한 뒤 commit합니다.

```powershell
git commit -m "Add transfer channel selection and multi-seed adaptation experiments"
git push origin main
```

`git status`에서 `data/`, `outputs/`, `.mat`, `.pt`가 보이거나 staged돼 있다면 commit하기 전에 중단하고 `.gitignore` 적용을 다시 확인합니다.

## Git으로 결과를 공유하는 방법

`outputs/` 전체는 무시되므로 결과 요약을 Git에 남기고 싶다면 필요한 작은 CSV와 PNG를 `results/transfer_sutl_norm20/` 같은 추적 가능한 폴더로 복사한 뒤 크기와 개인정보를 확인해야 합니다. 원시 prediction 배열과 checkpoint는 올리지 않는 것을 권장합니다.

## 논문 PDF

이동용 폴더에는 개인 참고용으로 PDF를 복사할 수 있지만 `.gitignore`가 `*.pdf`를 차단합니다. 출판사 저작권 조건을 확인하지 않은 상태에서 GitHub에 업로드하지 마세요.
