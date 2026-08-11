# Dataset download and placement

이 프로젝트는 O'Doherty et al.의 **Nonhuman Primate Reaching with Multichannel Sensorimotor Cortex Electrophysiology** 데이터셋 중 5개 Indy 세션을 사용합니다.

- 공식 배포 페이지: https://zenodo.org/records/3854034
- DOI: https://doi.org/10.5281/zenodo.3854034
- 필요한 다운로드 용량: 약 1.8 GB

> 공식 페이지에 연결된 별도의 raw broadband supplement가 아니라, 메인 레코드의 세션별 `.mat` 파일을 받으세요. 학습 코드는 `.mat` 내부의 `spikes`, `cursor_pos`, `target_pos`, `t`, `chan_names` 등을 사용합니다.

## 필요한 파일

| 파일 | Zenodo 표시 크기 | 공식 MD5 |
|---|---:|---|
| `indy_20160622_01.mat` | 909.0 MB | `c33d5fff31320d709d23fe445561fb6e` |
| `indy_20160630_01.mat` | 382.2 MB | `197413a5339630ea926cbd22b8b43338` |
| `indy_20170124_01.mat` | 152.0 MB | `ea1721fefe443420a356b4f93f5bb731` |
| `indy_20170127_03.mat` | 171.8 MB | `3d77802c59e3e4ebe6778326cfc68c0a` |
| `indy_20170131_02.mat` | 208.8 MB | `2790b1c869564afaa7772dbf9e42d784` |

## 방법 1: 웹 브라우저로 다운로드

1. [Zenodo 레코드](https://zenodo.org/records/3854034)를 엽니다.
2. `Files` 목록에서 위의 5개 파일 이름을 찾습니다.
3. 각 파일 오른쪽의 `Download`를 누릅니다.
4. 다운로드한 파일을 아래 경로로 옮깁니다.

```text
data/
└─ sabes_zenodo/
   └─ master_mat/
      ├─ indy_20160622_01.mat
      ├─ indy_20160630_01.mat
      ├─ indy_20170124_01.mat
      ├─ indy_20170127_03.mat
      └─ indy_20170131_02.mat
```

## 방법 2: PowerShell로 일괄 다운로드

저장소 루트에서 실행합니다.

```powershell
$DataDirectory = "data\sabes_zenodo\master_mat"
$ZenodoFiles = "https://zenodo.org/records/3854034/files"
$Sessions = @(
    "indy_20160622_01",
    "indy_20160630_01",
    "indy_20170124_01",
    "indy_20170127_03",
    "indy_20170131_02"
)

New-Item -ItemType Directory -Force -Path $DataDirectory | Out-Null

foreach ($Session in $Sessions) {
    $FileName = "${Session}.mat"
    $Destination = Join-Path $DataDirectory $FileName
    Invoke-WebRequest `
        -Uri "$ZenodoFiles/$FileName?download=1" `
        -OutFile $Destination
    Write-Host "Downloaded: $Destination"
}
```

연결이 중간에 끊기면 해당 파일을 삭제한 후 그 파일만 다시 받거나 브라우저 다운로드를 사용하세요.

## 무결성 확인

다운로드 후 공식 MD5와 비교합니다.

```powershell
$DataDirectory = "data\sabes_zenodo\master_mat"
$ExpectedMd5 = @{
    "indy_20160622_01.mat" = "c33d5fff31320d709d23fe445561fb6e"
    "indy_20160630_01.mat" = "197413a5339630ea926cbd22b8b43338"
    "indy_20170124_01.mat" = "ea1721fefe443420a356b4f93f5bb731"
    "indy_20170127_03.mat" = "3d77802c59e3e4ebe6778326cfc68c0a"
    "indy_20170131_02.mat" = "2790b1c869564afaa7772dbf9e42d784"
}

foreach ($Entry in $ExpectedMd5.GetEnumerator()) {
    $Path = Join-Path $DataDirectory $Entry.Key
    $ActualMd5 = (Get-FileHash -Algorithm MD5 -LiteralPath $Path).Hash.ToLowerInvariant()
    if ($ActualMd5 -ne $Entry.Value) {
        throw "MD5 mismatch: $($Entry.Key)"
    }
    Write-Host "OK: $($Entry.Key)"
}
```

5개 파일이 모두 `OK`이면 다음 단계로 진행합니다.

```powershell
uv run python scripts/run_experiment.py --config configs/baseline.yaml --dry-run
```

## 주의사항

- 원본 `.mat` 파일은 Git에 추가하지 않습니다. `.gitignore`에서 제외하도록 설정돼 있습니다.
- 전체 Zenodo 레코드를 받을 필요는 없습니다. 위의 5개 세션만 사용합니다.
- 데이터의 이용 조건과 공식 citation은 Zenodo 레코드에서 다시 확인하세요.
- 데이터 형식이나 preprocessing을 변경하면 YAML config와 결과 `protocol.json`에 기록하세요.

