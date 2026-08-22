# Phase 1 Fold-specific confidence profile

- Held-out evaluation fold: Fold 1
- Calibration folds: Fold 2, Fold 3
- Scope: archived OOF development predictions only; `release-validation` was not read.
- This is a research-side post-processing profile, not a deployed threshold or a model-selection result.

## Source files

- `fold02.jsonl`: 375 images, 67592 model-NMS predictions, SHA-256 `9d42f01d6b74018eb195b66962cf88700162ce5fb0bfdfcf4e9d82057f8963d9`
- `fold03.jsonl`: 376 images, 45394 model-NMS predictions, SHA-256 `bb729b1e53dafdc9e33d2054538a79d85318fc2086ec38e7e6e460dab7da335a`

## Phase 1 profile

- Default confidence: `0.25`
- Minimum non-held-out target support for an override: `100`
- Overrides: `{"airhole": 0.2, "bubble": 0.2, "bump": 0.3, "greenstain": 0.1, "watermark": 0.3, "waterstain": 0.3, "wuzi": 0.3}`

## Selection audit

| 类别 | 非 Fold 1 支持数 | 建议阈值 | Phase 1 阈值 | 选择 | F1@建议 |
| --- | ---: | ---: | ---: | --- | ---: |
| wuzi | 1777 | 0.30 | 0.30 | crossfold_override | 0.3795 |
| waterstain | 5355 | 0.30 | 0.30 | crossfold_override | 0.5229 |
| yellowstain | 57 | 0.10 | 0.25 | default_025 | 0.1205 |
| nodule | 65 | 0.40 | 0.25 | default_025 | 0.6066 |
| airhole | 931 | 0.20 | 0.20 | crossfold_override | 0.2519 |
| bubble | 621 | 0.20 | 0.20 | crossfold_override | 0.1638 |
| bump | 199 | 0.30 | 0.30 | crossfold_override | 0.2667 |
| fawu | 2 | 0.10 | 0.25 | default_025 | 1.0000 |
| overfire | 195 | 0.25 | 0.25 | default_025 | 0.3684 |
| NiOH | 5 | 0.90 | 0.25 | default_025 | 0.0000 |
| greenstain | 235 | 0.10 | 0.10 | crossfold_override | 0.1514 |
| suanwuran | 44 | 0.70 | 0.25 | default_025 | 0.6111 |
| quejiao | 36 | 0.40 | 0.25 | default_025 | 0.5846 |
| watermark | 1274 | 0.30 | 0.30 | crossfold_override | 0.4479 |
| youpao | 0 | n/a | 0.25 | default_025 | n/a |
| heibian | 7 | 0.50 | 0.25 | default_025 | 0.1818 |
| mijixiaokeng | 33 | 0.90 | 0.25 | default_025 | 0.0000 |
| fanxitong | 0 | n/a | 0.25 | default_025 | n/a |

The table deliberately excludes Fold 1 from threshold selection. Classes below the support threshold remain at the fixed default instead of inheriting noisy fold-specific optima.
