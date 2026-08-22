# L1 Dual-Model OOF WBF Evaluation

## Boundary

- `auto_only` development OOF evaluation only; this is not manual truth or a deployment decision.
- The frozen 198-image `release-validation` set was not read, mounted, or evaluated.
- Each image uses exactly `YOLO11m_foldK + YOLO26m_foldK`; other-fold weights are rejected during preflight.
- WBF is a deterministic inlined fallback because `ensemble-boxes` is absent from the fixed shared venv and was not installed.

## Input Coverage

| Fold | Images | YOLO11m raw boxes | YOLO26m raw boxes | YOLO26m weight SHA-256 |
| --- | ---: | ---: | ---: | --- |
| 1 | 1 | 51 | 70 | `4824308e1080237e23115e5cb8cd0b2953a6177360e8e6bbadec90b812ffb5fe` |
| 2 | 1 | 176 | 56 | `eab8c360e5e4995c78018196a8751462cd1ed187115821da995ca28b8f245a8f` |
| 3 | 1 | 220 | 159 | `033a2e7d8746a601858eceb780f9fff865e79db9b483518e9e5aace51a2008ea` |

- Pooled images: `3` (expected `1127`).
- Partial smoke run: `True`.
- The Phase 0 JSONL SHA-256 values and all YOLO26m raw prediction file SHA-256 values are in `preflight_and_output_audit.json`.

## Overall Metrics

Metric protocol: original-image coordinates, model-NMS input floor `0.001`, P/R/F1 at confidence `0.25` and IoU `0.50`, AP at IoU `0.50:0.95` using the established 101-point envelope.

| Variant | WBF IoU | Input boxes | Output boxes | P@0.25 | R@0.25 | F1@0.25 | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| yolo11m | n/a | n/a | 447 | 0.600000 | 0.095745 | 0.165138 | 0.156788 | 0.061032 |
| yolo26m | n/a | n/a | 285 | 0.541667 | 0.138298 | 0.220339 | 0.140095 | 0.052359 |
| wbf_iou_0.25 | 0.25 | 732 | 330 | 0.555556 | 0.159574 | 0.247934 | 0.181050 | 0.070812 |
| wbf_iou_0.35 | 0.35 | 732 | 382 | 0.555556 | 0.159574 | 0.247934 | 0.179661 | 0.067019 |
| wbf_iou_0.45 | 0.45 | 732 | 440 | 0.555556 | 0.159574 | 0.247934 | 0.178150 | 0.066919 |
| wbf_iou_0.55 | 0.55 | 732 | 522 | 0.555556 | 0.159574 | 0.247934 | 0.176958 | 0.067171 |

## Per-Class Metrics: WBF IoU 0.25

| Class | Support | P@0.25 | R@0.25 | F1@0.25 | AP50 | AP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| wuzi | 1 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| waterstain | 30 | 0.875000 | 0.466667 | 0.608696 | 0.596545 | 0.218137 |
| yellowstain | 1 | 0.000000 | 0.000000 | 0.000000 | 0.333333 | 0.133333 |
| nodule | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |
| airhole | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |
| bubble | 43 | 0.000000 | 0.000000 | 0.000000 | 0.012856 | 0.004510 |
| bump | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |
| fawu | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |
| overfire | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |
| NiOH | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |
| greenstain | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |
| suanwuran | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |
| quejiao | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |
| watermark | 7 | 0.250000 | 0.142857 | 0.181818 | 0.143564 | 0.068894 |
| youpao | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |
| heibian | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |
| mijixiaokeng | 12 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| fanxitong | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |

Fused-source composition: both=`87`, YOLO11m-only=`139`, YOLO26m-only=`104`.

## Per-Class Metrics: WBF IoU 0.35

| Class | Support | P@0.25 | R@0.25 | F1@0.25 | AP50 | AP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| wuzi | 1 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| waterstain | 30 | 0.875000 | 0.466667 | 0.608696 | 0.589879 | 0.222031 |
| yellowstain | 1 | 0.000000 | 0.000000 | 0.000000 | 0.333333 | 0.133333 |
| nodule | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |
| airhole | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |
| bubble | 43 | 0.000000 | 0.000000 | 0.000000 | 0.016521 | 0.004973 |
| bump | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |
| fawu | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |
| overfire | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |
| NiOH | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |
| greenstain | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |
| suanwuran | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |
| quejiao | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |
| watermark | 7 | 0.250000 | 0.142857 | 0.181818 | 0.138233 | 0.041775 |
| youpao | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |
| heibian | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |
| mijixiaokeng | 12 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| fanxitong | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |

Fused-source composition: both=`84`, YOLO11m-only=`173`, YOLO26m-only=`125`.

## Per-Class Metrics: WBF IoU 0.45

| Class | Support | P@0.25 | R@0.25 | F1@0.25 | AP50 | AP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| wuzi | 1 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| waterstain | 30 | 0.875000 | 0.466667 | 0.608696 | 0.586110 | 0.223458 |
| yellowstain | 1 | 0.000000 | 0.000000 | 0.000000 | 0.333333 | 0.133333 |
| nodule | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |
| airhole | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |
| bubble | 43 | 0.000000 | 0.000000 | 0.000000 | 0.015793 | 0.004340 |
| bump | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |
| fawu | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |
| overfire | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |
| NiOH | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |
| greenstain | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |
| suanwuran | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |
| quejiao | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |
| watermark | 7 | 0.250000 | 0.142857 | 0.181818 | 0.133663 | 0.040382 |
| youpao | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |
| heibian | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |
| mijixiaokeng | 12 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| fanxitong | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |

Fused-source composition: both=`82`, YOLO11m-only=`215`, YOLO26m-only=`143`.

## Per-Class Metrics: WBF IoU 0.55

| Class | Support | P@0.25 | R@0.25 | F1@0.25 | AP50 | AP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| wuzi | 1 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| waterstain | 30 | 0.875000 | 0.466667 | 0.608696 | 0.584563 | 0.223052 |
| yellowstain | 1 | 0.000000 | 0.000000 | 0.000000 | 0.333333 | 0.133333 |
| nodule | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |
| airhole | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |
| bubble | 43 | 0.000000 | 0.000000 | 0.000000 | 0.017617 | 0.005056 |
| bump | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |
| fawu | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |
| overfire | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |
| NiOH | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |
| greenstain | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |
| suanwuran | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |
| quejiao | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |
| watermark | 7 | 0.250000 | 0.142857 | 0.181818 | 0.126238 | 0.041584 |
| youpao | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |
| heibian | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |
| mijixiaokeng | 12 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| fanxitong | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |

Fused-source composition: both=`70`, YOLO11m-only=`272`, YOLO26m-only=`180`.

## Gate

- Status: **`partial_smoke_no_gate`**.
- Selected WBF IoU for the stated rule: `0.25`.
- Rule: `A --max-images-per-fold limit was used; no experiment-selection gate is valid.`
- Best pooled single-model mAP50-95 baseline: `0.061032`.
- Candidate deltas and every focus-class comparison are retained in `gate_decision.json`.

## Reproducibility Files

- `raw_predictions_yolo26/fold01.jsonl` through `fold03.jsonl`: Phase 0-compatible YOLO26m raw OOF schema.
- `fusion_predictions/wbf_iou_*.jsonl`: exact fused prediction records and original targets.
- `metrics_summary.csv`, `per_class_metrics.csv`, and `metrics/*.json`: machine-readable metric tables.
- `random_pair_audit.json`: deterministic five-image provenance sample; all source folds must match each image fold.
- `environment.json` and `preflight_and_output_audit.json`: runtime, file hashes, and no-leakage checks.

## Limitations

- OOF labels are `auto_only`; no result establishes manual-ground-truth accuracy or stable multi-seed benefit.
- WBF IoUs are scanned on this same development OOF set, so a keep result is only a candidate for later frozen evaluation under an approved protocol.
