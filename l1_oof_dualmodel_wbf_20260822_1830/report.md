# L1 Dual-Model OOF WBF Evaluation

## Boundary

- `auto_only` development OOF evaluation only; this is not manual truth or a deployment decision.
- The frozen 198-image `release-validation` set was not read, mounted, or evaluated.
- Each image uses exactly `YOLO11m_foldK + YOLO26m_foldK`; other-fold weights are rejected during preflight.
- WBF is a deterministic inlined fallback because `ensemble-boxes` is absent from the fixed shared venv and was not installed.

## Input Coverage

| Fold | Images | YOLO11m raw boxes | YOLO26m raw boxes | YOLO26m weight SHA-256 |
| --- | ---: | ---: | ---: | --- |
| 1 | 376 | 36840 | 41172 | `4824308e1080237e23115e5cb8cd0b2953a6177360e8e6bbadec90b812ffb5fe` |
| 2 | 375 | 67592 | 28401 | `eab8c360e5e4995c78018196a8751462cd1ed187115821da995ca28b8f245a8f` |
| 3 | 376 | 45394 | 30719 | `033a2e7d8746a601858eceb780f9fff865e79db9b483518e9e5aace51a2008ea` |

- Pooled images: `1127` (expected `1127`).
- Partial smoke run: `False`.
- The Phase 0 JSONL SHA-256 values and all YOLO26m raw prediction file SHA-256 values are in `preflight_and_output_audit.json`.

## Overall Metrics

Metric protocol: original-image coordinates, model-NMS input floor `0.001`, P/R/F1 at confidence `0.25` and IoU `0.50`, AP at IoU `0.50:0.95` using the established 101-point envelope.

| Variant | WBF IoU | Input boxes | Output boxes | P@0.25 | R@0.25 | F1@0.25 | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| yolo11m | n/a | n/a | 149826 | 0.468179 | 0.406994 | 0.435448 | 0.255454 | 0.144391 |
| yolo26m | n/a | n/a | 100292 | 0.530368 | 0.381150 | 0.443545 | 0.278321 | 0.148939 |
| wbf_iou_0.25 | 0.25 | 250118 | 111123 | 0.461983 | 0.464489 | 0.463232 | 0.291567 | 0.157876 |
| wbf_iou_0.35 | 0.35 | 250118 | 126120 | 0.459401 | 0.470423 | 0.464846 | 0.298550 | 0.162773 |
| wbf_iou_0.45 | 0.45 | 250118 | 145133 | 0.451128 | 0.473295 | 0.461946 | 0.303820 | 0.166466 |
| wbf_iou_0.55 | 0.55 | 250118 | 169187 | 0.440066 | 0.475337 | 0.457022 | 0.301300 | 0.167520 |

## Per-Class Metrics: WBF IoU 0.25

| Class | Support | P@0.25 | R@0.25 | F1@0.25 | AP50 | AP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| wuzi | 2632 | 0.393206 | 0.413374 | 0.403038 | 0.302446 | 0.121977 |
| waterstain | 7478 | 0.507295 | 0.585852 | 0.543751 | 0.530750 | 0.300985 |
| yellowstain | 78 | 0.080000 | 0.025641 | 0.038835 | 0.039662 | 0.008077 |
| nodule | 90 | 0.531250 | 0.755556 | 0.623853 | 0.724420 | 0.341118 |
| airhole | 1537 | 0.348399 | 0.240729 | 0.284725 | 0.187800 | 0.072653 |
| bubble | 763 | 0.282660 | 0.155963 | 0.201014 | 0.103757 | 0.038526 |
| bump | 281 | 0.263761 | 0.409253 | 0.320781 | 0.235623 | 0.115389 |
| fawu | 5 | 0.000000 | 0.000000 | 0.000000 | 0.270627 | 0.270627 |
| overfire | 304 | 0.451477 | 0.351974 | 0.395564 | 0.304491 | 0.157033 |
| NiOH | 7 | 0.000000 | 0.000000 | 0.000000 | 0.004641 | 0.002785 |
| greenstain | 334 | 0.392857 | 0.131737 | 0.197309 | 0.107483 | 0.033953 |
| suanwuran | 66 | 0.646154 | 0.636364 | 0.641221 | 0.623352 | 0.496824 |
| quejiao | 50 | 0.568966 | 0.660000 | 0.611111 | 0.610125 | 0.311262 |
| watermark | 2001 | 0.507839 | 0.453273 | 0.479007 | 0.406897 | 0.190263 |
| youpao | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |
| heibian | 12 | 0.300000 | 0.250000 | 0.272727 | 0.212993 | 0.064544 |
| mijixiaokeng | 33 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| fanxitong | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |

Fused-source composition: both=`35297`, YOLO11m-only=`46864`, YOLO26m-only=`28962`.

## Per-Class Metrics: WBF IoU 0.35

| Class | Support | P@0.25 | R@0.25 | F1@0.25 | AP50 | AP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| wuzi | 2632 | 0.392351 | 0.420973 | 0.406158 | 0.311758 | 0.127995 |
| waterstain | 7478 | 0.505501 | 0.589864 | 0.544433 | 0.537180 | 0.306941 |
| yellowstain | 78 | 0.076923 | 0.025641 | 0.038462 | 0.038372 | 0.007769 |
| nodule | 90 | 0.519084 | 0.755556 | 0.615385 | 0.720122 | 0.346361 |
| airhole | 1537 | 0.341772 | 0.245934 | 0.286039 | 0.195776 | 0.075592 |
| bubble | 763 | 0.279279 | 0.162516 | 0.205468 | 0.107958 | 0.042642 |
| bump | 281 | 0.263636 | 0.412811 | 0.321775 | 0.237862 | 0.117647 |
| fawu | 5 | 0.000000 | 0.000000 | 0.000000 | 0.270627 | 0.270627 |
| overfire | 304 | 0.444444 | 0.355263 | 0.394881 | 0.311598 | 0.163924 |
| NiOH | 7 | 0.000000 | 0.000000 | 0.000000 | 0.004641 | 0.002785 |
| greenstain | 334 | 0.394737 | 0.134731 | 0.200893 | 0.112524 | 0.035850 |
| suanwuran | 66 | 0.646154 | 0.636364 | 0.641221 | 0.627485 | 0.499917 |
| quejiao | 50 | 0.550000 | 0.660000 | 0.600000 | 0.608658 | 0.322097 |
| watermark | 2001 | 0.506233 | 0.466767 | 0.485699 | 0.423840 | 0.198212 |
| youpao | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |
| heibian | 12 | 0.300000 | 0.250000 | 0.272727 | 0.268397 | 0.086007 |
| mijixiaokeng | 33 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| fanxitong | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |

Fused-source composition: both=`35824`, YOLO11m-only=`55543`, YOLO26m-only=`34753`.

## Per-Class Metrics: WBF IoU 0.45

| Class | Support | P@0.25 | R@0.25 | F1@0.25 | AP50 | AP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| wuzi | 2632 | 0.383316 | 0.422492 | 0.401952 | 0.315456 | 0.131948 |
| waterstain | 7478 | 0.499887 | 0.592271 | 0.542172 | 0.539870 | 0.311800 |
| yellowstain | 78 | 0.076923 | 0.025641 | 0.038462 | 0.037169 | 0.007433 |
| nodule | 90 | 0.503759 | 0.744444 | 0.600897 | 0.700334 | 0.349076 |
| airhole | 1537 | 0.331046 | 0.251139 | 0.285609 | 0.201935 | 0.077703 |
| bubble | 763 | 0.277056 | 0.167759 | 0.208980 | 0.110410 | 0.045377 |
| bump | 281 | 0.261851 | 0.412811 | 0.320442 | 0.241164 | 0.120876 |
| fawu | 5 | 0.000000 | 0.000000 | 0.000000 | 0.270627 | 0.270627 |
| overfire | 304 | 0.442688 | 0.368421 | 0.402154 | 0.314536 | 0.172459 |
| NiOH | 7 | 0.000000 | 0.000000 | 0.000000 | 0.004641 | 0.002785 |
| greenstain | 334 | 0.386555 | 0.137725 | 0.203091 | 0.116056 | 0.035517 |
| suanwuran | 66 | 0.591549 | 0.636364 | 0.613139 | 0.624676 | 0.506422 |
| quejiao | 50 | 0.548387 | 0.680000 | 0.607143 | 0.655408 | 0.331787 |
| watermark | 2001 | 0.489074 | 0.469765 | 0.479225 | 0.425428 | 0.204139 |
| youpao | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |
| heibian | 12 | 0.300000 | 0.250000 | 0.272727 | 0.303406 | 0.095514 |
| mijixiaokeng | 33 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| fanxitong | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |

Fused-source composition: both=`35601`, YOLO11m-only=`67056`, YOLO26m-only=`42476`.

## Per-Class Metrics: WBF IoU 0.55

| Class | Support | P@0.25 | R@0.25 | F1@0.25 | AP50 | AP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| wuzi | 2632 | 0.372054 | 0.425912 | 0.397166 | 0.313548 | 0.134522 |
| waterstain | 7478 | 0.490927 | 0.593340 | 0.537297 | 0.539458 | 0.315358 |
| yellowstain | 78 | 0.074074 | 0.025641 | 0.038095 | 0.035531 | 0.007249 |
| nodule | 90 | 0.485507 | 0.744444 | 0.587719 | 0.694721 | 0.361427 |
| airhole | 1537 | 0.316517 | 0.253090 | 0.281273 | 0.201308 | 0.079846 |
| bubble | 763 | 0.266667 | 0.167759 | 0.205953 | 0.106764 | 0.046259 |
| bump | 281 | 0.257778 | 0.412811 | 0.317373 | 0.236388 | 0.121585 |
| fawu | 5 | 0.000000 | 0.000000 | 0.000000 | 0.270627 | 0.270627 |
| overfire | 304 | 0.429119 | 0.368421 | 0.396460 | 0.311391 | 0.172534 |
| NiOH | 7 | 0.000000 | 0.000000 | 0.000000 | 0.004641 | 0.002785 |
| greenstain | 334 | 0.388889 | 0.146707 | 0.213043 | 0.126943 | 0.038131 |
| suanwuran | 66 | 0.569444 | 0.621212 | 0.594203 | 0.617809 | 0.502417 |
| quejiao | 50 | 0.531250 | 0.680000 | 0.596491 | 0.642961 | 0.330731 |
| watermark | 2001 | 0.473815 | 0.474763 | 0.474289 | 0.424614 | 0.208505 |
| youpao | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |
| heibian | 12 | 0.272727 | 0.250000 | 0.260870 | 0.294095 | 0.088342 |
| mijixiaokeng | 33 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| fanxitong | 0 | 0.000000 | 0.000000 | 0.000000 | n/a | n/a |

Fused-source composition: both=`33859`, YOLO11m-only=`82530`, YOLO26m-only=`52798`.

## Gate

- Status: **`pass_keep`**.
- Selected WBF IoU for the stated rule: `0.55`.
- Rule: `At least one WBF IoU reaches +0.01 mAP50-95 versus the best single model or +0.01 F1 for a focus class.`
- Best pooled single-model mAP50-95 baseline: `0.148939`.
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
