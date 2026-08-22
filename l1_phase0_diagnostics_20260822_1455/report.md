# L1 Phase 0 CPU Diagnostics

## Boundary

- CPU-only prediction over the three archived OOF development validation folds.
- `release-validation` was not read, mounted, or evaluated.
- Results are `auto_only`; they tune no deployed threshold and select no final model.

## Fold Inputs

| Fold | Images | Raw predictions after model NMS | Weight SHA-256 |
| --- | ---: | ---: | --- |
| 1 | 376 | 36840 | `b00aad9cc8041424f0b1e7263a6c9a2be5675827fd7017fcdec395b40ccac0ce` |
| 2 | 375 | 67592 | `7b8c766b54efd3b25a4a3c8c03f51cf85bfde7d999769d77332eca3ae2579a8d` |
| 3 | 376 | 45394 | `6d0e39d87fc85d239ab3b2ec84fe027d072836e3715f8d6618999227544e044b` |

## Pooled Per-Class Threshold Recommendations

| Class | Targets | Threshold | P@0.5 | R@0.5 | F1 | Delta recall vs 0.25 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| wuzi | 2632 | 0.300 | 0.4415 | 0.3267 | 0.3755 | -0.0236 |
| waterstain | 7478 | 0.300 | 0.5621 | 0.4989 | 0.5286 | -0.0234 |
| yellowstain | 78 | 0.100 | 0.1351 | 0.0641 | 0.0870 | +0.0513 |
| nodule | 90 | 0.400 | 0.7051 | 0.6111 | 0.6548 | -0.0444 |
| airhole | 1537 | 0.250 | 0.3205 | 0.2010 | 0.2471 | +0.0000 |
| bubble | 763 | 0.250 | 0.2936 | 0.1389 | 0.1886 | +0.0000 |
| bump | 281 | 0.350 | 0.3025 | 0.2562 | 0.2775 | -0.0427 |
| fawu | 5 | 0.100 | 0.6667 | 0.4000 | 0.5000 | +0.4000 |
| overfire | 304 | 0.200 | 0.4353 | 0.3322 | 0.3769 | +0.0164 |
| NiOH | 7 | 0.900 | 0.0000 | 0.0000 | 0.0000 | +0.0000 |
| greenstain | 334 | 0.100 | 0.1982 | 0.1347 | 0.1604 | +0.0419 |
| suanwuran | 66 | 0.900 | 1.0000 | 0.4394 | 0.6105 | -0.1515 |
| quejiao | 50 | 0.400 | 0.6750 | 0.5400 | 0.6000 | -0.0600 |
| watermark | 2001 | 0.300 | 0.5007 | 0.3803 | 0.4323 | -0.0155 |
| youpao | 0 | n/a | n/a | n/a | n/a | n/a |
| heibian | 12 | 0.300 | 0.3333 | 0.2500 | 0.2857 | +0.0000 |
| mijixiaokeng | 33 | 0.900 | 0.0000 | 0.0000 | 0.0000 | +0.0000 |
| fanxitong | 0 | n/a | n/a | n/a | n/a | n/a |

Thresholds are selected by maximum pooled F1 at IoU=0.50; exact ties prefer the higher threshold.
The full threshold curves are in `per_class_threshold_curves.csv`; focus-class PR plots are in `focus_class_pr_curves.png` when matplotlib is available.

## G4b Attribution Availability

- Status: `inconclusive_no_per_box_details`.
- Detail files: 0.
- Phase 1 requirement: Persist per-tile raw boxes before and after cross-tile merging so the Phase 0 attribution gate can be measured.

The existing G4b ZIP only contains aggregate reports, so it cannot prove whether overlap duplicates or low-confidence boxes exceed 50%.
Phase 1 must persist raw per-tile predictions before evaluating WBF, edge filtering, or per-class thresholds.
