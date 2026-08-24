# L1 E1b Per-Class Confidence Scan

## Boundary

- `auto_only` development OOF research only; it is not manual truth, a final model selection, or a deployment threshold decision.
- No new inference or training ran. The scan only filters persisted post-model-NMS OOF dumps.
- The frozen `release-validation` set was not read, mounted, linked, or evaluated.
- Each row changes one focus class only; a six-class joint threshold profile has not been evaluated.

## Raw Anchor Reproduction

| Variant | Persisted boxes | Fixed E1 mAP50-95 | Reproduced mAP50-95 | Delta |
| --- | ---: | ---: | ---: | ---: |
| yolo26m | 100292 | 0.148939 | 0.148939 | +0.000000000 |
| wbf_iou_0.55 | 169187 | 0.167520 | 0.167520 | +0.000000000 |

The fixed E1 anchors use all persisted predictions down to the original model floor `0.001`. The scan gate instead uses an all-class `0.25` operating control, because a runtime confidence override must be compared with the same filtered decision path.

## Thresholded Controls

| Variant | P@0.25 | R@0.25 | F1@0.25 | mAP50 | mAP50-95 | Delta vs raw anchor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| yolo26m | 0.530368 | 0.381150 | 0.443545 | 0.200741 | 0.112763 | -0.036176 |
| wbf_iou_0.55 | 0.440066 | 0.475337 | 0.457022 | 0.242693 | 0.135488 | -0.032032 |

## Gate Candidates

Gate: focus-class Recall increase `>= +0.05` versus its `0.25` operating control and pooled mAP50-95 change `>= -0.005` versus that same control.

| Variant | Class | Threshold | P | R | F1 | Recall delta | Pooled mAP50-95 delta | Gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| yolo26m | wuzi | 0.05 | 0.273000 | 0.449848 | 0.339791 | +0.136018 | +0.001105 | pass |
| yolo26m | wuzi | 0.10 | 0.336010 | 0.397416 | 0.364143 | +0.083587 | +0.000751 | pass |
| yolo26m | waterstain | 0.05 | 0.381893 | 0.633458 | 0.476511 | +0.130650 | +0.001836 | pass |
| yolo26m | waterstain | 0.10 | 0.454951 | 0.586788 | 0.512527 | +0.083980 | +0.001325 | pass |
| yolo26m | waterstain | 0.15 | 0.505136 | 0.558973 | 0.530693 | +0.056165 | +0.000861 | pass |
| yolo26m | airhole | 0.05 | 0.241583 | 0.266103 | 0.253251 | +0.087833 | +0.000531 | pass |
| yolo26m | airhole | 0.10 | 0.306550 | 0.228367 | 0.261745 | +0.050098 | +0.000347 | pass |
| yolo26m | greenstain | 0.05 | 0.227758 | 0.191617 | 0.208130 | +0.074850 | +0.000555 | pass |
| yolo26m | watermark | 0.05 | 0.434155 | 0.466267 | 0.449639 | +0.111444 | +0.001501 | pass |
| yolo26m | watermark | 0.10 | 0.485210 | 0.426287 | 0.453844 | +0.071464 | +0.001045 | pass |
| wbf_iou_0.55 | wuzi | 0.05 | 0.202394 | 0.558891 | 0.297172 | +0.132979 | +0.000835 | pass |
| wbf_iou_0.55 | wuzi | 0.10 | 0.262589 | 0.517097 | 0.348305 | +0.091185 | +0.000626 | pass |
| wbf_iou_0.55 | wuzi | 0.15 | 0.301932 | 0.481003 | 0.370989 | +0.055091 | +0.000434 | pass |
| wbf_iou_0.55 | waterstain | 0.05 | 0.292328 | 0.719444 | 0.415733 | +0.126103 | +0.001353 | pass |
| wbf_iou_0.55 | waterstain | 0.10 | 0.362482 | 0.679593 | 0.472788 | +0.086253 | +0.001025 | pass |
| wbf_iou_0.55 | waterstain | 0.15 | 0.409904 | 0.649773 | 0.502690 | +0.056432 | +0.000683 | pass |
| wbf_iou_0.55 | airhole | 0.05 | 0.176544 | 0.368250 | 0.238668 | +0.115159 | +0.000458 | pass |
| wbf_iou_0.55 | airhole | 0.10 | 0.229848 | 0.324658 | 0.269148 | +0.071568 | +0.000320 | pass |
| wbf_iou_0.55 | greenstain | 0.05 | 0.166667 | 0.248503 | 0.199519 | +0.101796 | +0.000437 | pass |
| wbf_iou_0.55 | greenstain | 0.10 | 0.236486 | 0.209581 | 0.222222 | +0.062874 | +0.000297 | pass |
| wbf_iou_0.55 | watermark | 0.05 | 0.327449 | 0.574713 | 0.417196 | +0.099950 | +0.000958 | pass |
| wbf_iou_0.55 | watermark | 0.10 | 0.380049 | 0.540730 | 0.446370 | +0.065967 | +0.000690 | pass |

## Decision

- Status: **`pass_keep_candidate`**.
- Passing one-class candidates: `22`.
- Detailed curves and every threshold row are in `focus_threshold_curves.csv`; no passing row is a tested joint profile.

## Limitations

- Thresholds are selected and measured on the same OOF development labels, so selection bias remains.
- The OOF labels are `auto_only`; no result proves human-label accuracy, generalization, or a stable multi-seed benefit.
- The stored source predictions were already model-NMS output. This scan cannot recover candidates removed by model NMS.
