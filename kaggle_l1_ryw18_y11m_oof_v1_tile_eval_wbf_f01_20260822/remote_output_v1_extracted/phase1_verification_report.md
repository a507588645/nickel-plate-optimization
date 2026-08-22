# L1 Phase 1 tiled post-processing evaluation verification

- Archive: `E:\PythonProject\优化\kaggle_l1_ryw18_y11m_oof_v1_tile_eval_wbf_f01_20260822\remote_output_v1\dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_oofconsensusv1_tile1024o25_focus6_mix_fold01_evaluation_results.zip`
- Archive SHA-256: `252bcb40bc39b6de1b5cf3484ce1e2109e38b13d532fda7fc17d6a4337a4d4af`
- Scope: development-only Fold 1; frozen test: `None`
- Fixed baseline SHA-256: `b00aad9cc8041424f0b1e7263a6c9a2be5675827fd7017fcdec395b40ccac0ce`
- Recovered tile checkpoint SHA-256: `c4a903e8e8438798d44fc0548d666923a8728509f012d124101a8b980d7d875e`

## Profile metrics

| Profile | Baseline tiled mAP50-95 | Tile model tiled mAP50-95 | Tile model minus baseline full |
| --- | ---: | ---: | ---: |
| legacy | 0.023219 | 0.063427 | -0.104574 |
| perclass_nms | 0.021883 | 0.050408 | -0.117593 |
| wbf | 0.019617 | 0.050063 | -0.117939 |
| wbf_edge | 0.022955 | 0.052221 | -0.115781 |

## Attribution

| Model | Raw crop predictions | Below 0.25 | Touch internal crop edge | wbf_edge outputs |
| --- | ---: | ---: | ---: | ---: |
| baseline | 1309610 | 1260957 | 689069 | 21789 |
| tile_model | 470692 | 459709 | 326895 | 3458 |

## Phase 1 gate

- Status: `fail_stop_tile_direction`
- `tile_model_wbf_edge - baseline_full`: `-0.115781` mAP50-95
- Focus-class AP/Recall non-degradation: `False`
