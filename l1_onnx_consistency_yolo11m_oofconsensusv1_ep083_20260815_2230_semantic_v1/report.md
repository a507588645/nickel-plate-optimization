# L1 ONNX Consistency Verification

- Scope: deterministic development-only image sample; frozen release images were not used.
- Run: `E:\PythonProject\models\runs\dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_oofconsensusv1_finaldev_ep083`
- Release weight SHA-256: `bb501e224b48e610aac5c05293d0b3c63ef9eccad792712466a25803bd4de521`
- ONNX SHA-256: `901ccf6006f44689483aaf6f0d1d1b8c18132a85c7d881be7e62fcdccc255b21`
- Samples: 8
- Passed: True
- Global same-class match coverage: 0.966165
- Maximum per-image count delta: 1
- Maximum box absolute delta: 11.89550781
- Maximum confidence absolute delta: 0.15843368

The semantic gate uses same-class matches at the configured IoU threshold, global match coverage, and per-image count delta.
