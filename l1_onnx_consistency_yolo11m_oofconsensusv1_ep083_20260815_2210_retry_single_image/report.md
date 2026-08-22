# L1 ONNX Consistency Verification

- Scope: deterministic development-only image sample; frozen release images were not used.
- Run: `E:\PythonProject\models\runs\dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_oofconsensusv1_finaldev_ep083`
- Release weight SHA-256: `bb501e224b48e610aac5c05293d0b3c63ef9eccad792712466a25803bd4de521`
- ONNX SHA-256: `bf53f1a4612490d1749832bcc2a16fdac7991a752cb866dc616d14f2da00ae9c`
- Samples: 8
- Passed: False
- Maximum box absolute delta: 0.08215332
- Maximum confidence absolute delta: 0.00151104

The comparison uses the same imgsz, confidence, IoU, and max_det settings for both formats.
