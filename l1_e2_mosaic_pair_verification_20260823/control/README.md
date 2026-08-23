# L1 E2 paired Mosaic ablation run

Role: `control`; Mosaic: `1.0`; Pair: `l1_e2_mosaicpair_f01_ft30_v1`. This run is development-only Fold 1 and starts a new `resume=false` fine-tune from the shared archived checkpoint.  Its only intended training variable relative to its paired sibling is `mosaic`. The frozen release-validation set is not included.
