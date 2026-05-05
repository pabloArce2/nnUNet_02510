# Pig Binary MONAI Pipeline (Current Code)

```mermaid
flowchart TD
    A[Raw data folders\n- labeled animal CT\n- labeled liver masks\n- unlabeled CT pool] --> B[create_pig_binary_split.py]
    B --> C[splits_pig_binary.json\nkeys: ssl_pretrain, labeled, train, val, val_unlabeled, test_unlabeled]

    C --> D[train_monai_seg_pig_binary.py wrapper]
    D --> D1[Rebase paths to repo root]
    D1 --> D2[Canonicalize CT orientation (nib.as_closest_canonical)]
    D2 --> D3[Convert labels to binary liver mask (label>0)]
    D3 --> D4[Write split_preprocessed.json]

    D4 --> E[train_monai_seg.py]
    E --> E1[Optional: load SSL encoder init\n--init-ssl-checkpoint]
    E --> E2[Align labels to image grid\n(resample + axis-permutation fallback)]
    E2 --> E3[Label preflight checks\nshape/affine/foreground sanity]

    E3 --> F[Training transforms]
    F --> F1[Load + channel first + intensity scaling]
    F1 --> F2[Patch sampling]
    F2 --> F2a[RandSpatialCropd (default)]
    F2 --> F2b[RandCropByPosNegLabeld if --use-pos-neg-crop\nnum_samples=--train-num-samples]
    F2a --> F3[Augment: flips + gaussian noise]
    F2b --> F3

    F3 --> G[3D U-Net supervised training\nDiceCE loss]
    G --> H[Validation/full-volume metrics via sliding-window inference]
    H --> I[Artifacts: checkpoints, history.csv, tensorboard, debug_patches]

    C --> J[train_monai_ssl.py (optional pretraining branch)]
    J --> J1[Patch-based masked reconstruction SSL on ssl_pretrain split]
    J1 --> J2[SSL checkpoint]
    J2 --> E1
```
