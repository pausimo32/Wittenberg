# Model specification

Input: `float32[N,128,128,1]` from `docs/03-preprocessing-spec.md`. Output: softmax over
`config/classes.json`.

## 1. Two models, on purpose

**A — `baseline_paper`.** Faithful replication of Kumar & Wani (2024) §3.3: one `Conv2D(64,
3×3)` + ReLU, `MaxPooling2D`, `Flatten`, `Dense(n_classes)` + softmax, Adam, grayscale + Canny
input. Exists so the TDR can compare against the published architecture on our own plate.
Do not "improve" it. Its numbers are a citation, not a target.

**B — `wittenberg_v1`.** The model actually shipped.

```
Input(128,128,1)
[Conv2D(32,3,pad=same) → BN → ReLU] ×2 → MaxPool(2)          # 64×64
[Conv2D(64,3,pad=same) → BN → ReLU] ×2 → MaxPool(2)          # 32×32
[Conv2D(128,3,pad=same) → BN → ReLU] ×2 → MaxPool(2)         # 16×16
GlobalAveragePooling2D                                        # not Flatten — see §2
Dropout(0.4)
Dense(n_classes, softmax)
```

290 030 parameters (289 134 trainable). Fits in a well-under-2 MB quantised ONNX file,
which matters for the web app. Verify with `model.summary()` before quoting the figure.

## 2. Design decisions worth defending in the TDR

- **GlobalAveragePooling instead of Flatten.** `Flatten` after 16×16×128 gives a 32 768-wide
  dense layer — more parameters than we have training images, and it ties the prediction to
  absolute pixel positions. GAP makes the model read nodal-line *density and structure*, which
  is the physics. The classification head shrinks from 458 766 parameters to 1 806 — a 99.6 %
  cut of the head, which is a 61 % cut of the whole model (746 990 → 290 030). Quote whichever
  of the two you mean; do not quote a single "~95 %" figure, which matches neither.
- **BatchNorm before ReLU**, standard ordering; keeps training stable at batch 32 on a small set.
- **No pretrained backbone.** ImageNet features are tuned for natural-image texture and colour;
  our input is a binary edge map. A small model trained from scratch is both better here and
  far easier to justify in a high-school research document.
- **Class weights** if any class has < 70 % of the median class count. Prefer collecting more
  images to reweighting.

## 3. Training

| Setting | Value |
|---|---|
| Optimizer | Adam, `lr=1e-3`, cosine decay to `1e-5` |
| Batch | 32 |
| Epochs | 80, `EarlyStopping(patience=12, restore_best_weights=True)` on `val_loss` |
| Loss | categorical cross-entropy, `label_smoothing=0.05` |
| Seed | 42, and `tf.config.experimental.enable_op_determinism()` |
| Split | `GroupShuffleSplit` on `session_id` (see `docs/02-dataset-spec.md` §2) |

Every run writes `runs/<timestamp>/` containing `config.json` (full resolved config incl.
preprocessing version), `metrics.json`, `history.csv`, `model.keras`, `report/`.

## 4. Evaluation — mandatory outputs

1. Overall accuracy **with a 95 % Wilson interval**, computed on independent captures, not
   frames. Under leave-one-session-out, the independent unit is the *session*, so with 4
   sessions a per-class interval is built on 4 observations, not on the frame count. Report the
   frame-level interval too, labelled as such, but do not present it as the headline.
2. Per-class precision / recall / F1 (`classification_report`).
3. Confusion matrix, absolute and row-normalised, saved as PNG **and** CSV (the CSV is a TDR annex).
4. Learning curves (loss/accuracy, train vs val).
5. **Frequency-distance error analysis** — for each misclassification, `|f_true − f_pred|`.
   The research question is whether confusions cluster among neighbouring modes with similar
   nodal geometry. A confusion between 740 Hz and 760 Hz means something very different from
   one between 740 Hz and 3200 Hz. This plot answers research question 6 of the Pla de treball.
6. **Grad-CAM** on 3 correct and 3 incorrect samples per class. Verifies the model looks at
   nodal lines and not at a scratch on the plate or the rig shadow. Non-negotiable — without it
   there is no evidence the model learned the physics.

## 5. Ablations — these ARE the results chapter

Run each with 3 seeds, report mean ± std. Each row is a sentence in the TDR discussion.

| # | Variant | Question it answers |
|---|---|---|
| A1 | grayscale only, no Canny | Does Canny help, as the reference study claims? |
| A2 | grayscale + Canny | baseline |
| A3 | no CLAHE | Is our lighting-normalisation deviation justified? |
| A4 | random split vs session split | How much does frame leakage inflate accuracy? |
| A5 | 64×64 vs 128×128 vs 224×224 | How much spatial detail do nodal lines need? |
| A6 | 25 % / 50 % / 100 % of training data | Learning curve — would more capture sessions help? |
| A7 | `baseline_paper` vs `wittenberg_v1` | Does the added depth earn its cost? Note this varies depth, BN and head at once — it does not isolate any of them. |
| A8 | sand vs semolina vs salt | Which granular material gives the most separable patterns? |
| A9 | `wittenberg_v1` with a `Flatten` head vs the `GAP` head | Isolates the head choice, which A7 confounds |

## 6. Acceptance gates — ship or do not ship

| Gate | Threshold |
|---|---|
| Session-wise test accuracy | ≥ 80 % (the reference study's floor across all 14 classes) |
| Worst-class recall | ≥ 60 % — no class may be effectively unrecognisable |
| Grad-CAM | mass concentrated on nodal lines in ≥ 80 % of inspected samples |
| Parity | `npm run parity` green (`docs/03-preprocessing-spec.md` §4) |
| Quantised ONNX vs Keras | max softmax difference ≤ 1e-2, argmax agreement 100 % |
| Model file | ≤ 4 MB, first inference ≤ 1.5 s on a mid-range phone |

If a gate fails, the honest move is to report it in the TDR, not to loosen it.

## 7. Export

`make export`: Keras → ONNX (`tf2onnx`, opset 17) → `onnxruntime.quantization` dynamic INT8 →
`web/public/model/wittenberg_v1.onnx` + `classes.json` + `preprocess.json`.

The exported bundle carries the preprocessing version. The web app refuses to run if the
bundle's preprocessing version does not match `config/preprocess.json`. This is what stops a
silently-retrained model from being served with mismatched preprocessing.
