# Model specification

Input: `float32[N,128,128,1]` from `docs/03-preprocessing-spec.md`. Output: softmax over
`config/classes.json`.

## 1. Two models, on purpose

Framework: **PyTorch**, not Keras — see `docs/adr/0003`. The architecture below is unchanged
by that decision.

**A — `baseline_paper`.** Faithful replication of the architecture in Kumar & Wani (2024) §3.3:
one `Conv2d(1, 64, 3)` + ReLU, `MaxPool2d`, flatten, `Linear(n_classes)` + softmax, Adam,
grayscale + Canny input. Exists so the TDR can compare against the published architecture on our
own plate. Do not "improve" it. Its numbers are a citation, not a target.

Say in the TDR what is true: same architecture and training procedure, our own plate, a
different framework. Do not claim bit-identity with the authors' code.

**B — `wittenberg_v1`.** The model actually shipped.

```
Input (N,1,128,128)                                     # NCHW — PyTorch order
[Conv2d(3, padding=1) → BatchNorm2d → ReLU] ×2 → MaxPool2d(2)   #  32ch,  64×64
[Conv2d(3, padding=1) → BatchNorm2d → ReLU] ×2 → MaxPool2d(2)   #  64ch,  32×32
[Conv2d(3, padding=1) → BatchNorm2d → ReLU] ×2 → MaxPool2d(2)   # 128ch,  16×16
AdaptiveAvgPool2d(1) → flatten                          # GAP, not flatten — see §2
Dropout(0.4)
Linear(128, n_classes)                                  # softmax in the loss
```

Note the input is **NCHW** here while the preprocessing spec produces **NHWC** `(128,128,1)`.
The permute happens in the dataset loader, never inside `preprocess.py` — the preprocessing
contract is framework-agnostic and must stay that way.

**289 134 parameters**, plus 896 non-trainable BatchNorm buffers (`running_mean`/`running_var`
for 448 channels), which PyTorch counts as buffers rather than parameters. Verify before
quoting: `sum(p.numel() for p in model.parameters())` → 289 134.

## 2. Design decisions worth defending in the TDR

- **GlobalAveragePooling instead of Flatten.** `Flatten` after 16×16×128 gives a 32 768-wide
  dense layer — more parameters than we have training images, and it ties the prediction to
  absolute pixel positions. GAP makes the model read nodal-line *density and structure*, which
  is the physics. The classification head shrinks from 458 766 parameters to 1 806 — a 99.6 %
  cut of the head, which is a 61 % cut of the whole model (746 094 → 289 134). Quote whichever
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
| Optimizer | `torch.optim.AdamW`, `lr=1e-3`, `CosineAnnealingLR` to `1e-5` |
| Batch | 32 |
| Epochs | 80, early stopping patience 12 on `val_loss`, restore best weights |
| Device | CUDA. `tasks.py check-gpu` must report the GPU before any run counted in the TDR |
| AMP | off. Mixed precision costs determinism and buys nothing at this model size |
| Loss | `nn.CrossEntropyLoss(label_smoothing=0.05)` (takes logits — no softmax in the model) |
| Seed | 42 for `random`/`numpy`/`torch`, plus `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` |
| Split | `GroupShuffleSplit` on `session_id` (see `docs/02-dataset-spec.md` §2) |

Every run writes `runs/<timestamp>/` containing `config.json` (full resolved config incl.
preprocessing version and the framework/CUDA versions), `metrics.json`, `history.csv`,
`model.pt`, `report/`. It also updates `runs/latest.txt` — a pointer file, not a symlink
(Windows). **Nine variants × 3 seeds = 27 runs** is the F4 budget; on the GPU each is minutes,
so run all of §5.

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

Nine variants, each with 3 seeds — 27 runs. Report mean ± std. Each row is a sentence in the
TDR discussion.

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
| Config hash | `python tasks.py check-model` green (`docs/03-preprocessing-spec.md` §4) |
| Latency | `elapsed_ms` ≤ 200 ms on the local GPU. Above that the model is probably on the CPU |

If a gate fails, the honest move is to report it in the TDR, not to loosen it.

## 7. Serving

There is no export step. The local API loads `runs/<id>/model.pt` directly onto the GPU
(`docs/adr/0004`). ONNX, INT8 quantisation and a served model artefact were consequences of
shipping to a browser; none of them apply now.

Every checkpoint records the resolved `config/preprocess.json` and its SHA-256. The API
asserts the hash at startup and **refuses to start** on mismatch
(`docs/03-preprocessing-spec.md` §4). This is what stops a checkpoint trained under one
pipeline from being served under another.
