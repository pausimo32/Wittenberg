# ADR 0003 — PyTorch instead of TensorFlow/Keras for training

- **Date:** 2026-08-17
- **Status:** accepted. The export/ONNX consequences below are themselves superseded by ADR 0004
- **Trigger:** the team develops on native Windows with an NVIDIA GPU

## Context

The reference study (Kumar & Wani 2024) uses TensorFlow/Keras. The original plan followed it.

Then the environment was fixed: **native Windows**, **NVIDIA GPU**, no WSL2. TensorFlow's own
install page states:

> TensorFlow 2.10 was the last TensorFlow release that supported GPU on native-Windows.
> Starting with TensorFlow 2.11, you will need to install TensorFlow in WSL2, or install
> `tensorflow` or `tensorflow-cpu` and, optionally, try the TensorFlow-DirectML-Plugin.

TF 2.10 is a dead end: it caps Python at 3.10 and needs CUDA 11.2 / cuDNN 8.1. DirectML is a
side plugin with partial op coverage. So on this environment the real options are:

**A. TF ≥ 2.11 on CPU.** Keeps framework parity with the paper. Wastes the GPU. F4 is
9 ablation
variants × 3 seeds = 27 training runs; on CPU that is the difference between an afternoon and
several days, and it will push the team to cut ablations — which are the results chapter.

**B. TF 2.10 pinned.** Gets the GPU, at the cost of an obsolete Python and CUDA stack that
neither `tf2onnx` nor `onnxruntime` current versions are tested against. Trades a known problem
for an unknown one.

**C. WSL2 for training only.** Works, but splits the project across two filesystems and two
toolchains, on a team of three who all have to reproduce it. Rejected with the environment choice.

**D. PyTorch.** First-class native Windows CUDA wheels. `torch.onnx.export` is built into the
framework rather than a third-party converter. `torch.use_deterministic_algorithms(True)` gives
the determinism the project requires.

## Decision

**Option D — PyTorch.**

## Consequences

- `docs/04-model-spec.md` is expressed in PyTorch. The **architecture is unchanged**: same
  layers, same 289 134 parameters, same hyperparameters, same acceptance gates.
- ~~F5 exports with `torch.onnx.export` instead of `tf2onnx`.~~ Superseded by ADR 0004: there is
  no export step at all. The local API loads `model.pt` directly onto the GPU.
- `docs/03-preprocessing-spec.md` is **completely unaffected**: preprocessing is OpenCV and never
  depended on the training framework. This is the payoff of having kept the preprocessing
  contract separate from the model — and it held again through ADR 0004.
- `requirements.txt` drops `tensorflow` and `tf2onnx`, adds `torch` and `torchvision` (CUDA
  build, from the PyTorch index URL). `onnx` was added here and removed again by ADR 0004.

## What this means for the TDR

`baseline_paper` remains a faithful replication of the **architecture** published by Kumar &
Wani — one `Conv2D(64, 3×3)` + ReLU + max-pool + flatten + dense + softmax, Adam, grayscale +
Canny input — implemented in PyTorch rather than Keras.

State this in the methodology. The framework is not a scientific variable here: for a
single-convolution model trained with Adam, Keras and PyTorch compute the same function and
the same gradients. What would compromise the replication is changing the architecture,
the input representation or the optimiser, and none of those change.

Do not claim the replication is bit-identical to the authors' code. Claim what is true: the
same architecture and training procedure, on our own plate, in a different framework.
