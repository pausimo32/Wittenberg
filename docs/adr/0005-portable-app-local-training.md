# ADR 0005 — Portable CPU app; training on the team's own GPU

- **Date:** 2026-08-18
- **Status:** accepted. Amends ADR 0003 and ADR 0004; neither is reversed.
- **Trigger:** the app must run on any computer the team chooses. Training happens on the team's
  RTX 5070; it is not part of the app.
- **Amended 2026-08-18:** an RTX 5070 (12 GB) with 64 GB system RAM turned out to be available.
  Training moves back in-house; **nothing else in this ADR changes.** The split that matters is
  by workload, not by who owns the hardware.

## Context

Two requirements arrived together, and they pull in opposite directions only if you assume
inference needs a GPU:

1. The app must be runnable on **any** machine the team picks — a school laptop, someone's
   desktop, a borrowed computer. Not deployed publicly, but not tied to one box either.
2. There is no local GPU, so the model must be trained on **rented** compute.

The natural reading is "the app calls a rented GPU service to run the model". That reading is
wrong, and the measurement says so.

## The measurement that settles it

`wittenberg_v1` is a 3-block CNN over a 128×128 single-channel input.

| | |
|---|---|
| Parameters | 288 876 (12 classes) — `9·c_in·c_out + c_out` summed, plus `128·n + n` for the head |
| Checkpoint on disk | **1.17 MB** |
| Inference, 1 image, CPU, 4 threads | **median 11.8 ms**, p95 13.3 ms |
| Inference, 1 image, CPU, 2 threads | median 7.4 ms |
| Preprocessing, 1 image, CPU | ≈ 24 ms |

**End to end on a plain CPU: about 36 ms.** The model is 1.17 MB. There is nothing here that
needs a GPU, a network call, or a rented service.

## Decision

**Split by workload, not by machine.**

- **Inference — always local, always CPU.** `torch.device("cpu")` by default; CUDA is used only
  if it happens to be present. The app carries its own 1.17 MB checkpoint and needs no network
  at all. It runs on any computer with Python, which is exactly requirement 1.
- **Training — the team's RTX 5070, offline, a handful of times.** F4 is a local script. It is
  not part of the app and the app never calls it. Nothing is uploaded anywhere, so ADR 0004's
  privacy property is now absolute rather than merely preserved.

### Is one RTX 5070 enough? Computed, not guessed

| | |
|---|---|
| Weights + grads + Adam states | 4.6 MB |
| Activations kept for backward, batch 32 | 470 MB |
| **Training peak** | **≈ 474 MB of 12 288 MB** — 4 % of the card |
| Whole dataset as uint8 in VRAM | 65 MB |
| Whole dataset as float32 in VRAM | 262 MB |
| One epoch of 4 000 images | ≈ 15 TFLOP |

The card is not the constraint; it is not close to being the constraint. Two consequences:

- **Preload the entire dataset into VRAM once and never touch the disk again.** 65 MB as uint8.
  With a `DataLoader` reading files each epoch, the GPU would idle waiting for Python; indexing a
  resident tensor with a random permutation removes the bottleneck entirely. Expect seconds per
  epoch, so all 27 runs of §5 in well under an hour.
- **The 64 GB of system RAM is irrelevant here.** The whole dataset is 65 MB. Do not let its
  presence justify a bigger batch, a bigger image or a bigger model: none of those are limited
  by memory, and all of them would need a reason from the ablations, not from the hardware.

### The trap that WILL bite: RTX 5070 is Blackwell, sm_120

The RTX 50 series is compute capability **sm_120**. PyTorch builds compiled before it existed —
`cu121`, `cu124`, which is what most pinned requirements still say — contain **no kernels for
this architecture**. They install cleanly, `torch.cuda.is_available()` returns `True`, the device
name prints correctly, and then the first real operation dies with:

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

No driver update fixes it. The build must be **cu128 or newer**.

**Therefore `torch.cuda.is_available()` is not a sufficient check**, and `tasks.py check-device`
must actually execute an operation on the device rather than query availability. A check that
passes and then lets training crash is worse than no check.

## Why not put inference behind a rented endpoint

Every property of that design is worse here:

| | Remote inference endpoint | Local CPU |
|---|---|---|
| Works on any computer | only with internet and a valid key | **yes, offline** |
| Latency | network round trip, tens to hundreds of ms | **12 ms** |
| Cost | per call, forever | **zero** |
| Secrets | an API key inside a repo a TDR will publish | **none** |
| Privacy | images leave the machine | **nothing leaves** |
| Field session risk | a wifi failure kills the session | **none** |

A remote endpoint would buy nothing and cost all of that, for a model that runs in 12 ms on a
laptop. Note this stays true *even though a GPU is now available*: putting the 5070 behind the
app would tie the tool to one machine, which is exactly what requirement 1 forbids.

## Consequences

- `requirements.txt` installs **CPU PyTorch** by default — the app must install on any computer.
  GPU wheels are only for the training machine: `requirements-train.txt`, **cu128 or newer**.
- `python tasks.py check-device` replaces `check-gpu`: it reports the device and, on CPU, states
  the expected latency instead of warning. CPU is the supported configuration, not a fallback.
- ADR 0003's PyTorch decision **stands**, with its motivation shifted: it was chosen because
  TensorFlow could not use a local Windows GPU. ~~Now there is no local GPU at all~~ — per the
  2026-08-18 amendment there is one, the team's RTX 5070, and it trains. PyTorch is still right
  either way: one line switches device, and CPU inference is fast.
- ADR 0004's core property is **untouched and now stronger**: training and inference still call
  the same `src/preprocess.py`. ~~Because that preprocessing happens locally in both cases, the
  cloud only ever sees tensors that the local pipeline produced.~~ Per the 2026-08-18 amendment
  there is no cloud: both run on machines the team owns, and the guard that matters is the
  startup hash below, not who owns the hardware.
- The training script records the resolved `config/preprocess.json` and its hash into the
  checkpoint. The API's startup guard does not care which machine computed the weights — it
  cares that the pipeline matches.

## What to say in the TDR

Say what is true and it reads well: the model is small enough that inference runs in **12 ms on
an ordinary processor**, so the tool is portable, offline and needs no special hardware; the only
step that benefits from a GPU is training, which used the team's own card and whose peak memory
was **4 % of it**. Report the measurements — they are an engineering result, not a limitation,
and "it fits on any laptop" is a stronger claim than "we had a good GPU".
