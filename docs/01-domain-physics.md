# Domain: Chladni patterns

Read this when writing didactic copy, labelling patterns, or choosing frequencies.
It is the physics the app must teach correctly — not a general acoustics primer.

## 1. The phenomenon

A thin plate driven at one of its resonant frequencies forms a **standing wave**. Points that
never move are **nodes**; points of maximum displacement are **antinodes**. Granular material
scattered on the plate is thrown off the antinodes and accumulates along the **nodal lines**,
making an invisible vibration mode visible. Ernst Chladni described these figures in 1787.

**Terminology trap present in the reference paper.** Kumar & Wani (2024) §1 writes "Node is
the point where amplitude of vibration is maximum. Antinode is the point where amplitude of
vibration is minimum or zero." That is inverted. The correct definitions are the ones above,
and they are also what the paper's own figures show. Do not propagate the error into the app
or the TDR — and mention the discrepancy in the discussion; catching it is a point in the
project's favour.

## 2. What determines the pattern

For a plate, the mode shape depends on:

- **Geometry** — side length and thickness. Square and circular plates give different families.
- **Material** — Young's modulus, Poisson ratio, density. Ours is **1 mm black-lacquered steel,
  180 × 180 mm**: typical carbon steel is E ≈ 200 GPa, ν ≈ 0.29, ρ ≈ 7850 kg/m³ — textbook
  values, not measured on our plate, and the lacquer coat is not in them.
- **Boundary conditions** — centre-clamped vs free edges changes everything. Ours is
  centre-driven with free edges; state this explicitly in the TDR methodology.
- **Frequency** — higher frequency → higher mode order → more nodal lines → more complex figure.

The last point is the qualitative relationship the didactic module should land: *more lines
means a faster vibration*. It is monotone in mode order, **not** strictly proportional to
frequency, and the app must not claim proportionality.

## 3. What the model can and cannot do

The CNN maps *pattern → frequency*. This only works because the mapping is near-injective
**for one specific plate**. It is not a general physical law. The same photo taken on a plate
of different size or thickness maps to a different frequency.

State this limit in the UI. It is also the honest answer to research question 6 of the Pla de
treball ("main sources of error and limitations").

Failure modes to expect and describe:

- **Degenerate modes** — a square plate has symmetry, so two distinct modes can share a nodal
  geometry that differs only by orientation. These are genuinely ambiguous from a photo.
- **Neighbouring modes** — modes 40–80 Hz apart can look nearly identical at 128×128.
- **Incomplete patterns** — too little sand, too short a dwell, or a frequency slightly off
  resonance gives a smeared figure that belongs to no class.
- **Superposition** — driving between two modes gives a mixture that is not in the training set.

## 4. Concepts the didactic module must convey

Ordered by the Pla de treball's teaching goal, for participants with no physics background.
Each maps to a pre/post-test question (`docs/06-research-protocol.md`).

| # | Concept | Anchor in the app |
|---|---|---|
| 1 | Sound is a mechanical wave | audible tone plays while the pattern forms |
| 2 | Frequency = how fast it vibrates; pitch is what we hear | slider: pitch and pattern change together |
| 3 | Resonance: some frequencies "fit" the plate, most do not | sweep animation — the pattern only snaps into focus at modes |
| 4 | Standing wave: a wave that does not travel | side-by-side travelling vs standing animation |
| 5 | Node: does not move. Antinode: moves most | overlay labels the sand lines as nodes |
| 6 | Nodal lines are where the sand collects | this is the "aha" — sand goes where nothing moves. The pipeline viewer's Canny step (`docs/05-webapp-spec.md` §8) shows it happening: everything but the nodal lines disappears |
| 7 | Higher frequency → more complex figure | gallery of the project's own modes, ordered |

Write these in Catalan, using `docs/07-glossary-ca-en.md`. Never introduce a term in the UI
before the concept above it has been shown.

## 5. Sources already vetted in the Pla de treball

Kumar & Wani (2024) OJA 12(1) 1–16 · Kumar, Chary & Wani (2020) SAE 2020-28-0320 ·
Zhu & Hu (2012) SPAWDA · Latifi, Wijaya & Zhou (2017) MARSS · Borković et al. (2014) ·
Igea & Cicirello (2018) JPCS 1106 012031 · Jenny (2001) *Cymatics* ·
Marvin (1996) *Meteoritics & Planetary Science* 31, 545–588.

Pedagogy: Vygotsky (1978) · Chand (2023) · Bezen & Bayrak (2020) · Planinic et al. (2024) ·
Richland, Kornell & Kao (2009) · Latimier et al. (2019) · Aduo et al. (2025).

Use the Mendeley library as the source of truth for citations; do not re-derive them.
