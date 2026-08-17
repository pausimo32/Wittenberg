# Glossary CA ↔ EN

Authoritative. UI strings, TDR prose and code identifiers all resolve through this table.
Never invent a Catalan physics term — if it is missing here, add it here first.

| English (code) | Català (UI / TDR) | Castellano | Note |
|---|---|---|---|
| standing wave | ona estacionària | onda estacionaria | not "ona parada" |
| travelling wave | ona viatgera | onda viajera | |
| node | node | nodo | plural: **nodes** |
| antinode | antinode | antinodo | plural: antinodes |
| nodal line | línia nodal | línea nodal | what the sand draws |
| resonance | ressonància | resonancia | double s |
| resonant frequency | freqüència de ressonància | frecuencia de resonancia | |
| mode / mode shape | mode de vibració | modo de vibración | not "forma modal" |
| mode order | ordre del mode | orden del modo | |
| amplitude | amplitud | amplitud | |
| wavelength | longitud d'ona | longitud de onda | |
| pitch | to | tono | perceptual |
| frequency | freqüència | frecuencia | dieresis on the ü |
| plate | placa | placa | |
| granular material | material granular | material granular | |
| sand / semolina / salt | sorra / sèmola / sal | arena / sémola / sal | |
| oscillator | oscil·lador | oscilador | **punt volat**: l·l |
| signal generator | generador de senyal | generador de señal | |
| sweep | escombrat | barrido | frequency sweep = escombrat de freqüències |
| dwell time | temps d'estabilització | tiempo de estabilización | |
| dataset | conjunt de dades | conjunto de datos | "dataset" acceptable in technical text |
| training / validation / test set | conjunt d'entrenament / validació / prova | — | |
| label | etiqueta | etiqueta | |
| labelling | etiquetatge | etiquetado | |
| preprocessing | preprocessament | preprocesamiento | |
| grayscale | escala de grisos | escala de grises | |
| edge detection | detecció de contorns | detección de bordes | |
| threshold | llindar | umbral | |
| convolutional neural network | xarxa neuronal convolucional | red neuronal convolucional | XNC / CNN |
| layer | capa | capa | |
| supervised learning | aprenentatge supervisat | aprendizaje supervisado | |
| overfitting | sobreajust | sobreajuste | |
| accuracy | exactitud | exactitud | **not** "precisió" — see below |
| precision | precisió | precisión | TP/(TP+FP) |
| recall | exhaustivitat | exhaustividad | TP/(TP+FN) |
| confusion matrix | matriu de confusió | matriz de confusión | |
| learning curve | corba d'aprenentatge | curva de aprendizaje | |
| inference | inferència | inferencia | |
| prediction | predicció | predicción | |
| confidence | confiança | confianza | |

**The accuracy/precision trap.** Catalan and Spanish "precisió/precisión" is routinely used
for *accuracy*, but in a work that also reports precision and recall per class that ambiguity
is fatal. Fix the convention: **exactitud** = accuracy (overall), **precisió** = precision
(per class). State the convention once in the TDR methodology and hold it everywhere.

The Pla de treball §5 currently uses "precisió" for accuracy — correct it on the next revision
and note the change.

## UI microcopy — fixed strings

| Key | Català |
|---|---|
| `analyze.cta` | Analitza una imatge |
| `analyze.result.high` | Probablement **{freq} Hz** |
| `analyze.result.mid` | Podria ser **{freq1} Hz** o **{freq2} Hz** |
| `analyze.result.low` | No ho sé prou |
| `analyze.result.low.hint` | Prova-ho amb més sorra, espera uns segons més o comprova que la freqüència sigui de ressonància. |
| `privacy.notice` | La imatge no surt del teu dispositiu. Tot el càlcul es fa aquí. |
| `learn.cta` | Comença |
| `error.model` | No s'ha pogut carregar el model. |
