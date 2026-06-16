# Wittenberg Project Context

## 1. Project Summary

**Wittenberg** is a scientific web app created for a TDR research project about **Chladni patterns**, sound frequencies, wave propagation, resonance, image recognition, and machine learning.

The goal of the project is to design and build a visual tool capable of identifying or estimating the sound frequency that produced a Chladni pattern on a vibrating plate.

The name **Wittenberg** refers to the birthplace of **Ernst Chladni**, the physicist associated with the visual study of vibration patterns on plates.

Wittenberg should be treated as a serious educational and research prototype, not as a toy or casual science demo.

---

## 2. Scientific Background

The project is based on the physical phenomenon of **Chladni patterns**.

When a rigid plate vibrates at certain frequencies, standing waves appear on its surface. Some zones vibrate strongly, while others remain almost still.

Key concepts:

- **Frequency**: number of oscillations per second, measured in hertz (Hz).
- **Amplitude**: intensity or magnitude of vibration.
- **Resonance**: condition in which the plate vibrates strongly at a natural modal frequency.
- **Standing wave**: wave pattern that remains spatially stable over time.
- **Nodes**: areas of the plate with minimal or no vibration.
- **Antinodes / ventres**: areas of stronger vibration.
- **Chladni pattern**: visible pattern formed when granular material accumulates mainly along nodal lines.

The project connects physics, acoustics, image processing, and supervised machine learning.

---

## 3. Physical Experiment

The experimental setup consists of a controlled environment where Chladni patterns are generated and photographed.

Main elements:

- A **square vibrating plate**.
- A vibration source, such as a speaker, mechanical oscillator, or equivalent actuator.
- A sound or frequency generator.
- A granular medium, preferably fine and visually clear, such as fine salt, fine sand, sugar, or similar.
- A camera or visual recording system.
- Controlled lighting.
- A fixed camera position.
- Constant experimental conditions.

The experiment should generate visible patterns by applying known frequencies to the plate. Each generated image must be associated with the frequency that created it.

The project should assume that the physical setup may evolve during development. The software must remain flexible enough to adapt to changes in plate size, material, camera setup, granular medium, and frequency range.

---

## 4. Controlled Conditions

Consistency is essential because the future machine learning model depends on visual regularity.

The following variables should be kept constant whenever images are captured:

- Camera position.
- Camera angle.
- Distance from camera to plate.
- Lighting direction and intensity.
- Quantity of granular material.
- Granular material type.
- Exposure time to vibration.
- Plate material.
- Plate size and geometry.
- Frequency value.
- Signal amplitude.
- Background and surrounding environment.

The app should communicate that inconsistent experimental conditions can reduce classification accuracy.

---

## 5. Machine Learning Goal

The project aims to create a supervised image classification system.

Expected pipeline:

1. Generate Chladni patterns using known frequencies.
2. Capture many images for each frequency.
3. Label every image with its corresponding frequency.
4. Build a structured dataset.
5. Train a supervised machine learning model, preferably a CNN.
6. Evaluate the model with new images.
7. Use the trained model to estimate the frequency from an uploaded or captured image.

Possible model architecture:

- CNN trained on labelled images.
- TensorFlow / Keras during training.
- TensorFlow.js, ONNX, or API inference in the web app.

The first versions of the app may use a mock prediction service. This is acceptable as long as the architecture clearly separates UI from prediction logic.

---

## 6. Dataset Principles

The dataset is central to the project.

Each image should be linked to:

- Frequency in Hz.
- Experimental session.
- Plate type.
- Granular material.
- Lighting setup.
- Camera setup.
- Optional notes about quality or anomalies.

Recommended dataset structure:

```txt
/dataset
  /330Hz
  /440Hz
  /587Hz
  /660Hz
  /880Hz
  /1100Hz
  /1320Hz
  /1760Hz
  /2200Hz
  /2640Hz
  /3080Hz
  /3520Hz
  /3910Hz
```

Suggested placeholder frequency classes:

- 330 Hz
- 440 Hz
- 587 Hz
- 660 Hz
- 880 Hz
- 1100 Hz
- 1320 Hz
- 1760 Hz
- 2200 Hz
- 2640 Hz
- 3080 Hz
- 3520 Hz
- 3910 Hz

These are placeholder values. They must be replaced by the real experimental frequencies once the dataset is collected.

---

## 7. Web App Purpose

The web app is the main interface of the Wittenberg project.

It should allow the user to:

- Understand the scientific purpose of the project.
- Learn how Chladni patterns are generated.
- Capture an image using the camera.
- Upload an image manually.
- Preview the selected image.
- Analyze the pattern.
- Receive a predicted or simulated frequency.
- See a confidence score.
- Understand the limitations of the result.
- Prepare or document a future labelled dataset.

The app should feel like a complete scientific prototype that could be presented as part of a TDR defense.

---

## 8. Target Users

Primary users:

- The student developing the TDR.
- Teachers or evaluators reviewing the project.
- People interested in physics, acoustics, and machine learning.

Secondary users:

- Students learning about waves and resonance.
- Users testing the visual classifier once the model is trained.

The app must be accessible, clear, and visually serious.

---

## 9. Language and Tone

The entire app UI must be written in **French**.

The language should be:

- Clear.
- Scientific.
- Educational.
- Precise.
- Mature.
- Accessible to non-specialists.

Avoid:

- Childish expressions.
- Overly casual language.
- Exaggerated claims.
- Marketing language that sounds unrealistic.
- Saying the AI is accurate if no trained model is connected.

Preferred examples:

- “Identifier une fréquence sonore à partir d’un motif de Chladni.”
- “Capturez ou importez une image du motif formé sur la plaque vibrante.”
- “Le système analysera la structure visuelle du motif et proposera une fréquence estimée.”
- “Fréquence estimée.”
- “Confiance du modèle.”
- “Conditions expérimentales.”
- “Base de données d’images.”
- “Modèle supervisé.”
- “Aucun modèle entraîné n’est encore connecté à cette version.”
- “Cette estimation est une simulation destinée à valider l’interface.”

---

## 10. Core Pages

The app should include, at minimum, the following pages or sections.

### Home / Landing Page

Purpose:

- Present Wittenberg.
- Explain the project.
- Introduce Chladni patterns.
- Explain how sound becomes visible through vibration.
- Guide the user toward the analyzer.

Suggested sections:

- Hero section.
- Project explanation.
- “Comment ça fonctionne.”
- Scientific concepts.
- Machine learning overview.
- Limitations.
- Call to action.

### Analyzer Page

Purpose:

- Main tool for image capture, upload, preview, and prediction.

Required features:

- Camera access.
- Image upload.
- Image preview.
- Analysis button.
- Loading state.
- Result card.
- Predicted frequency.
- Confidence score.
- Warning if prediction is simulated.
- Error states.

The prediction logic must live outside the UI component, for example in `services/predictionService.ts`.

### Dataset / Training Page

Purpose:

- Prepare the future dataset and training workflow.

Content:

- Image capture by known frequency.
- Labelled dataset structure.
- Frequency classes.
- Training / validation / test split.
- Dataset quality requirements.
- Placeholder upload or statistics components.

### Experiment Page

Purpose:

- Explain the physical setup.

Content:

- Square plate.
- Granular material.
- Vibration source.
- Frequency generator.
- Camera.
- Lighting.
- Controlled variables.

### About / Research Page

Purpose:

- Present the TDR as an academic project.

Content:

- Research objective.
- Physics basis.
- Computer science basis.
- Expected limitations.
- Educational applications.

---

## 11. Prediction Architecture

The prediction feature should be implemented in a modular way.

Recommended architecture:

```txt
/src
  /services
    predictionService.ts
  /data
    frequencies.ts
  /components
    ImageUploader.tsx
    CameraCapture.tsx
    PredictionResult.tsx
```

The system should be ready for future integration with:

- TensorFlow.js.
- ONNX Runtime Web.
- Backend API inference.
- Python-trained CNN exported to a web-compatible format.

Expected prediction pipeline:

1. Receive image input.
2. Validate image.
3. Optionally preprocess image.
4. Resize image.
5. Convert to grayscale or normalized tensor.
6. Run inference.
7. Return predicted frequency and confidence.
8. Display interpretation.

If no trained model exists, use a mock prediction service and clearly communicate this in the UI.

---

## 12. Prototype Limits

The app must be honest about its state.

If the real AI model is not connected, the app must not pretend that predictions are scientifically valid.

Use clear messages such as:

- “Aucun modèle entraîné n’est encore connecté.”
- “Résultat simulé pour tester l’interface.”
- “La précision dépendra de la qualité du jeu de données et des conditions expérimentales.”

Expected limitations to mention:

- Similarity between patterns at nearby frequencies.
- Irregular distribution of the granular material.
- Poor lighting.
- Different camera angles.
- Small dataset.
- Plate imperfections.
- Changes in amplitude.
- Vibrations from external sources.

---

## 13. UX Principles

The app must feel:

- Scientific.
- Clean.
- Precise.
- Elegant.
- Calm.
- Structured.
- Reliable.

Interface principles:

- Avoid clutter.
- Use clear hierarchy.
- Make actions obvious.
- Keep the analyzer simple.
- Make limitations visible but not intrusive.
- Provide useful empty states.
- Use accessible labels.
- Ensure responsive behavior.

---

## 14. Development Principles

Follow the repository’s `CLAUDE.md` instructions.

Claude Code should always read this file before implementing or modifying:

- UI copy.
- Pages.
- Components.
- Prediction architecture.
- Dataset features.
- Experiment explanations.
- Branding-related decisions.

Code principles:

- Use TypeScript where possible.
- Keep components modular.
- Keep prediction logic separate from UI.
- Avoid hardcoded content inside complex components when data files would be cleaner.
- Maintain responsive layouts.
- Run available lint, typecheck, and build commands before finishing.

---

## 15. Things to Avoid

Avoid:

- Generic startup-style landing pages.
- Playful or childish science aesthetics.
- Fake claims of AI accuracy.
- Unclear prediction states.
- Overuse of animations.
- Overuse of the accent color.
- Hardcoding the future model directly into UI components.
- Mixing languages in the UI.
- Treating the project as a finished commercial product.
- Ignoring experimental limitations.

---

## 16. Summary for Claude Code

Wittenberg is a French-language scientific web app for a TDR project. It studies Chladni patterns and aims to identify sound frequencies through visual pattern recognition and future machine learning integration.

Always preserve:

- Scientific precision.
- French UI language.
- Serious educational tone.
- Modular prediction architecture.
- Honest prototype limitations.
- Consistent visual identity.
- Clean, accessible, responsive design.
