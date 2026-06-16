# Wittenberg Visual Identity Guide

## 1. Brand Overview

**Wittenberg** is a scientific and educational web app focused on Chladni patterns, acoustic vibration, frequency identification, and machine learning.

The visual identity must communicate:

- Scientific precision.
- Acoustic analysis.
- Calm technical confidence.
- Research-oriented seriousness.
- Elegant minimalism.
- Educational clarity.

The app should look like a polished research prototype, suitable for a TDR presentation and future development.

It must not look like a childish experiment, a generic SaaS template, or an entertainment app.

---

## 2. Brand Personality

Wittenberg should feel:

- Precise.
- Analytical.
- Structured.
- Calm.
- Elegant.
- Modern.
- Academic.
- Trustworthy.

The brand should combine physics, visual pattern recognition, and machine learning without becoming visually overloaded.

---

## 3. Color Palette

Use the following palette as the official design system.

### Main Color

```txt
#205f75
```

Use for:

- Primary buttons.
- Active navigation states.
- Main highlights.
- Important headings.
- Key interface elements.
- Graph or frequency visualizations.

Meaning:

- Scientific depth.
- Acoustic identity.
- Stability.
- Precision.

### Secondary Color

```txt
#123e4d
```

Use for:

- Dark navigation areas.
- Footer.
- Hero contrast sections.
- Strong text emphasis.
- Deep backgrounds.
- High-contrast UI zones.

Meaning:

- Depth.
- Technical seriousness.
- Contrast.
- Structure.

### Background 1

```txt
#F4F8F9
```

Use for:

- Main page background.
- Large layout surfaces.
- Clean scientific canvas.

Meaning:

- Clarity.
- Air.
- Neutrality.
- Laboratory-like cleanliness.

### Background 2

```txt
#DCE9ED
```

Use for:

- Cards.
- Secondary sections.
- Panels.
- Upload areas.
- Empty states.
- Soft separators.

Meaning:

- Subtle structure.
- Low-contrast technical environment.
- Visual calm.

### Accent Color

```txt
#CFA96A
```

Use sparingly for:

- Key callouts.
- Frequency results.
- Confidence badges.
- Small icons.
- Graph markers.
- Scientific annotations.
- Important micro-interactions.

Meaning:

- Precision highlight.
- Warm academic detail.
- Premium research accent.

Do not overuse this color. It should act as a controlled highlight, not as a dominant brand color.

---

## 4. Suggested Semantic Tokens

When creating styles, define semantic tokens instead of scattering raw hex values.

Recommended tokens:

```css
--color-primary: #205f75;
--color-secondary: #123e4d;
--color-background: #F4F8F9;
--color-surface: #DCE9ED;
--color-accent: #CFA96A;
--color-text: #102F3A;
--color-muted: #5F7780;
--color-border: rgba(18, 62, 77, 0.16);
--color-white: #FFFFFF;
```

Use tokens consistently across the app.

---

## 5. Color Usage Rules

General hierarchy:

- Use `#F4F8F9` as the dominant background.
- Use `#DCE9ED` for structured surfaces and cards.
- Use `#205f75` for actions and primary identity.
- Use `#123e4d` for dark contrast and serious framing.
- Use `#CFA96A` only for important accents.

Accessibility:

- Ensure sufficient contrast for all text.
- Do not place low-contrast teal text on blue-gray backgrounds without checking readability.
- Accent gold should not be used for long text.
- Primary buttons should have clear hover, focus, and disabled states.

---

## 6. Typography Direction

The typography should feel modern, scientific, and readable.

Recommended style:

- Sans-serif typeface.
- Clear hierarchy.
- Large, confident headings.
- Moderate line height for readability.
- Avoid decorative fonts.
- Avoid overly futuristic fonts.

If using system fonts:

```css
font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
```

Suggested typographic mood:

- Headings: precise, slightly condensed if available, confident.
- Body: readable and calm.
- Labels: small, uppercase or medium-weight when appropriate.
- Data values: clear and numeric-friendly.

---

## 7. Logo System

The project may include three SVG assets:

- Isotype.
- Imagotype.
- Logotype.

Use them as follows:

### Isotype

Use for:

- Favicon.
- Small header mark.
- Loading states.
- Empty states.
- App icon.

### Imagotype

Use for:

- Main header.
- Landing page hero.
- Footer.
- About page.

### Logotype

Use for:

- Text-based brand lockups.
- Footer or simple navigation.
- Situations where the symbol is not needed.

If SVG colors do not match the palette, adapt them using CSS when technically appropriate.

Preferred SVG strategy:

- Use `currentColor` where possible.
- Apply color through CSS classes.
- Use the primary or secondary color for standard logo use.
- Use white or near-white logo versions on dark backgrounds.
- Use the accent color only for small details, not the full logo unless visually justified.

The logo must remain clean, legible, and aligned with the scientific identity.

---

## 8. Visual Motifs

The visual system may use subtle motifs inspired by:

- Chladni lines.
- Nodal structures.
- Waveforms.
- Frequency scales.
- Scientific diagrams.
- Modal vibration patterns.
- Grid systems.
- Image analysis overlays.

These motifs should be abstract, subtle, and geometric.

They can appear in:

- Hero background.
- Empty states.
- Cards.
- Analyzer preview states.
- Section dividers.
- Data visualization elements.

Avoid decorative visuals that do not relate to acoustics, vibration, physics, or image analysis.

---

## 9. Layout Principles

The layout should be:

- Spacious.
- Structured.
- Responsive.
- Grid-based.
- Calm.
- Easy to scan.

Preferred style:

- Rounded cards with subtle borders.
- Soft shadows only when useful.
- Clear section rhythm.
- Strong but not aggressive hero section.
- Data panels that feel analytical.
- Visual balance between text and scientific UI components.

Avoid:

- Dense clutter.
- Excessive gradients.
- Random decorative blobs.
- Heavy shadows.
- Neon effects.
- Generic corporate illustrations.

---

## 10. Component Style

### Buttons

Primary buttons:

- Background: `#205f75`.
- Text: white.
- Hover: slightly darker or deeper.
- Focus: clear visible outline.

Secondary buttons:

- Border: primary or secondary color.
- Text: primary or secondary color.
- Background: transparent or very light.

Accent buttons:

- Use only for very important actions.
- Accent should usually appear as a small detail rather than a full button background.

### Cards

Cards should use:

- White or `#DCE9ED` surfaces.
- Subtle borders.
- Rounded corners.
- Clear headings.
- Controlled spacing.

### Badges

Badges can use:

- Primary color for active scientific states.
- Accent color for important data highlights.
- Muted tones for prototype or “coming soon” states.

### Forms and Inputs

Inputs should be:

- Clear.
- Accessible.
- Calm.
- High contrast.
- Labelled.

Upload zones may use `#DCE9ED` with dashed borders and clear instructions.

### Result Cards

Prediction result cards should highlight:

- Estimated frequency.
- Confidence score.
- Model state.
- Interpretation.

The frequency value may use the accent color as a controlled highlight.

---

## 11. Analyzer Visual Style

The analyzer page is the core product experience.

It should feel like a scientific instrument interface, but remain simple.

Visual elements may include:

- Image preview panel.
- Upload/camera controls.
- Analysis status.
- Frequency result card.
- Confidence indicator.
- Model status badge.
- Notes about prototype limitations.

The analyzer must clearly distinguish between:

- Image ready.
- Analysis running.
- Result available.
- Error state.
- Mock prediction / no model connected.

Avoid making the mock result look like a validated scientific measurement.

---

## 12. Data Visualization Style

If charts, graphs, or frequency scales are used:

- Keep them minimal.
- Use primary and secondary colors.
- Use accent only for the key result marker.
- Avoid rainbow color scales unless scientifically necessary.
- Use clear labels.
- Prioritize readability over decoration.

Possible visualizations:

- Frequency scale.
- Confidence bar.
- Dataset distribution by frequency.
- Training/validation/test split.
- Image class cards.

---

## 13. Motion and Interaction

Animations should be restrained.

Acceptable uses:

- Soft hover transitions.
- Subtle loading indicators.
- Gentle card entrance animations.
- Smooth page transitions if already supported.

Avoid:

- Bouncy animations.
- Excessive motion.
- Distracting effects.
- Decorative animation unrelated to function.

Motion should support clarity, not entertainment.

---

## 14. Iconography

Use icons only when they improve comprehension.

Preferred icon themes:

- Camera.
- Upload.
- Waveform.
- Frequency.
- Plate/grid.
- Dataset.
- Neural network.
- Warning/info.
- Analysis.

Icons should be line-based, simple, and consistent.

Use primary, secondary, muted, or accent color depending on semantic importance.

---

## 15. French UI Tone

All visible UI copy must be in French.

Preferred tone:

- Scientific.
- Clear.
- Calm.
- Accurate.
- Educational.

Examples:

```txt
Identifier une fréquence sonore à partir d’un motif de Chladni.
```

```txt
Capturez ou importez une image du motif formé sur la plaque vibrante.
```

```txt
Fréquence estimée
```

```txt
Confiance du modèle
```

```txt
Aucun modèle entraîné n’est encore connecté à cette version.
```

```txt
Cette estimation est une simulation destinée à valider l’interface.
```

Avoid exaggerated claims such as:

```txt
Détection parfaite de toutes les fréquences.
```

```txt
Notre IA reconnaît instantanément n’importe quel son.
```

---

## 16. Page-Specific Visual Direction

### Home Page

Should feel like:

- Scientific introduction.
- Elegant research landing page.
- Clear project explanation.

Use:

- Strong hero.
- Abstract Chladni background.
- Short explanation cards.
- Call-to-action toward analyzer.

### Analyzer Page

Should feel like:

- Core scientific tool.
- Calm instrument panel.
- Focused workflow.

Use:

- Large upload/camera area.
- Clear preview.
- Result panel.
- Model status notices.

### Dataset Page

Should feel like:

- Research preparation area.
- Dataset management concept.
- Structured and analytical.

Use:

- Tables.
- Frequency class cards.
- Placeholder statistics.
- Dataset quality notes.

### Experiment Page

Should feel like:

- Laboratory documentation.
- Physical setup explanation.
- Educational scientific guide.

Use:

- Setup cards.
- Process steps.
- Controlled variable notes.

### About / Research Page

Should feel like:

- Academic explanation.
- TDR context.
- Project rationale.

Use:

- Clean sections.
- Research objectives.
- Limitations.
- Educational applications.

---

## 17. Things to Avoid Visually

Avoid:

- Bright neon colors.
- Random gradients unrelated to the palette.
- Overly playful illustrations.
- Cartoon visuals.
- Generic SaaS blue-purple styles.
- Excessive gold accents.
- Low-contrast text.
- Crowded layouts.
- Fake dashboards with meaningless metrics.
- Visual claims that imply a finished validated AI if the model is still a prototype.

---

## 18. Implementation Notes for Claude Code

Before implementing visual changes, Claude Code should read this file and preserve the identity system.

When creating new components:

- Use the official palette.
- Keep French UI copy.
- Preserve the scientific tone.
- Prefer reusable design tokens.
- Use consistent card, button, badge, and layout patterns.
- Ensure responsive behavior.
- Ensure accessible contrast.

When uncertain, choose the calmer, more scientific, more precise option.

---

## 19. Summary

Wittenberg’s visual identity is a precise, calm, elegant scientific interface built around Chladni patterns, acoustics, and machine learning.

The design should always preserve:

- Main color: `#205f75`.
- Secondary color: `#123e4d`.
- Background 1: `#F4F8F9`.
- Background 2: `#DCE9ED`.
- Accent: `#CFA96A`.
- French UI language.
- Serious educational tone.
- Modular scientific interface.
- Honest prototype communication.
