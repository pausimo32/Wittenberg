# Recreación de Diseño Web

## Flujo de Trabajo

Cuando el usuario proporcione una imagen de referencia (captura de pantalla) y opcionalmente algunas clases CSS o notas de estilo:

1. **Generar** un único archivo `index.html` usando Tailwind CSS (vía CDN). Incluye todo el contenido en línea — sin archivos externos a menos que se solicite.
2. **Capturar** la página renderizada usando Puppeteer (`npx puppeteer screenshot index.html --fullpage` o equivalente). Si la página tiene secciones distintas, captúralas individualmente también.
3. **Comparar** tu captura de pantalla con la imagen de referencia. Busca discrepancias en:
   - Espaciado y relleno (medir en px)
   - Tamaños de fuente, grosores y alturas de línea
   - Colores (valores hex exactos)
   - Alineación y posicionamiento
   - Radios de borde, sombras y efectos
   - Comportamiento responsive
   - Tamaño y ubicación de imágenes/iconos
4. **Corregir** cada discrepancia encontrada. Edita el código HTML/Tailwind.
5. **Volver a capturar** y comparar de nuevo.
6. **Repetir** los pasos 3–5 hasta que el resultado esté dentro de ~2–3px de la referencia en todas partes.

NO te detengas después de una sola pasada. Realiza siempre al menos 2 rondas de comparación. Solo detente cuando el usuario lo diga o cuando no queden diferencias visibles.

## Valores Técnicos Predeterminados

- Usar Tailwind CSS vía CDN (`<script src="https://cdn.tailwindcss.com"></script>`)
- Usar imágenes de marcador de posición de `https://placehold.co/` cuando no se proporcionen imágenes de origen
- Diseño responsive mobile-first
- Un único archivo `index.html` a menos que el usuario solicite lo contrario

## Reglas

- No añadas características, secciones o contenido que no estén presentes en la imagen de referencia
- Iguala la referencia exactamente — no "mejores" el diseño
- Si el usuario proporciona clases CSS o tokens de estilo, úsalos literalmente
- Mantén el código limpio pero no lo abstraigas demasiado — las clases de Tailwind en línea están bien
- Al comparar capturas de pantalla, sé específico sobre lo que está mal (p. ej., "el encabezado es de 32px pero la referencia muestra ~24px", "el espacio entre tarjetas es de 16px pero debería ser de 24px")
-  Antes de implementar, modificar o refactorizar cualquier parte de la web app Wittenberg —incluyendo interfaz, arquitectura, copy, componentes, estilos, rutas, lógica de predicción o experiencia de usuario— debes leer y respetar siempre `WITTENBERG_CONTEXT.md` y `WITTENBERG_BRAND_GUIDE.md`, manteniendo la coherencia científica, funcional y visual definida en ambos documentos.