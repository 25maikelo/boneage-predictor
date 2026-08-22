# Comentario 16 — Mapas de saliencia sin reproducibilidad

> Decisión: **no se generan ejemplos nuevos**. La saliencia es puramente ilustrativa — muestra que la atención del gradiente del modelo cae dentro de los segmentos anatómicos entrenados, sin pretender relevancia clínica ni anatómica. La respuesta al comentario es la alternativa que el propio revisor ofrece: reforzar explícitamente esa limitación en vez de ampliar la muestra.

## Lo que ya está bien en el manuscrito (no hace falta tocarlo)

El párrafo actual de la Figura 7 (Sección 3.2) ya es razonablemente cauteloso:

> *"These maps highlight the image regions with the highest gradient magnitude for each segment model's prediction, offering a qualitative, illustrative view of the system's inference process rather than a quantitative confirmation of anatomical coherence."*

Esto ya evita afirmar "evidencia de coherencia anatómica" — el problema no es esa frase, es que (a) faltan detalles metodológicos exactos que pide el comentario, y (b) el lector podría seguir interpretando "illustrative view of the system's inference process" como una afirmación implícita de que el modelo aprendió patrones anatómicamente relevantes, cuando la intención real es más limitada: solo mostrar que el gradiente se concentra dentro de los segmentos, no en el fondo o en artefactos irrelevantes.

## Metodología exacta (extraída del código, sin cómputo nuevo)

De `src/07_validation.py` (`compute_saliency_map`, `overlay_masked`):

- **Tipo de mapa**: gradiente vainilla — `∂(predicción)/∂(imagen de entrada)`, calculado con `tf.GradientTape` sobre la imagen normalizada de cada segmento.
- **Capa objetivo**: ninguna capa intermedia específica — el gradiente se toma directamente respecto al tensor de entrada, no es Grad-CAM ni una variante basada en activaciones de una capa convolucional particular.
- **Reducción a un solo canal**: máximo absoluto del gradiente a través de los 3 canales de color.
- **Normalización**: min-max a [0,1] por imagen individual (`sal / max(sal)`).
- **Umbral de visualización**: solo se superpone el percentil 97 más alto de magnitud de gradiente (`np.percentile(heatmap, 97)`), con mapa de color JET y mezcla alpha=0.4 sobre la radiografía original.
- **Semilla aleatoria**: no aplica — el cálculo es determinístico dado el modelo ya entrenado y la imagen de entrada; no hay ningún componente estocástico en el cómputo del mapa en sí (la única aleatoriedad en todo el pipeline es la selección de qué 3 pacientes mostrar, vía *reservoir sampling* con `random.randint`, ya fijada en las imágenes ya generadas).
- **Fuente de las imágenes**: confirmado en el código — los 3 ejemplos de la Figura 7 se toman del **dataset de validación real de RSNA** (`data/validation/`, el split oficial de 1,425 imágenes, nunca visto en entrenamiento), no del conjunto de entrenamiento. Esto ya es correcto y coincide con lo que dice la leyenda actual ("three randomly selected patients from the validation dataset").

**📝 Agrega esto** justo después del párrafo ya existente de la Figura 7 (Sección 3.2), sin reemplazar nada, solo ampliando:
> "Saliency was computed as a vanilla-gradient map: the gradient of the scalar bone-age prediction was taken directly with respect to each segment model's 112×112×3 input image via automatic differentiation, with no specific intermediate convolutional layer targeted (unlike Grad-CAM-based approaches). The resulting gradient was reduced to a single-channel map by taking the maximum absolute value across color channels and min-max normalized to [0,1] per image; only the top 3% of gradient magnitude values (97th percentile) are overlaid on the radiograph, to visually emphasize the highest-attention regions. This computation is deterministic given the trained model weights and input image; no random seed applies to the saliency map itself. The three patients shown were selected via reservoir sampling from the independent RSNA validation set (n = 1,425), never used during training."

## El punto central: reforzar que es puramente ilustrativo, sin relevancia clínica

**📝 Agrega esto** como frase de cierre del párrafo de la Figura 7, o como nota aparte:
> "These examples are provided solely to illustrate that the model's gradient attention is spatially concentrated within the four segmented anatomical regions used as input, rather than on background or irrelevant image artifacts. They are not intended as evidence of anatomically meaningful, clinically valid, or diagnostically relevant feature learning, and no interpretability claim beyond this basic sanity check is made. A systematic, quantitative evaluation of saliency-based interpretability across a larger, predefined sample is identified as necessary future work."

Esta última frase es importante porque responde directamente a la disyuntiva que plantea el comentario ("extend... or withdraw interpretability claims") — se elige explícitamente la segunda opción, dejando constancia de que no se está afirmando interpretabilidad real, en vez de dejarlo ambiguo como está hoy.
