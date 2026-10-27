# Revisión de literatura: mejora de calidad de imagen radiográfica

> Complementa [`diagnostico.md`](diagnostico.md) (Etapa 4 de
> [`fase_segmentacion.md`](../../../planning/doctorado/fase_segmentacion.md)). Búsqueda bibliográfica sobre
> técnicas de mejora de calidad en radiografías, para contrastar las propuestas de
> `diagnostico.md` contra lo reportado en la literatura antes de implementarlas.

## 1. Mejora de contraste en radiografías: alternativas a CLAHE simple

- **G-CLAHE (2024)**: variante que combina ecualización global e histograma adaptativo limitado,
  buscando preservar contraste local y global a la vez; pensada específicamente para
  radiografías.
- **N-CLAHE**: agrega un paso de log-normalización antes de CLAHE; reportado con mejoras claras
  en radiografías de tórax frente a ecualización simple, unsharp masking y CLAHE puro.
- **Pipeline híbrido**: filtro de mediana (reducción de ruido) + unsharp masking (realce de
  bordes) + ecualización adaptativa de histograma, en vez de solo CLAHE.
- Ningún resultado de 2023-2024 compara CLAHE directamente contra mejora basada en aprendizaje
  profundo para radiografías; no se investiga esa dirección aquí por ser desproporcionada para
  el problema actual (serviría para un modelo de mejora de imagen, no para un paso de
  preprocesamiento determinístico).

## 2. Preprocesamiento específico para bone age / radiografías de mano

- El patrón más citado es **segmentar la mano, normalizar contraste, detectar puntos clave y
  registrar las imágenes** antes de entrenar. Confirma que el recorte/alineación (lo que hace
  nuestro script 02) es un paso estándar y valioso, no superfluo.
- Un paper nota que los **métodos simples de remoción de fondo no dan resultados
  satisfactorios**, y en su lugar usa un proceso iterativo de "minería positiva" combinando
  etiquetado manual con procesamiento automático. Relevante: confirma que el problema que
  encontramos (Otsu simple fallando en una fracción del dataset) es un problema conocido en la
  literatura, no una peculiaridad de nuestro pipeline.
- **Contrapeso importante**: un estudio (Ewha Medical Journal, 2024) comparó modelos entrenados
  con y sin conversión de contraste como preprocesamiento, y **no encontró diferencia
  significativa** en el desempeño de la predicción de edad ósea. Esto no invalida mejorar la
  calidad de imagen por otras razones (ej. que el recorte funcione), pero sí es una señal de que
  no hay que sobre-invertir en ajustar CLAHE esperando una ganancia grande en el modelo final.
- Un paper de 2024 reporta mejoras al **enfocar el input en regiones pequeñas y de mayor
  calidad** de la radiografía en vez de la mano completa, consistente con nuestro propio hallazgo
  de que el modo `cropped` de segmentación supera a `spatial` en el pipeline de fusión.

## 3. Alternativas a Otsu para fondo con iluminación/contraste desparejo

Esto es lo más directamente aplicable al hallazgo 2 de `diagnostico.md` (contraste global bajo
rompe el umbral de Otsu):

- **Corrección de iluminación por sustracción de fondo** ("rolling ball" / top-hat morfológico):
  se estima el fondo localmente con una apertura morfológica de kernel grande (más grande que la
  estructura de interés) y se resta de la imagen original antes de umbralizar. Es el método
  estándar para iluminación despareja en microscopía y puede implementarse directamente con
  `cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_grande)`. Limitación documentada: puede introducir
  artefactos si el kernel no está bien calibrado al tamaño del objeto de interés.
- **Umbralización difusa (fuzzy thresholding)**: un estudio comparando Otsu, k-means y
  umbralización difusa para fondos de radiografías encontró que la difusa tuvo el menor error
  absoluto de los tres.
- **Umbral local adaptativo + bordes de Canny adaptativos**: un paper de segmentación de huesos
  carpales combina ambos (con operación XOR) y reporta Dice medio de 0.976 en 30 imágenes, muy
  por encima de lo que logra un único umbral global.
- **Umbral local por mediana**: un estudio en radiografías dentales encontró que el umbral local
  por mediana da mejores resultados que la media o que Otsu simple.
- **Negativo documentado**: aplicar Otsu por regiones (dividir la imagen en ~100 ROIs) fue
  reportado como lento y con resultados "feos" por quien lo probó; no se recomienda sin más
  evidencia.

## 4. Precedente de BoneXpert: rechazar en vez de forzar

BoneXpert (herramienta comercial de referencia para bone age automatizado) **no intenta reparar
radiografías de mala calidad**: las rechaza explícitamente cuando no se pueden reconocer
suficientes huesos, o cuando la pose/calidad de imagen es inadecuada. Las tasas de rechazo
reportadas en cohortes reales son de **2.3%** (causas: nitidez, forma ósea anormal, pose o
calidad pobre) y de 2 pacientes en otra cohorte. Esto es del mismo orden de magnitud que la tasa
de activación de nuestra salvaguarda de descarte (5.25% sobre 800 imágenes, hallazgo 1 de
`diagnostico.md`), lo que sugiere que **una tasa de descarte/rechazo de pocos puntos porcentuales
no es en sí misma un problema a eliminar por completo**: es un patrón esperado y aceptado en la
práctica clínica automatizada, siempre que las imágenes rechazadas queden identificadas y no se
fuercen con un recorte degenerado silencioso (que es lo que hace hoy `02_frame_and_zoom.py`: cae
de vuelta a la imagen completa sin avisar en ningún log agregado).

## 5. Reducción de ruido (denoising)

Investigado pero de **baja prioridad para este proyecto**: la literatura de denoising de rayos X
se enfoca en ruido cuántico (Poisson) a nivel del detector, lo cual requiere acceso a la imagen
cruda del sensor o a la dosis de exposición. El dataset RSNA que usamos ya son PNG de 8 bits
post-procesados por cada institución de origen, sin metadata de adquisición ni acceso a la señal
cruda, así que las técnicas de esta literatura no son aplicables directamente sin esa
información.

## Propuestas resultantes (priorizadas)

1. **Reemplazar la propuesta de `diagnostico.md`** ("normalizar contraste antes de Otsu" via
   `equalizeHist`, ya en validación) **por sustracción de fondo morfológica** (top-hat/rolling
   ball), que es el método documentado específicamente para el problema de iluminación/contraste
   despareja que diagnosticamos, en vez de una ecualización global genérica. Se evalúa con la
   misma metodología sin modelo (fill ratio) ya usada.
2. Agregar un **log explícito de las imágenes que caen en cualquier salvaguarda** de
   `02_frame_and_zoom.py` (hoy el fallback es silencioso), siguiendo el precedente de BoneXpert
   de identificar y no forzar los casos límite, en vez de asumir que el recorte siempre debe
   producir algo.
3. Confirmar `clipLimit=4.0` (ya en validación) como mejora incremental de CLAHE, sin esperar que
   por sí sola cambie mucho el desempeño del modelo final, dado el hallazgo de la sección 2.
4. No se investiga denoising ni mejora basada en aprendizaje profundo por ahora: desproporcionado
   frente a los datos disponibles (ver sección 5) y frente al hallazgo de la sección 2 sobre el
   impacto limitado del contraste en el modelo final.

## Fuentes consultadas

- [Image enhancement on digital X-ray images using N-CLAHE](https://repository.li.mahidol.ac.th/entities/publication/37cd0e4d-2e75-43df-94dd-a565b6f737e3/full)
- [Chest radiographs enhancement with contrast limited adaptive histogram equalization](https://pure.skku.edu/en/publications/chest-radiographs-enhancement-with-contrast-limited-adaptive-hist/)
- [Medical X-Ray Image Enhancement Using Global Contrast-Limited Adaptive Histogram Equalization](https://digitalcommons.njit.edu/fac_pubs/180)
- [Pediatric Bone Age Prediction Using Deep Learning (arXiv)](https://arxiv.org/pdf/2607.16936)
- [arXiv 2405.14986: preprocesamiento y regiones pequeñas en bone age](https://arxiv.org/abs/2405.14986v1)
- [PMC5820091: hand segmentation/registration para bone age](https://pmc.ncbi.nlm.nih.gov/articles/PMC5820091)
- [Segmentación adaptativa de huesos carpales (umbral local + Canny), IEEE Access](https://doaj.org/article/612449b0fbdc4a07a601d366016848c8)
- [Umbralización difusa vs. Otsu/k-means en fondos de rayos X (tesis VSB)](https://dspace.vsb.cz/items/64fce832-2011-4b04-9079-d10a5c1d5512/full)
- [Método adaptativo de contraste para huesos de mano (fuzzy)](https://scholar.hit.edu.cn/en/publications/adaptive-adjusting-algorithm-for-the-contrast-of-the-x-ray-images/)
- [The BoneXpert method for automated determination of skeletal maturity](https://research.regionh.dk/en/publications/the-bonexpert-method-for-automated-determination-of-skeletal-matu/)
- [BoneXpert Online, Instructions for Use (tasas de rechazo)](https://visiana.zohodesk.eu/portal/en/kb/articles/bonexpert-online-instructions-for-use)
- [Rolling ball background subtraction, ImageJ](https://imagej.net/imagej-wiki-static/Rolling_Ball_Background_Subtraction)
- [Morphological Background Detection and Illumination Normalization of Text Image with Poor Lighting](https://pmc.ncbi.nlm.nih.gov/articles/PMC4245115)
