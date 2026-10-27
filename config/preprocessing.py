"""
Configuración de los scripts de preprocesamiento:

    src/preprocessing/02_frame_and_zoom.py          (recorte y rotación)
    src/preprocessing/03_histogram_equalization.py  (ecualización CLAHE)

Centraliza los parámetros que antes estaban fijos dentro de cada función, para
poder ablacionarlos sin editar el código del pipeline. Ver
docs/results/doctorado/preprocesamiento_diagnostico/diagnostico.md para el diagnóstico
que motivó varios de los valores por default actuales.
"""

# ─── 02_frame_and_zoom.py: detección de la mano (Otsu + contorno) ─────────────

# Suavizado antes de umbralizar con Otsu. Kernel mas grande = menos sensible a
# ruido de grano, pero tambien difumina bordes finos del contorno.
GAUSSIAN_BLUR_KERNEL = (5, 5)

# Dilatación aplicada a la máscara binaria antes de buscar contornos, para
# cerrar huecos internos (espacios entre dedos) y que la mano quede como un
# solo contorno externo en vez de fragmentarse.
DILATION_KERNEL_SIZE = (9, 9)
DILATION_ITERATIONS = 4

# ─── 02_frame_and_zoom.py: ángulo de rotación ──────────────────────────────────

# cv2.minAreaRect devuelve angle=90 como valor de frontera para rectángulos ya
# alineados a los ejes; dentro de esta tolerancia (grados) alrededor de 90 no
# se aplica la corrección +90 de la lógica normal, para evitar una rotación
# espuria de 90° en manos que ya estaban bien orientadas (ver diagnostico.md,
# hallazgo 5). Subir este valor hace la detección de "ya alineado" más laxa.
ANGLE_BOUNDARY_TOLERANCE = 1.0

# ─── 02_frame_and_zoom.py: recorte final ───────────────────────────────────────

# Si el recorte final ocupa menos de esta fracción del área de la imagen
# original, se descarta y se conserva la imagen completa sin recortar
# (estado "discard_area"). Ver diagnostico.md, hallazgo 2: en la práctica esto
# ocurre sobre todo en imágenes de contraste global muy bajo donde Otsu no
# logra separar la mano del fondo.
MIN_CROP_AREA_RATIO = 0.2

# ─── 03_histogram_equalization.py: CLAHE ───────────────────────────────────────

# Límite de recorte del histograma por tile: más alto amplifica más el
# contraste local, pero también puede amplificar ruido. clipLimit=4.0 fue
# validado contra clipLimit=2.0 (el default de OpenCV) sobre las 12,811
# imágenes raw: preserva más contraste global y satura menos píxeles
# (diagnostico.md, hallazgo 3).
CLAHE_CLIP_LIMIT = 4.0

# Tamaño de la grilla de tiles sobre la que CLAHE calcula el histograma local.
# Más chico = realce más local (más sensible a textura fina); más grande =
# realce más global.
CLAHE_TILE_GRID_SIZE = (8, 8)

# ─── 03_histogram_equalization.py: mezcla con la imagen original ──────────────

# Proporción de la imagen ecualizada con CLAHE en la mezcla final
# (mezcla = (1 - BLEND_CLAHE_WEIGHT) * original + BLEND_CLAHE_WEIGHT * clahe).
# Evita el aspecto "sobre-realzado" de aplicar CLAHE puro.
BLEND_CLAHE_WEIGHT = 0.4

# ─── 03_histogram_equalization.py: ajuste final de contraste/brillo ───────────

# Ajuste lineal final: salida = clip(entrada * FINAL_CONTRAST_ALPHA + FINAL_CONTRAST_BETA, 0, 255).
# Nota: no usar cv2.convertScaleAbs para aplicar esto (ver diagnostico.md,
# hallazgo 6): refleja en vez de saturar los valores que quedarían negativos,
# corrompiendo los tonos más oscuros (el fondo de la radiografía).
FINAL_CONTRAST_ALPHA = 0.9
FINAL_CONTRAST_BETA = -10
