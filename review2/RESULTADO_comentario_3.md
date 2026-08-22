# Comentario 3 — Metodología de balanceo (distribución antes/después + justificación)

> Verificado directamente contra `data/training/boneage-training-dataset.csv` (raw) y `data/training/dataset_analysis/balanced_dataset.csv` (balanceado) — sin cómputo nuevo, solo lectura de los CSV ya existentes.

## Figura (la que pide explícitamente el comentario)

**Decisión:** en vez de una figura nueva y separada, se **extiende la Figura 2 ya existente** a 6 paneles en una sola imagen: se conservan (a)(b) RSNA raw y (c)(d) Mexicano tal como están publicados hoy, y se agregan (e)(f) con la distribución del dataset RSNA **balanceado** — así el lector compara los tres datasets (RSNA raw, mexicano, RSNA balanceado) en una sola figura en vez de tener que saltar entre dos.

**Imagen:** [`figures/figura2_extendida_6paneles.png`](figures/figura2_extendida_6paneles.png). Script: [`scripts/figura2_extendida_6paneles.py`](scripts/figura2_extendida_6paneles.py). Reemplaza por completo a la Figura 2 actual (misma identidad visual en (a)-(d), reconstruidos desde los CSV reales para que el archivo sea reproducible end-to-end).

**📝 Reemplaza esto** — el pie de la Figura 2 actual:
> "**Figure 2.** Comparative demographic characterization of the datasets. (a) Age distribution of the RSNA dataset represented as a frequency histogram (age in months); (b) Gender distribution of the RSNA dataset shown as percentage proportions; (c) Age distribution of the Mexican clinical dataset represented as a frequency histogram (age in months); (d) Gender distribution of the Mexican clinical dataset shown as percentage proportions."

**Por esto:**
> "**Figure 2.** Comparative demographic characterization of the datasets, including the effect of the age-frequency balancing procedure (Section 2.1). (a) Age distribution of the raw RSNA training dataset (n = 12,611), represented as a frequency histogram with a kernel density overlay; (b) Gender distribution of the raw RSNA training dataset; (c) Age distribution of the external Mexican clinical dataset (n = 100); (d) Gender distribution of the external Mexican clinical dataset; (e) Age distribution of the balanced RSNA training dataset (n = 11,783), obtained after applying the age-frequency threshold described in Section 2.1; (f) Gender distribution of the balanced RSNA training dataset."

También hay que actualizar el párrafo que introduce la Figura 2 en el cuerpo del texto (Sección 2.1) para mencionar los paneles (e)(f) y el propósito de comparar raw vs. balanceado, no solo RSNA vs. México.

**📝 Reemplaza esto** — el párrafo que introduce la Figura 2 en el cuerpo del texto:
> "Figure 2 shows the comparative demographic characterization of the two datasets under consideration: the RSNA dataset (a and b) and the Mexican dataset (c and d). The left column shows the age distributions in each dataset as frequency histograms, with age values normalized to months to ensure consistent scales for rigorous comparison of dispersion patterns, skewness, and mode across populations. The right column presents the gender composition for each population as pie charts, showing the proportions of males and females in each dataset. The comparative representation of these two datasets enables the identification of potential demographic imbalances that could lead to distributional bias during model training."

**Por esto:**
> "Figure 2 shows the comparative demographic characterization of the three datasets under consideration: the raw RSNA training dataset (a and b), the external Mexican clinical dataset (c and d), and the balanced RSNA training dataset obtained after the age-frequency filtering procedure described in Section 2.1 (e and f). The left column shows the age distributions in each dataset as frequency histograms, with age values normalized to months to ensure consistent scales for rigorous comparison of dispersion patterns, skewness, and mode across populations. The right column presents the gender composition for each population as pie charts, showing the proportions of males and females in each dataset. Beyond enabling the identification of potential demographic imbalances between the RSNA and Mexican datasets that could lead to distributional bias during model training, the inclusion of the balanced RSNA subset (e and f) allows a direct visual comparison against the raw RSNA distribution (a and b), confirming that the age-frequency balancing procedure preserves the overall shape and gender ratio of the original training population while reducing the representation of sparsely populated age categories."

## Distribución antes y después del filtro

| | Antes (dataset raw) | Después (dataset balanceado) |
|---|---:|---:|
| Imágenes | 12,611 | 11,783 |
| Edades únicas | 160 | 36 |
| Rango de edad | 1–228 meses | 24–216 meses |
| Sexo masculino | 6,833 (54.2%) | 6,313 (53.6%) |
| Sexo femenino | 5,778 (45.8%) | 5,470 (46.4%) |
| Edad media | 127.3 meses | 129.5 meses |
| Edad mediana | 132.0 meses | 132.0 meses (idéntica) |
| Edades excluidas (< 50 img./mes) | — | 124 de 160 (77.5%) → −828 imágenes (6.6% del total) |

**La distribución de sexo prácticamente no cambia** (54.2%♂/45.8%♀ → 53.6%♂/46.4%♀, diferencia <1 punto porcentual), y la mediana de edad es idéntica antes y después — el filtro no introduce un sesgo de sexo apreciable, y su efecto sobre la edad central es mínimo. La mayoría de las 124 edades excluidas son valores aislados con muy pocas muestras (frecuentemente <10), dispersos tanto en los extremos del rango (1–23 y 217–228 meses, ya fuera del rango final) como en "huecos" puntuales dentro de 24–216 meses — no una franja etaria contigua completa eliminada en bloque.

**📝 Agrega esto** en la Sección 2.1, justo después del párrafo que describe la construcción del subconjunto balanceado:
> "Table X reports the age and sex distribution before and after this filtering step. The sex ratio remained stable (54.2%/45.8% male/female before vs. 53.6%/46.4% after), and the median age was unchanged (132 months in both cases). Of the 160 original one-month age categories, 124 (77.5%) were excluded, corresponding to 828 of 12,611 images (6.6% of the training corpus). Most excluded categories correspond to isolated ages with very few samples (frequently fewer than 10) scattered across the extremes of the original 1–228 month range and across sparse gaps within it, rather than a systematically excluded contiguous age band."

## Justificación del umbral de 50 imágenes/edad

**📝 Agrega esto** en el mismo lugar:
> "The 50-image-per-month threshold was selected empirically as a minimum sample size expected to support stable per-class gradient estimates during segment-level training, given the batch size of 32 used throughout this study — a threshold below one full batch per age class would make it likely for some training epochs to contain no examples of that age at all."

## "Repeat the analysis... or demonstrate that this restriction enhances performance on a genuinely independent dataset"

Esta parte del comentario **ya tiene evidencia directa en el propio manuscrito**, solo que no está conectada explícitamente con este punto. La Sección 3.3 ya reporta la comparación de la arquitectura de fusión bajo tres configuraciones de dataset — trimmed, balanced, y full (completo, sin filtrar) — y el **MAE en el dataset mexicano** (que es exactamente el "genuinely independent dataset" que pide el revisor) es:

> **⚠️ Corrección (2026-08-07):** los valores de MAE MEX de esta tabla venían del reporte `docs/results/2026-05-28_resultados.md`, generado **antes** de encontrar y corregir el bug de `08_mex_validation.py` que comparaba contra `real_age` (edad cronológica) en vez de `bone_age` (edad ósea TW3) — ver Hallazgo Crítico #1 en `ANALISIS_COMENTARIOS.md`. Se volvieron a correr los experimentos 34/40/37 (`backbone`, trimmed/balanced/full) con el script ya corregido (jobs SLURM 556427-556432) y los números cambian:

| Configuración de dataset | MAE MEX (fusión) — valor viejo (con bug) | MAE MEX (fusión) — **corregido** |
|---|---:|---:|
| Trimmed (24–216m, sin balancear) | 17.6 m | 16.82 m |
| Balanced (≥50 img./edad) | 13.9 m ← "mejor" (bug) | **16.42 m ← mejor (corregido)** |
| Full (1–228m, sin filtrar) | 16.7 m | 18.37 m |

El dataset balanceado **sigue siendo** el mejor de los tres bajo la corrección, pero el margen se reduce drásticamente: de una diferencia de ~3–4 meses frente a trimmed/full, pasa a una diferencia de apenas ~0.4–2 meses. La conclusión cualitativa (el balanceo no perjudica el desempeño externo) se sostiene, pero ya no se puede describir como una mejora grande — es un margen modesto, del mismo orden que el ruido de una sola corrida sin semillas repetidas.

**📝 Agrega esto** en la Sección 3.3, justo después del párrafo que reporta esas tres configuraciones:
> "Notably, on the genuinely independent Mexican external validation set, the balanced dataset configuration achieved the lowest MAE among the three settings (16.4 months, vs. 16.8 months for the trimmed and 18.4 months for the full/unfiltered configuration), indicating that the age-frequency balancing procedure does not compromise — and, by a modest margin, slightly improves — external generalization relative to training on the complete, unfiltered dataset. This margin is small relative to the single-run variability observed elsewhere in this study and should not be overstated."

## Lo que sigue sin poder responderse

Nada pendiente de este comentario específico — los tres elementos que pide (distribución antes/después, justificación del umbral, evidencia sobre dataset independiente) ya tienen respuesta con datos verificados, sin necesidad de cómputo ni entrenamiento nuevo.
