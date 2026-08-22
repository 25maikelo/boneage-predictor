# Revisión de texto — comparación "fusion model vs. baseline" (backbone vs. simple_cnn)

> No corresponde a un único comentario numerado del revisor — es una revisión directa de un
> párrafo del manuscrito que el usuario señaló. Se relaciona con el Comentario 15 (necesita un
> baseline whole-hand real, que ya existe — ver `RESULTADO_comentario_15.md`) y con el Hallazgo
> Crítico #1 (bug `real_age`/`bone_age`). Decisión tomada: **opción (b)** — conservar la
> comparación backbone-vs-simple_cnn, mismo tema (misma fusión de 4 segmentos, ambos), pero: (1)
> renombrar "the baseline" con precisión, (2) definir "Fusion MAE", (3) corregir los números de
> Mex MAE contra `bone_age`.

## Texto original (tal como está en el manuscrito)

> "The results are consistent across all datasets. On the trimmed dataset, the fusion model
> achieves 14.6 m vs. 39.2 m (Val MAE), 17.6 m vs. 35.9 m (Mex MAE), and 9.2 m vs. 34.5 m
> (Fusion MAE), with 50.9% vs. 11.8% in ±12m accuracy. On the balanced dataset, it achieves
> 15.1 m vs. 43.5 m, 13.9 m vs. 35.9 m, and 9.0 m vs. 41.8 m, with 47.1% vs. 9.2%. On the full
> (complete) dataset, it achieves 15.4 m vs. 30.2 m, 16.7 m vs. 22.2 m, and 6.8 m vs. 25.5 m,
> with 48.9% vs. 22.5%. These results demonstrate that the proposed fusion strategy consistently
> outperforms the baseline under identical conditions across all datasets."

## Tres problemas identificados

1. **"The baseline" no es un modelo whole-hand.** Los números coinciden exactamente con
   `docs/results/2026-05-28_resultados.md` (tabla del 2026-04-29): "fusion model" = experimentos
   34/40/37 (`MODEL_TYPE=backbone`, DenseNet121 preentrenado por segmento), "baseline" =
   experimentos 33/39/36 (`MODEL_TYPE=simple_cnn`, 4 CNNs pequeñas *desde cero* por segmento).
   **Ambos** segmentan la mano en 4 regiones y fusionan — la única diferencia es el extractor de
   características por segmento (backbone preentrenado vs. CNN pequeña sin preentrenar). La
   frase final ("the proposed fusion strategy... outperforms the baseline") sugiere que se está
   probando fusión-vs-no-fusión, cuando en realidad se está probando backbone-vs-CNN-pequeña. Esa
   comparación real (fusión-vs-whole-hand) es exactamente el Comentario 15, y ya existe con datos
   reales — ver `RESULTADO_comentario_15.md`.

2. **"Fusion MAE" es una tercera métrica interna, sin definir en el texto.** Según la leyenda de
   la tabla fuente (`docs/results/2026-05-28_experiments_summary.md`): *"Fusión MAE: mejor
   val_mae del integrador final"* — es el MAE del propio split de validación interna usado
   durante el entrenamiento del modelo de fusión (no una evaluación externa). Es distinto de
   "Val MAE" (que en esa tabla es la validación externa oficial RSNA, 1,393 imágenes) y de "Mex
   MAE" (validación externa mexicana). El texto del manuscrito nunca aclara esto, y "Val MAE"
   puede confundirse con "validación interna" cuando en realidad es la validación *externa*
   oficial — nomenclatura opuesta a la intuición.

3. **Los valores de "Mex MAE" venían del bug `real_age`/`bone_age`.** La tabla fuente es del
   2026-05-28, anterior al hallazgo y corrección del bug en `08_mex_validation.py` (Hallazgo
   Crítico #1). Se volvieron a correr los 6 experimentos (33/34/36/37/39/40) con el script ya
   corregido (jobs SLURM 556427-556432):

| Exp | Arquitectura | Dataset | RSNA ext. MAE | Mex MAE — viejo (bug) | Mex MAE — **corregido** | Fusión (val. interna) MAE | ±12m |
|---|---|---|---:|---:|---:|---:|---:|
| 34 | `backbone` | trimmed | 14.6 m | 17.6 m | 16.82 m | 9.2 m | 50.9% |
| 40 | `backbone` | balanced | 15.1 m | 13.9 m | 16.42 m | 9.0 m | 47.1% |
| 37 | `backbone` | full | 15.4 m | 16.7 m | 18.37 m | 6.8 m | 48.9% |
| 33 | `simple_cnn` | trimmed | 39.2 m | 35.9 m | 38.21 m | 34.5 m | 11.8% |
| 39 | `simple_cnn` | balanced | 43.5 m | 35.9 m | 37.88 m | 41.8 m | 9.2% |
| 36 | `simple_cnn` | full | 30.2 m | 22.2 m | 24.35 m | 25.5 m | 22.5% |

(RSNA ext. MAE y ±12m no cambian — esas sí comparaban contra la variable correcta desde el
inicio; solo Mex MAE estaba afectado.)

La corrección **no cambia la conclusión cualitativa** (backbone preentrenado le gana a
simple_cnn en las tres variantes de dataset, con margen amplio), pero sí cambia los números
exactos que se citan, y afecta a otro punto del manuscrito: la Sección 3.3 también citaba estos
mismos MEX MAE para argumentar que el dataset balanceado es el mejor de los tres — ver la
corrección aplicada en `RESULTADO_comentario_3.md` (el margen se reduce de ~3-4 meses a
~0.4-2 meses).

## 📝 Reemplaza esto

> "The results are consistent across all datasets. On the trimmed dataset, the fusion model
> achieves 14.6 m vs. 39.2 m (Val MAE), 17.6 m vs. 35.9 m (Mex MAE), and 9.2 m vs. 34.5 m
> (Fusion MAE), with 50.9% vs. 11.8% in ±12m accuracy. On the balanced dataset, it achieves
> 15.1 m vs. 43.5 m, 13.9 m vs. 35.9 m, and 9.0 m vs. 41.8 m, with 47.1% vs. 9.2%. On the full
> (complete) dataset, it achieves 15.4 m vs. 30.2 m, 16.7 m vs. 22.2 m, and 6.8 m vs. 25.5 m,
> with 48.9% vs. 22.5%. These results demonstrate that the proposed fusion strategy consistently
> outperforms the baseline under identical conditions across all datasets."

## Por esto

> "The results are consistent across all three dataset variants. We compared the proposed
> pretrained-backbone fusion model against a from-scratch CNN fusion baseline — four
> independently trained, non-pretrained convolutional branches (one per anatomical segment)
> combined through the same segmentation-and-fusion pipeline, differing from the proposed model
> only in the per-segment feature extractor. Three metrics are reported for each configuration:
> external RSNA validation MAE (Table 8 protocol), external Mexican cohort MAE (Table 9
> protocol), and the fusion model's own internal validation MAE (best epoch, same held-out split
> used during training). On the trimmed dataset: 14.6 vs. 39.2 months (RSNA), 16.8 vs. 38.2
> months (Mexican cohort), and 9.2 vs. 34.5 months (internal validation), with 50.9% vs. 11.8%
> of predictions within ±12 months. On the balanced dataset: 15.1 vs. 43.5 months (RSNA), 16.4
> vs. 37.9 months (Mexican cohort), and 9.0 vs. 41.8 months (internal validation), with 47.1%
> vs. 9.2% within ±12 months. On the full dataset: 15.4 vs. 30.2 months (RSNA), 18.4 vs. 24.3
> months (Mexican cohort), and 6.8 vs. 25.5 months (internal validation), with 48.9% vs. 22.5%
> within ±12 months. These results show that, under an identical data split and the same
> segmentation-and-fusion pipeline, a pretrained backbone per anatomical segment consistently
> and substantially outperforms a from-scratch shallow CNN of the same architecture family. This
> comparison isolates the contribution of the per-segment feature extractor; it does not test
> whether anatomical segmentation and fusion themselves improve over a whole-hand, non-segmented
> model — that comparison is reported separately in Section 3.X (whole-hand baseline)."

## Nota de proceso

Los números de "Fusion MAE"/validación interna (9.2/9.0/6.8 para backbone; 34.5/41.8/25.5 para
simple_cnn) **no** fueron recalculados — no dependían del bug `real_age`/`bone_age` (ese bug
solo afecta la comparación contra el dataset mexicano). Se citan tal cual del reporte original.
