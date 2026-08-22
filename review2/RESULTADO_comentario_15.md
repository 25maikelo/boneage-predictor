# Comentario 15 — Ablación insuficiente (baseline whole-hand)

> Resuelto de forma independiente al Comentario 12: el baseline whole-hand es una sola CNN
> end-to-end (sin segmentos que fusionar), así que la fuga de datos del pipeline de fusión no
> le aplica estructuralmente. Se entrenó y evaluó con datos reales en las tres condiciones que
> ya usa el resto del manuscrito (validación interna, RSNA externa oficial, MEX) — sin fabricar
> ningún resultado.

## Diseño experimental

Nuevo `MODEL_TYPE="whole_hand"` en `src/06_training.py` (experimento 60): una sola rama
DenseNet121 (misma arquitectura que `create_segment_model`), alimentada con la imagen de mano
completa ya preprocesada (recorte + CLAHE, `data/images/equalized/`, sin segmentar en 4
regiones), en vez de las 4 regiones anatómicas.

Controlado contra el experimento 27 (F-DenseNet121, modelo de referencia del manuscrito):
- Mismo dataset (`balanced_dataset.csv`, `AGE_RANGE=(24,216)`).
- Mismo split: `train_test_split(test_size=0.2, random_state=42)` → idénticos 9,426 train / 2,357
  val (mismos pacientes en ambos experimentos).
- Mismo backbone, mismos hiperparámetros (`WEIGHTS=None`, `NUM_LAYERS_UNFREEZE=10`, `LR=0.001`,
  `BATCH_SIZE=32`, misma augmentación, misma función de pérdida `attention_loss`).
- Protocolo de dos fases espejo del pipeline de fusión: fase 1 (15 épocas, backbone
  parcialmente descongelado) + fine-tuning (10 épocas, backbone completo descongelado, LR/10),
  mismos callbacks `EarlyStopping`/`ReduceLROnPlateau` que el resto del pipeline.
- Entrenado en SLURM (job 556419, partición GPU, nvd01), 2h42m total.

Evaluado en las tres condiciones que ya reporta el manuscrito para los demás backbones:
validación interna (mismo split que arriba), validación externa oficial RSNA ("MAE Test",
Tabla 8) y validación externa geográfica mexicana ("MEX", Tabla 9) — mismo preprocesamiento
(`frame_and_zoom` + CLAHE) que usan `07_validation.py`/`08_mex_validation.py`, sin el paso de
segmentación.

## Resultado

| Evaluación | Whole-hand MAE (meses) | Fusión 4 segmentos, F-DenseNet121 (meses) | Δ (whole-hand − fusión) |
|---|---:|---:|---:|
| Validación interna (n = 2,357) | 13.52 | 13.13 | +0.39 (≈2.9%) |
| RSNA externa oficial (n = 1,425, 0 fallos) vs (n = 1,393, 32 fallos por segmentación) | 15.87 | 14.18 | +1.69 (≈11.9%) |
| Externa geográfica MEX (n = 100, 0 fallos) vs (n = 99, 1 fallo) | 23.98 | 16.38 | **+7.60 (≈46.4%)** |

**Hallazgo principal:** la ventaja de la segmentación anatómica es modesta en el split de
validación interna (mismos pacientes vistos de forma similar durante el ajuste de
hiperparámetros), pero se amplía sustancialmente bajo *distribution shift* — y de forma muy
marcada en el cohorte mexicano externo, donde el modelo whole-hand comete en promedio 7.6 meses
más de error que el modelo con fusión de 4 regiones (46% peor). Esto conecta directamente con el
hallazgo central del artículo: el desempeño de validación interna no predice de forma confiable
la generalización externa, y en este caso específico, la segmentación en 4 regiones anatómicas
es uno de los factores que sí ayuda a esa generalización, no solo una elección arquitectónica
neutra.

Nota secundaria: el modelo whole-hand no depende del modelo de segmentación, así que procesó el
100% de ambos conjuntos externos (0 fallos), mientras que la fusión pierde 32/1,425 (RSNA) y
1/100 (MEX) casos por fallos de segmentación (ver hallazgo del Comentario 2/9 sobre
`empty_segment`, predominantemente en el pulgar). Es una ventaja real de robustez de pipeline,
pero de magnitud muy inferior a la pérdida de precisión.

## Significancia estadística (pareada por paciente)

Aunque es una corrida única por condición (sin semillas repetidas — mismo límite ya reconocido
en el Comentario 13), sí se puede probar significancia de forma pareada por paciente dentro de
cada condición, con la misma metodología ya usada en el resto del artículo (Wilcoxon
signed-rank + bootstrap pareado de 10,000 remuestreos, `review2/scripts/whole_hand_significance.py`):

| Evaluación | n pareado | ΔMAE (whole-hand − fusión) | IC 95% (bootstrap) | p (Wilcoxon) | p (bootstrap) | ¿Significativo? |
|---|---:|---:|---:|---:|---:|:---:|
| Validación interna | 2,357 | +0.39 | [−0.11, 0.89] | 0.112 | 0.141 | No |
| RSNA externa oficial | 1,393 | +0.86 | [0.17, 1.55] | 0.016 | 0.016 | **Sí** |
| Externa geográfica MEX | 99 | +7.67 | [4.74, 10.71] | <0.0001 | <0.0001 | **Sí (fuerte)** |

La diferencia **no** es estadísticamente significativa en el split interno (el IC cruza 0 — es
plausible que sea ruido de una sola corrida), pero **sí** lo es en ambas validaciones externas,
con una magnitud y significancia que crecen sustancialmente bajo *distribution shift*. Esto
convierte el hallazgo de "observación puntual" a "diferencia estadísticamente respaldada" en las
dos condiciones que más importan para la tesis del artículo (generalización externa).

**📝 Agrega esto** en la Sección 3 (o como nueva subsección de ablación), después de la
comparación de backbones existente:

> "To isolate the contribution of the four-region anatomical segmentation itself, we trained a
> whole-hand baseline: a single DenseNet121 branch with identical hyperparameters, data split,
> loss function, and two-phase training protocol as the segmented fusion model, but taking the
> full preprocessed hand image (crop + CLAHE, no anatomical segmentation) as input. On the
> internal validation split (n = 2,357), the whole-hand baseline achieved an MAE of 13.52
> months versus 13.13 months for the segmented fusion model (Δ = +0.39 months, ≈2.9%); this
> difference was not statistically significant (paired Wilcoxon p = 0.112; paired bootstrap 95%
> CI [−0.11, 0.89]). However, under external evaluation the gap widened substantially and became
> statistically significant: on the official RSNA validation set (n = 1,393 common cases) the
> whole-hand MAE was 15.04 months versus 14.18 months for the fusion model (Δ = +0.86 months,
> Wilcoxon p = 0.016, bootstrap 95% CI [0.17, 1.55]), and on the external Mexican cohort (n = 99)
> the whole-hand MAE was 24.05 months versus 16.38 months (Δ = +7.67 months, Wilcoxon p < 0.0001,
> bootstrap 95% CI [4.74, 10.71]). This indicates that anatomical segmentation contributes
> significantly more to cross-population generalization than to in-distribution accuracy,
> reinforcing this study's central finding that internal validation performance is a poor proxy
> for external, real-world reliability. As a secondary observation, the whole-hand model, lacking
> a dependency on the segmentation step, processed 100% of both external sets, whereas the fusion
> pipeline lost 32/1,425 (RSNA) and 1/100 (MEX) cases to segmentation failures — a modest
> robustness advantage that does not offset its accuracy loss. As this comparison reflects a
> single training run per condition rather than repeated seeds, the paired significance tests
> should be interpreted with that caveat in mind, though the consistency and magnitude of the
> external results (particularly MEX) make it unlikely to be an artifact of a single unlucky
> initialization."

## Lo que queda fuera de alcance

Las comparaciones adicionales que también pide el comentario (efecto de la variable de sexo,
CLAHE, regiones individuales por separado) son experimentos completamente nuevos no
implementados — quedan como trabajo futuro declarado explícitamente, no se fabrican resultados
para ellos.
