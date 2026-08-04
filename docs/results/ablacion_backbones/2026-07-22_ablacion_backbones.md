# Ablación de Backbone — Experimentos 23, 26, 27, 28

> Objetivo: comparar cuatro arquitecturas de backbone manteniendo todo lo demás constante, usando datos balanceados e imágenes con información espacial completa (modo `spatial`).

---

## Configuración experimental

| Parámetro | Valor |
|-----------|-------|
| `MODEL_TYPE` | `backbone` |
| `IMAGE_SIZE` | 112 × 112 |
| `SEGMENT_MODE` | `spatial` |
| `DATASET_PATH` | `balanced_dataset.csv` (n ≈ 11,783) |
| `AGE_RANGE` | (24, 216) meses |
| `USE_GENDER` | True |
| `FREEZE_EXTRACTORS` | True |
| `LEARNING_RATE` | 1e-3 |
| `BATCH_SIZE` | 32 |
| `LOSS_FUNCTION_NAME` | `attention_loss` |
| `USE_WARMUP` | False |

Variable cambiada: `BASE_MODEL_CHOICE` ∈ {resnet50, vgg16, densenet121, inceptionv3}

---

## Resultados de entrenamiento

| Exp | Backbone | Épocas fusión | Best val MAE (fusión) | Best val MAE (fine-tuning) | Tiempo total |
|-----|---------|:-------------:|:---------------------:|:--------------------------:|:------------:|
| 23 | ResNet50 | 9/20 | 24.6 m | **15.0 m** | ~12 h |
| 26 | VGG16 | 7/20 | 34.6 m | 36.2 m | ~7 h |
| 27 | DenseNet121 | 8/20 | 16.9 m | **13.1 m** | ~10 h |
| 28 | InceptionV3 | 19/20 | 12.8 m | **12.4 m** | ~14 h |

> VGG16 detuvo el entrenamiento pronto (early stopping en epoch 7) y no mejoró en fine-tuning — señal de que la arquitectura no converge bien en esta tarea.

---

## Resultados de validación

### RSNA (n = 1,393 imágenes)

| Exp | Backbone | MAE Global | 0–6 años | 6–12 años | 12–19 años |
|-----|---------|:----------:|:--------:|:---------:|:----------:|
| 28 | InceptionV3 | **13.6 m** | 13.8 m | 12.5 m | 14.5 m |
| 27 | DenseNet121 | 14.2 m | 19.5 m | 13.5 m | 13.5 m |
| 23 | ResNet50 | 16.2 m | 21.0 m | 14.7 m | 16.4 m |
| 26 | VGG16 | 37.2 m | 58.0 m | 17.5 m | 52.9 m |

### MEX — dataset mexicano (n = 98 imágenes)

| Exp | Backbone | MAE Global | 0–6 años | 6–12 años | 12–19 años |
|-----|---------|:----------:|:--------:|:---------:|:----------:|
| 28 | InceptionV3 | **17.2 m** | 18.4 m | 17.1 m | 17.2 m |
| 23 | ResNet50 | 17.4 m | 30.7 m | 18.8 m | 12.9 m |
| 27 | DenseNet121 | 17.9 m | 29.8 m | 18.4 m | 15.3 m |
| 26 | VGG16 | 28.9 m | 70.9 m | 16.7 m | 48.2 m |

---

## Pruebas estadísticas pareadas

Método: **Wilcoxon signed-rank test** + **Bootstrap pareado** (n = 10,000 remuestreos).  
Corrección de comparaciones múltiples: **Holm step-down** (6 pares, α = 0.05) — más potente que Bonferroni con el mismo control de FWER.  
Las pruebas se realizan sobre la **intersección de IDs** comunes entre cada par de modelos.

> ΔMAE = MAE(A) − MAE(B). Positivo = A es peor que B.  
> Tamaño de efecto r: |r| > 0.1 pequeño · > 0.3 mediano · > 0.5 grande.  
> Sig. = Wilcoxon / Bootstrap (ambos deben ser significativos para considerarse conclusivo).

### RSNA (n = 1,393 muestras comunes)

| Par | ΔMAE | IC 95% | p Wilcoxon (Holm) | p Bootstrap (Holm) | r | Sig. |
|-----|:----:|:------:|:-----------------:|:------------------:|:---:|:----:|
| ResNet50 vs VGG16 | −21.0 m | [−22.3, −19.7] | < 0.001 | < 0.001 | 0.887 | ✅ ✅ |
| ResNet50 vs DenseNet121 | +2.0 m | [+1.4, +2.6] | < 0.001 | < 0.001 | 0.603 | ✅ ✅ |
| ResNet50 vs InceptionV3 | +2.6 m | [+1.9, +3.2] | < 0.001 | < 0.001 | 0.614 | ✅ ✅ |
| VGG16 vs DenseNet121 | +23.0 m | [+21.7, +24.3] | < 0.001 | < 0.001 | 0.905 | ✅ ✅ |
| VGG16 vs InceptionV3 | +23.6 m | [+22.3, +24.9] | < 0.001 | < 0.001 | 0.915 | ✅ ✅ |
| **DenseNet121 vs InceptionV3** | **+0.59 m** | **[−0.05, +1.19]** | **0.050** | **0.065** | 0.530 | **✅ ❌** |

> ⚠️ DenseNet121 vs InceptionV3: Wilcoxon y Bootstrap divergen. El IC bootstrap cruza el 0 y la diferencia (0.59 m) no tiene relevancia clínica — se interpreta como equivalencia práctica.

### MEX (n = 98 muestras comunes)

| Par | ΔMAE | IC 95% | p Wilcoxon (Holm) | p Bootstrap (Holm) | r | Sig. |
|-----|:----:|:------:|:-----------------:|:------------------:|:---:|:----:|
| ResNet50 vs VGG16 | −11.5 m | [−16.6, −6.4] | < 0.001 | < 0.001 | 0.732 | ✅ ✅ |
| ResNet50 vs DenseNet121 | −0.5 m | [−3.1, +2.1] | 1.000 | 1.000 | 0.506 | ❌ ❌ |
| ResNet50 vs InceptionV3 | +0.3 m | [−2.7, +3.2] | 1.000 | 1.000 | 0.522 | ❌ ❌ |
| VGG16 vs DenseNet121 | +11.0 m | [+5.8, +16.1] | < 0.001 | < 0.001 | 0.736 | ✅ ✅ |
| VGG16 vs InceptionV3 | +11.8 m | [+6.7, +16.6] | < 0.001 | < 0.001 | 0.757 | ✅ ✅ |
| DenseNet121 vs InceptionV3 | +0.76 m | [−1.7, +3.2] | 1.000 | 1.000 | 0.541 | ❌ ❌ |

> El menor n en MEX (98 vs 1,393) reduce el poder estadístico: ResNet50, DenseNet121 e InceptionV3 son indistinguibles entre sí en este dataset.

---

## Conclusiones

1. **VGG16 no es apto para esta tarea** — MAE 2.5× peor que los demás en RSNA, significativamente inferior en ambos datasets con efecto grande (r ≈ 0.73–0.92). Falla especialmente en edades extremas (0–6 y 12–19 años).

2. **DenseNet121 e InceptionV3 son equivalentes** — la diferencia de 0.59 m en RSNA y 0.76 m en MEX no tiene relevancia clínica. En RSNA el Wilcoxon bordea el umbral (p = 0.050) pero el bootstrap no lo confirma (IC cruza 0); en MEX ambos tests coinciden en no-significancia. La elección de DenseNet121 sobre InceptionV3 se justifica por tiempo de entrenamiento (~10 h vs ~14 h) sin costo en rendimiento.

3. **ResNet50 es significativamente peor que DenseNet121 e InceptionV3 en RSNA** (r ≈ 0.60, efecto mediano-grande). En MEX la diferencia no alcanza significancia estadística, atribuible al menor n (98 muestras).

4. **Ranking consolidado:** DenseNet121 ≈ InceptionV3 > ResNet50 >> VGG16.

---

## Archivos generados

| Archivo | Descripción |
|---------|-------------|
| `experiments/paired/backbone_ablation_23_28.json` | Prueba pareada RSNA — corrección Holm |
| `experiments/paired/backbone_ablation_23_28_mex.json` | Prueba pareada MEX — corrección Holm |
| `src/11_paired_validation.py` | Script reutilizable de prueba pareada y ablación |
