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

### MEX — dataset mexicano (n = 99 imágenes)

| Exp | Backbone | MAE Global | 0–6 años | 6–12 años | 12–19 años |
|-----|---------|:----------:|:--------:|:---------:|:----------:|
| 27 | DenseNet121 | **16.4 m** | 38.9 m | 14.6 m | 13.2 m |
| 28 | InceptionV3 | 17.3 m | 27.0 m | 13.0 m | 19.1 m |
| 23 | ResNet50 | 19.3 m | 42.5 m | 16.2 m | 17.2 m |
| 26 | VGG16 | 40.0 m | 68.0 m | 19.5 m | 53.1 m |

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

### MEX (n = 99 muestras comunes)

| Par | ΔMAE | IC 95% | p Wilcoxon (Holm) | p Bootstrap (Holm) | r | Sig. |
|-----|:----:|:------:|:-----------------:|:------------------:|:---:|:----:|
| ResNet50 vs VGG16 | −20.7 m | [−25.3, −16.2] | < 0.001 | < 0.001 | 0.890 | ✅ ✅ |
| ResNet50 vs DenseNet121 | +2.9 m | [+0.4, +5.5] | 0.095 | 0.070 | 0.624 | ❌ ❌ |
| ResNet50 vs InceptionV3 | +2.1 m | [−1.1, +5.1] | 0.706 | 0.400 | 0.548 | ❌ ❌ |
| VGG16 vs DenseNet121 | +23.7 m | [+18.8, +28.6] | < 0.001 | < 0.001 | 0.915 | ✅ ✅ |
| VGG16 vs InceptionV3 | +22.8 m | [+18.1, +27.2] | < 0.001 | < 0.001 | 0.919 | ✅ ✅ |
| DenseNet121 vs InceptionV3 | −0.9 m | [−3.6, +1.7] | 0.706 | 0.505 | 0.554 | ❌ ❌ |

> El menor n en MEX (99 vs 1,393) reduce el poder estadístico: ResNet50, DenseNet121 e InceptionV3 son indistinguibles entre sí en este dataset.

---

## Conclusiones

1. **VGG16 no es apto para esta tarea**, MAE 2.5× peor que los demás en RSNA, significativamente inferior en ambos datasets con efecto grande (r ≈ 0.73–0.92). Falla especialmente en edades extremas (0–6 y 12–19 años).

2. **DenseNet121 e InceptionV3 son equivalentes**: diferencia de 0.59 m en RSNA y 0.90 m en MEX sin relevancia clínica ni significancia estadística consistente entre Wilcoxon y bootstrap. DenseNet121 tiene el MAE numéricamente más bajo en MEX (16.4 m vs. 17.3 m), pero la diferencia no es significativa. La elección de DenseNet121 sobre InceptionV3 se justifica por tiempo de entrenamiento (~10 h vs ~14 h) sin costo en rendimiento.

3. **ResNet50 no se distingue de DenseNet121 ni de InceptionV3 en MEX** (p > 0.05 en ambos tests), pero sí es significativamente peor que ambos en RSNA (r ≈ 0.55–0.62, efecto mediano-grande), atribuible al mayor poder estadístico de RSNA (n=1,393 vs. 99).

4. **Ranking consolidado:** DenseNet121 ≈ InceptionV3 ≈ ResNet50 (en MEX) / DenseNet121 ≈ InceptionV3 > ResNet50 (en RSNA) >> VGG16 (en ambos).

---

## Archivos generados

| Archivo | Descripción |
|---------|-------------|
| `experiments/paired/backbone_ablation_23_28.json` | Prueba pareada RSNA — corrección Holm |
| `experiments/paired/backbone_ablation_23_28_mex.json` | Prueba pareada MEX — corrección Holm |
| `src/11_paired_validation.py` | Script reutilizable de prueba pareada y ablación |
