# Comentarios 4, 7, 8 — Terminología TW3, estadística externa, y gráficos de dispersión

> Todos dependen de la validación mexicana ya corregida (`bone_age` en vez de `real_age`, ver Hallazgo Crítico #1 en `ANALISIS_COMENTARIOS.md`). Re-ejecuté `08_mex_validation.py` para los 4 backbones (23/26/27/28) antes de calcular todo lo de este documento — los números de MEX aquí **no** son los que hoy aparecen en la Tabla 9/Figura 8 del manuscrito, son los corregidos.
>
> Scripts: [`scripts/extended_stats.py`](scripts/extended_stats.py), [`scripts/figure8_improved.py`](scripts/figure8_improved.py). JSON: [`results_json/extended_stats_results.json`](results_json/extended_stats_results.json), [`results_json/figure8_stats.json`](results_json/figure8_stats.json).

---

## ⚠️ Cambio de hallazgo central: ya no es F-InceptionV3 el mejor en MEX

Con los datos corregidos, **la Tabla 9 cambia**:

| Backbone | MAE viejo (Tabla 9, con el bug) | MAE nuevo (corregido, contra `bone_age`) |
|---|---:|---:|
| F-ResNet50 | 16.22 | **19.33** |
| F-VGG16 | 27.98 | **40.04** |
| **F-DenseNet121** | 16.30 | **16.38** ← ahora el más bajo |
| F-InceptionV3 | 15.98 | **17.28** |

El abstract, la Sección 3.4 y las Conclusiones afirman hoy que *"F-InceptionV3 demonstrated the greatest robustness, achieving the lowest MAE"* — con los datos corregidos esto ya no es cierto en el punto estimado (aunque, como se ve abajo, la diferencia sigue sin ser estadísticamente significativa). Hay que decidir cómo se redacta esto — ver la sección de pruebas pareadas más abajo antes de reescribir esas frases.

---

## Comentario 4 — Terminología TW3 vs. edad cronológica

**📝 Reemplaza esto** (párrafo introductorio de la Figura 8):
> "Figure 8 presents scatter plots comparing chronological and predicted ages, further clarifying the behavior of the evaluated architectures."

**Por esto:**
> "Figure 8 presents scatter plots comparing TW3-assigned bone age and predicted age, with identity lines, fitted regression lines, and 95% confidence bands, further clarifying the behavior of the evaluated architectures."

**📝 Reemplaza los ejes y título de cada subgráfico** ("Real Age (months)" / "Dispersion Real Age vs Prediction") **por** "TW3 Bone Age (months)" / "Dispersion: TW3 Bone Age vs Prediction" — ya aplicado directamente en la figura regenerada (ver abajo), no hace falta editarlo a mano.

**Nueva Figura 8** (identidad + regresión + IC 95%, terminología corregida): [`figures/figura8_mejorada_identidad_regresion.png`](figures/figura8_mejorada_identidad_regresion.png)

---

## Comentario 7 — Análisis estadístico externo (IC, RMSE, mediana, sesgo, estratificación, pruebas pareadas)

### RSNA (n=1,393) — no afectado por el bug, datos sin cambios

| Backbone | MAE | RMSE | Mediana AE | Sesgo | DE | ≤6m | ≤12m |
|---|---:|---:|---:|---:|---:|---:|---:|
| F-ResNet50 | 16.17 | 22.44 | 12.48 | −2.79 | 22.27 | 25.2% | 48.3% |
| F-VGG16 | 37.19 | 43.74 | 36.80 | −18.23 | 39.76 | 7.0% | 11.7% |
| F-DenseNet121 | 14.18 | 20.17 | 11.10 | +0.44 | 20.17 | 29.1% | 53.4% |
| F-InceptionV3 | 13.59 | 19.88 | 10.03 | −3.71 | 19.53 | 32.7% | 57.5% |

### MEX (n=99) — **corregido**, contra `bone_age`

| Backbone | MAE | RMSE | Mediana AE | Sesgo | DE | ≤6m | ≤12m |
|---|---:|---:|---:|---:|---:|---:|---:|
| F-ResNet50 | 19.33 | 25.00 | 13.68 | +2.08 | 24.91 | 21.2% | 44.4% |
| F-VGG16 | 40.04 | 46.40 | 37.34 | −14.81 | 43.98 | 2.0% | 14.1% |
| F-DenseNet121 | 16.38 | 22.19 | 13.59 | **+5.68** | 21.45 | 22.2% | 44.4% |
| F-InceptionV3 | 17.28 | 22.99 | 15.08 | **−2.37** | 22.87 | 23.2% | 43.4% |

**Sesgo — hallazgo clínicamente relevante que se mantiene tras la corrección:** aunque el MAE de DenseNet121 y InceptionV3 en MEX es casi idéntico (16.38 vs. 17.28), su **sesgo** es muy distinto — DenseNet121 **sobreestima sistemáticamente** la edad ósea (+5.68 meses en promedio), mientras que InceptionV3 subestima levemente (−2.37 meses), mucho más cerca de cero. Esto no se ve en el MAE agregado y es justo el tipo de hallazgo que el revisor pide ("signed error or mean bias").

### Pruebas pareadas (Wilcoxon + bootstrap pareado 10,000 remuestreos, corrección de Holm sobre 6 comparaciones)

**RSNA (n=1,393):**

| Par | ΔMAE | IC 95% | p Wilcoxon (Holm) | p Bootstrap (Holm) | Sig. |
|---|---:|---:|---:|---:|:---:|
| ResNet50 vs VGG16 | −21.03 | [−22.30, −19.74] | <0.001 | <0.001 | ✅ |
| ResNet50 vs DenseNet121 | +1.99 | [1.40, 2.58] | <0.001 | <0.001 | ✅ |
| ResNet50 vs InceptionV3 | +2.58 | [1.89, 3.24] | <0.001 | <0.001 | ✅ |
| VGG16 vs DenseNet121 | +23.01 | [21.71, 24.32] | <0.001 | <0.001 | ✅ |
| VGG16 vs InceptionV3 | +23.60 | [22.31, 24.88] | <0.001 | <0.001 | ✅ |
| **DenseNet121 vs InceptionV3** | +0.59 | [−0.05, 1.19] | 0.050 | 0.074 | ❌ |

**MEX (n=99), corregido:**

| Par | ΔMAE | IC 95% | p Wilcoxon (Holm) | p Bootstrap (Holm) | Sig. |
|---|---:|---:|---:|---:|:---:|
| ResNet50 vs VGG16 | −20.71 | [−25.29, −16.16] | <0.001 | <0.001 | ✅ |
| ResNet50 vs DenseNet121 | +2.95 | [0.37, 5.51] | 0.095 | 0.066 | ❌ |
| ResNet50 vs InceptionV3 | +2.05 | [−1.07, 5.13] | 0.706 | 0.398 | ❌ |
| VGG16 vs DenseNet121 | +23.66 | [18.78, 28.60] | <0.001 | <0.001 | ✅ |
| VGG16 vs InceptionV3 | +22.76 | [18.11, 27.20] | <0.001 | <0.001 | ✅ |
| **DenseNet121 vs InceptionV3** | −0.90 | [−3.56, 1.70] | 0.706 | 0.504 | ❌ |

**Conclusión estadística:** DenseNet121 y InceptionV3 siguen siendo estadísticamente indistinguibles en MEX incluso con los datos corregidos (p=0.71/0.50). El punto estimado cambió de favorito (antes InceptionV3, ahora DenseNet121), pero **ninguno de los dos "gana" formalmente** — es la misma conclusión de equivalencia práctica que ya existía para RSNA, ahora también confirmada para MEX con los datos correctos.

**📝 Reemplaza esto** (Sección 3.4, párrafo 2):
> "Within this demographically distinct dataset, the F-InceptionV3 model demonstrated the greatest robustness, achieving the lowest MAE (15.98 months). F-ResNet50 (16.22 months) and F-DenseNet121 (16.30 months) followed closely in performance. In contrast, F-VGG16 showed poor generalization, resulting in a substantially higher error (27.98 months)."

**Por esto:**
> "Within this demographically distinct dataset, F-DenseNet121 achieved the numerically lowest MAE (16.38 months), followed closely by F-InceptionV3 (17.28 months) and F-ResNet50 (19.33 months); paired statistical testing (below) confirms these three do not differ significantly in MAE. F-DenseNet121 showed a substantial positive bias (mean signed error = +5.68 months), systematically overestimating bone age in this cohort, whereas F-InceptionV3 showed a smaller, negative bias (−2.37 months). In contrast, F-VGG16 showed a pronounced generalization failure, with a substantially higher error (40.04 months) consistent with the regression collapse described in Section 3.4.1."

**📝 Agrega esto** justo después de la Tabla 9, antes del párrafo de la Figura 8. "Table X" y "Table Y" abajo son las dos tablas nuevas que hay que insertar — la numeración real (¿Tabla 9b/9c? ¿se corre el Tabla 10 de literatura a Tabla 11?) depende de cómo se quiera renumerar el resto del manuscrito; aquí van con contenido completo, listas para pegar, solo falta decidir el número:

> "Table X reports additional error statistics — RMSE, median absolute error, mean signed error (bias), error standard deviation, and the proportion of predictions within clinically meaningful thresholds of 6 and 12 months — computed on both the RSNA (n = 1,393) and Mexican (n = 99) validation sets. Paired bootstrap resampling (10,000 iterations) and Wilcoxon signed-rank tests with Holm correction for multiple comparisons (Table Y) confirm that F-DenseNet121 and F-InceptionV3 do not differ significantly in MAE on either dataset (RSNA: ΔMAE = 0.59 months, 95% CI [−0.05, 1.19]; Mexican cohort: ΔMAE = −0.90 months, 95% CI [−3.56, 1.70]), despite the numerically lowest MAE alternating between the two architectures across datasets. Given this statistical equivalence, backbone selection is better justified by training-time efficiency (Section 3.2) than by external MAE alone."

**Table X — contenido real** (RSNA y MEX combinados en una tabla, lista para insertar):

| Backbone | Dataset | MAE | RMSE | Median AE | Bias | SD | ≤6m | ≤12m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| F-ResNet50 | RSNA | 16.17 | 22.44 | 12.48 | −2.79 | 22.27 | 25.2% | 48.3% |
| F-VGG16 | RSNA | 37.19 | 43.74 | 36.80 | −18.23 | 39.76 | 7.0% | 11.7% |
| F-DenseNet121 | RSNA | 14.18 | 20.17 | 11.10 | **+0.44** | 20.17 | 29.1% | 53.4% |
| F-InceptionV3 | RSNA | **13.59** | **19.88** | **10.03** | −3.71 | **19.53** | **32.7%** | **57.5%** |
| F-ResNet50 | MEX | 19.33 | 25.00 | 13.68 | **+2.08** | 24.91 | 21.2% | **44.4%** |
| F-VGG16 | MEX | 40.04 | 46.40 | 37.34 | −14.81 | 43.98 | 2.0% | 14.1% |
| F-DenseNet121 | MEX | **16.38** | **22.19** | **13.59** | +5.68 | **21.45** | 22.2% | **44.4%** |
| F-InceptionV3 | MEX | 17.28 | 22.99 | 15.08 | −2.37 | 22.87 | **23.2%** | 43.4% |

**Table Y — contenido real** (pruebas pareadas, RSNA y MEX combinados; las 6 comparaciones de RSNA + las 6 de MEX, corrección de Holm aplicada dentro de cada dataset por separado):

| Dataset | Par | ΔMAE | IC 95% | p Wilcoxon (Holm) | p Bootstrap (Holm) | Sig. |
|---|---|---:|---:|---:|---:|:---:|
| RSNA | ResNet50 vs VGG16 | −21.03 | [−22.30, −19.74] | <0.001 | <0.001 | ✅ |
| RSNA | ResNet50 vs DenseNet121 | +1.99 | [1.40, 2.58] | <0.001 | <0.001 | ✅ |
| RSNA | ResNet50 vs InceptionV3 | +2.58 | [1.89, 3.24] | <0.001 | <0.001 | ✅ |
| RSNA | VGG16 vs DenseNet121 | +23.01 | [21.71, 24.32] | <0.001 | <0.001 | ✅ |
| RSNA | VGG16 vs InceptionV3 | +23.60 | [22.31, 24.88] | <0.001 | <0.001 | ✅ |
| RSNA | **DenseNet121 vs InceptionV3** | +0.59 | [−0.05, 1.19] | 0.050 | 0.074 | ❌ |
| MEX | ResNet50 vs VGG16 | −20.71 | [−25.29, −16.16] | <0.001 | <0.001 | ✅ |
| MEX | ResNet50 vs DenseNet121 | +2.95 | [0.37, 5.51] | 0.095 | 0.066 | ❌ |
| MEX | ResNet50 vs InceptionV3 | +2.05 | [−1.07, 5.13] | 0.706 | 0.398 | ❌ |
| MEX | VGG16 vs DenseNet121 | +23.66 | [18.78, 28.60] | <0.001 | <0.001 | ✅ |
| MEX | VGG16 vs InceptionV3 | +22.76 | [18.11, 27.20] | <0.001 | <0.001 | ✅ |
| MEX | **DenseNet121 vs InceptionV3** | −0.90 | [−3.56, 1.70] | 0.706 | 0.504 | ❌ |

### Estratificación por sexo y edad

Disponible completa en `results_json/extended_stats_results.json` (por backbone × dataset × sexo × grupo etario). Ejemplo — F-DenseNet121 en MEX:

| Grupo | n | MAE | Sesgo |
|---|---:|---:|---:|
| Sexo F | ~50 | ver JSON | ver JSON |
| Sexo M | ~49 | ver JSON | ver JSON |
| 0–6 años | pocos casos | — | — |
| 6–12 años | mayoría | — | — |
| 12–19 años | resto | — | — |

*(No transcribí la tabla completa de 4 backbones × 2 datasets × 2 sexos × 3 grupos de edad aquí para no saturar el documento — está completa y lista para usar en el JSON. Avísame si la quieres formateada como tabla para pegar directo.)*

---

## Comentario 8 — Gráficos de dispersión + diagnóstico de F-VGG16

**Nueva Figura 8** (identidad, regresión, IC 95%): [`figures/figura8_mejorada_identidad_regresion.png`](figures/figura8_mejorada_identidad_regresion.png)

| Backbone | r | Pendiente | Intercepto |
|---|---:|---:|---:|
| F-ResNet50 | 0.82 | 0.62 | 49.9 |
| **F-VGG16** | **−0.06** | **−0.00** | 110.3 |
| F-DenseNet121 | 0.87 | 0.71 | 41.4 |
| F-InceptionV3 | 0.85 | 0.69 | 36.7 |

**Nueva figura Bland-Altman**: [`figures/figura8_bland_altman.png`](figures/figura8_bland_altman.png)

| Backbone | Sesgo medio | DE | Límites de acuerdo 95% |
|---|---:|---:|---:|
| F-ResNet50 | +2.08 | 24.91 | [−46.7, +50.9] |
| F-VGG16 | −14.81 | 43.98 | [−101.0, +71.4] |
| F-DenseNet121 | +5.68 | 21.45 | [−36.3, +47.7] |
| F-InceptionV3 | −2.37 | 22.87 | [−47.2, +42.4] |

### Diagnóstico de F-VGG16 (con evidencia, no solo descripción)

Revisando `experiments/26/training_history/` (intacto, no depende de nada de esta sesión):

- **No es entrenamiento insuficiente**: `EarlyStopping(patience=4)` detuvo la fase de fusión en la época 7 de 20, y el fine-tuning en la época 5 de 10 — pero `val_loss`/`val_mae` **se estancan desde la primera época registrada** (val_mae oscila entre 34.6 y 40.5 en las 12 épocas efectivamente entrenadas, sin tendencia de mejora). Más épocas no habrían ayudado.
- **No es fuga de datos**: una fuga produciría desempeño artificialmente *bueno*, no un colapso.
- **Es, con alta probabilidad, una falla de optimización específica de la arquitectura**: con `WEIGHTS=None` (sin ImageNet, igual que los otros 3 backbones), VGG16 es la única de las cuatro **sin Batch Normalization ni conexiones residuales/densas** (ResNet50 tiene residuales, DenseNet121 tiene conexiones densas, InceptionV3 usa BatchNorm extensivamente). La pendiente de regresión prácticamente nula (−0.0046) y la correlación negativa (r=−0.06) confirman cuantitativamente que el optimizador convergió a una solución trivial (predecir cerca de la media, ~110 meses) en vez de aprender la relación imagen→edad.

**📝 Agrega esto** justo después del párrafo ya existente sobre el colapso de F-VGG16 (Sección 3.4):
> "The collapse of F-VGG16 was investigated using its full training history. Early stopping halted training at epoch 7/20 (fusion phase) and epoch 5/10 (fine-tuning), but validation loss and MAE plateaued from the first recorded epoch onward (val MAE oscillating between 34.6 and 40.5 months with no improving trend across 12 total epochs), indicating that additional training would not have resolved the issue. Data leakage is unlikely, as it would be expected to produce artificially strong, not collapsed, performance. The most plausible explanation is an architecture-specific optimization failure: unlike the other three backbones, VGG16 was trained from scratch (WEIGHTS = None, identical across all four backbones) without batch normalization or residual/dense connections — mechanisms present in ResNet50, DenseNet121, and InceptionV3, respectively, known to substantially ease gradient-based optimization from random initialization. This is confirmed quantitatively by the near-zero regression slope (−0.0046) and near-zero correlation (r = −0.06, Figure 8) between predicted and true bone age for F-VGG16, indicating the optimizer converged to a trivial solution near the mean training age rather than learning a genuine image-to-age relationship."
