# Experimentos de Optimización — Fase 6 (Exps 47+)

> Documento vivo: se llena a medida que completan los experimentos.
> Baselines: **Exp 37** (`backbone`, completo) y **Exp 43** (`backbone_vectors` libre, completo) — todos los experimentos nuevos usan `AGE_RANGE=(1,228)` para que ambas arquitecturas sean comparables sobre el mismo dataset.

---

## Baselines de referencia

| Exp | Arquitectura | Dataset | Val MAE | Mex MAE | Fusión MAE | ±12m | Sesgo |
|-----|-------------|---------|:-------:|:-------:|:----------:|:----:|:-----:|
| **34** *(referencia histórica)* | `backbone` | recortado | 14.6 m | 17.6 m | 9.2 m | 50.9% | +0.8 m |
| **37** | `backbone` | completo | **15.4 m** | 16.7 m | 6.8 m | 48.9% | +1.3 m |
| **43** | `backbone_vectors` libre | completo | **23.0 m** | 18.5 m | 17.3 m | 28.8% | −12.8 m |

> **Nota:** Exp 34 (recortado) es el ganador global por MAE en RSNA, pero usa `AGE_RANGE=(24,216)`, distinto del dataset de `backbone_vectors` (completo). Para que las ablaciones de género, LR y épocas sean comparables entre ambas arquitecturas sobre el mismo dataset, se adoptó **Exp 37** (completo) como baseline de `backbone`. Se mantiene Exp 34 como referencia histórica del mejor resultado absoluto.

---

## 2.1 · Experimento de género (`USE_GENDER = False`)

**Pregunta:** ¿Es el género indispensable en estas arquitecturas?
**Variable cambiada:** `USE_GENDER = False` (todo lo demás igual al baseline)

| Exp | Arquitectura | USE_GENDER | Val MAE | Mex MAE | ±12m | Sesgo | Δ Val MAE |
|-----|-------------|:----------:|:-------:|:-------:|:----:|:-----:|:---------:|
| 37 | `backbone` | True *(baseline)* | 15.4 m | 16.7 m | 48.9% | +1.3 m | — |
| ✅ 47 | `backbone` | **False** | 16.5 m | 16.9 m | 48.0% | −4.0 m | +1.1 m ↑ |
| 43 | `backbone_vectors` libre | True *(baseline)* | 23.0 m | 18.5 m | 28.8% | −12.8 m | — |
| ✅ 48 | `backbone_vectors` libre | **False** | 18.5 m | 16.2 m | 41.5% | −9.9 m | **−4.5 m ↓** |

---

## 2.2 · Experimento de LR y épocas de fusión

**Pregunta:** ¿El LR de fusión actual (1e-3 / 20 épocas) es óptimo?
**Variable cambiada:** `LEARNING_RATE` y `FUSION_EPOCHS`

> El baseline real (37/43) usa LR=1e-3. Por eso esta ablación cubre dos preguntas distintas:
> - **¿LR menor?** → exp 49/51 (1e-4, también con menos épocas)
> - **¿Más épocas con el mismo LR?** → exp 50/52 (1e-3, igual que baseline, pero 30 épocas)

### backbone (base: Exp 37)

| Exp | LR fusión | Épocas fusión | Val MAE | Mex MAE | ±12m | Sesgo | Δ Val MAE |
|-----|:---------:|:-------------:|:-------:|:-------:|:----:|:-----:|:---------:|
| 37 | 1e-3 *(baseline)* | 20 | 15.4 m | 16.7 m | 48.9% | +1.3 m | — |
| ✅ 49 | **1e-4** | **10** | 17.4 m | 17.0 m | 44.1% | −9.3 m | +2.0 m ↑ |
| ✅ 50 | 1e-3 *(igual)* | **30** | 16.8 m | 16.5 m | 48.0% | −3.7 m | +1.4 m ↑ |
| — | *intermedio* | *intermedio* | — | — | — | — | — |

### backbone_vectors (base: Exp 43)

| Exp | LR fusión | Épocas fusión | Val MAE | Mex MAE | ±12m | Sesgo | Δ Val MAE |
|-----|:---------:|:-------------:|:-------:|:-------:|:----:|:-----:|:---------:|
| 43 | 1e-3 *(baseline)* | 20 | 23.0 m | 18.5 m | 28.8% | −12.8 m | — |
| ✅ 51 | **1e-4** | **10** | 19.0 m | 16.1 m | 39.6% | −12.7 m | **−4.0 m ↓** |
| ✅ 52 | 1e-3 *(igual)* | **30** | 26.2 m | 20.0 m | 26.5% | −24.0 m | +3.2 m ↑↑ |
| — | *intermedio* | *intermedio* | — | — | — | — | — |

> Solo se agregan puntos intermedios si los extremos muestran señal clara de mejora.

---

## 2.2b · Combinación género + LR (`backbone_vectors`)

**Pregunta:** ¿La mejora de quitar género (exp 48, −4.5m) y la mejora de LR bajo (exp 51, −4.0m) son aditivas?
**Variables cambiadas:** `USE_GENDER=False` + `LEARNING_RATE=1e-4` + `FUSION_EPOCHS=10`

| Exp | USE_GENDER | LR | Épocas | Val MAE | Mex MAE | ±12m | Sesgo | Δ vs 43 |
|-----|:----------:|:--:|:------:|:-------:|:-------:|:----:|:-----:|:-------:|
| 43 | True *(base)* | 1e-3 | 20 | 23.0 m | 18.5 m | 28.8% | −12.8 m | — |
| 48 | **False** | 1e-3 | 20 | 18.5 m | 16.2 m | 41.5% | −9.9 m | −4.5 m |
| 51 | True | **1e-4** | **10** | 19.0 m | 16.1 m | 39.6% | −12.7 m | −4.0 m |
| ✅ 53 | **False** | **1e-4** | **10** | 18.3 m | 17.1 m | 41.3% | −11.7 m | −4.7 m |

---

## 2.3 · Experimento de tamaño de imagen (`IMAGE_SIZE`)

**Pregunta:** ¿224×224 mejora respecto a 112×112?
**Condición:** ejecutar solo si 2.1–2.2 están saturados (mejora < 0.5 m MAE)
**Nota:** el tiempo de entrenamiento podría acercarse al límite del job (4 días) — monitorear

| Exp | Arquitectura | IMAGE_SIZE | Val MAE | Mex MAE | ±12m | Sesgo | Δ vs baseline |
|-----|-------------|:----------:|:-------:|:-------:|:----:|:-----:|:-------------:|
| 37 | `backbone` | 112×112 *(baseline)* | 15.4 m | 16.7 m | 48.9% | +1.3 m | — |
| ⏳ 55 | `backbone` | **224×224** | — | — | — | — | — |
| 48 | `backbone_vectors` libre | 112×112 *(mejor bbone_vec)* | 18.5 m | 16.2 m | 41.5% | −9.9 m | — |
| ⏳ 56 | `backbone_vectors` libre | **224×224** | — | — | — | — | — |

---

## 2.4 · Datos clínicos adicionales

**Pregunta:** ¿Agregar talla/peso/z-score mejora la predicción?
**Condición:** verificar disponibilidad en datasets RSNA y MEX

| Dataset | Campos disponibles | Estado |
|---------|-------------------|--------|
| RSNA training | — | ⏳ Por verificar |
| RSNA validation | — | ⏳ Por verificar |
| MEX validation | — | ⏳ Por verificar |

---

## Resumen de prioridades

| Prioridad | Experimento | Exps | Costo aprox. | Condición |
|:---------:|-------------|------|:------------:|-----------|
| 1 | Género | 47, 48 | ~30 h | Siempre |
| 2 | LR extremos / épocas | 49, 50, 51, 52 | ~80 h | Siempre |
| 3 | LR/épocas intermedio | TBD | ~40 h | Si hay señal en extremos |
| 4 | Imagen 224×224 | 55, 56 | ~160 h | Si 1–3 saturados |
| 5 | Datos clínicos | TBD | variable | Sujeto a disponibilidad |
