# Experimentos de Optimización — Fase 6 (Exps 47+)

> Documento vivo: se llena a medida que completan los experimentos.
> Baselines: **Exp 34** (`backbone`) y **Exp 43** (`backbone_vectors` libre) — los mejores de cada arquitectura.

---

## Baselines de referencia

| Exp | Arquitectura | Dataset | Val MAE | Mex MAE | ±12m | Sesgo |
|-----|-------------|---------|:-------:|:-------:|:----:|:-----:|
| **34** | `backbone` | recortado | **14.6 m** | 17.6 m | 50.9% | +0.8 m |
| **43** | `backbone_vectors` libre | completo | **23.0 m** | 18.5 m | 28.8% | −12.8 m |

---

## 2.1 · Experimento de género (`USE_GENDER = False`)

**Pregunta:** ¿Es el género indispensable en estas arquitecturas?
**Variable cambiada:** `USE_GENDER = False` (todo lo demás igual al baseline)

| Exp | Arquitectura | USE_GENDER | Val MAE | Mex MAE | ±12m | Sesgo | Δ vs baseline |
|-----|-------------|:----------:|:-------:|:-------:|:----:|:-----:|:-------------:|
| 34 | `backbone` | True *(baseline)* | 14.6 m | 17.6 m | 50.9% | +0.8 m | — |
| ⏳ 47 | `backbone` | **False** | — | — | — | — | — |
| 43 | `backbone_vectors` libre | True *(baseline)* | 23.0 m | 18.5 m | 28.8% | −12.8 m | — |
| ⏳ 48 | `backbone_vectors` libre | **False** | — | — | — | — | — |

---

## 2.2 · Experimento de LR y épocas de fusión

**Pregunta:** ¿El LR de fusión actual (5e-4 / 20 épocas) es óptimo?
**Variable cambiada:** `LEARNING_RATE` y `FUSION_EPOCHS`

### backbone (base: Exp 34)

| Exp | LR fusión | Épocas fusión | Val MAE | Mex MAE | ±12m | Sesgo | Δ vs 34 |
|-----|:---------:|:-------------:|:-------:|:-------:|:----:|:-----:|:-------:|
| 34 | 5e-4 *(baseline)* | 20 | 14.6 m | 17.6 m | 50.9% | +0.8 m | — |
| ⏳ 49 | **1e-4** | **10** | — | — | — | — | — |
| ⏳ 50 | **1e-3** | **30** | — | — | — | — | — |
| ⏳ 51 | *intermedio* | *intermedio* | — | — | — | — | — |

### backbone_vectors (base: Exp 43)

| Exp | LR fusión | Épocas fusión | Val MAE | Mex MAE | ±12m | Sesgo | Δ vs 43 |
|-----|:---------:|:-------------:|:-------:|:-------:|:----:|:-----:|:-------:|
| 43 | 5e-4 *(baseline)* | 20 | 23.0 m | 18.5 m | 28.8% | −12.8 m | — |
| ⏳ 52 | **1e-4** | **10** | — | — | — | — | — |
| ⏳ 53 | **1e-3** | **30** | — | — | — | — | — |
| ⏳ 54 | *intermedio* | *intermedio* | — | — | — | — | — |

> Solo se agregan puntos intermedios (51, 54) si los extremos muestran señal clara de mejora.

---

## 2.3 · Experimento de tamaño de imagen (`IMAGE_SIZE`)

**Pregunta:** ¿224×224 mejora respecto a 112×112?
**Condición:** ejecutar solo si 2.1–2.2 están saturados (mejora < 0.5 m MAE)
**Nota:** el tiempo de entrenamiento podría acercarse al límite del job (4 días) — monitorear

| Exp | Arquitectura | IMAGE_SIZE | Val MAE | Mex MAE | ±12m | Sesgo | Δ vs baseline |
|-----|-------------|:----------:|:-------:|:-------:|:----:|:-----:|:-------------:|
| 34 | `backbone` | 112×112 *(baseline)* | 14.6 m | 17.6 m | 50.9% | +0.8 m | — |
| ⏳ 55 | `backbone` | **224×224** | — | — | — | — | — |
| 43 | `backbone_vectors` | 112×112 *(baseline)* | 23.0 m | 18.5 m | 28.8% | −12.8 m | — |
| ⏳ 56 | `backbone_vectors` | **224×224** | — | — | — | — | — |

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
| 1 | Género | 47, 48 | ~80 h | Siempre |
| 2 | LR extremos | 49–53 | ~160 h | Siempre |
| 3 | LR intermedio | 51, 54 | ~80 h | Si hay señal en extremos |
| 4 | Imagen 224×224 | 55, 56 | ~160 h | Si 1–3 saturados |
| 5 | Datos clínicos | TBD | variable | Sujeto a disponibilidad |
