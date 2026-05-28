# Resumen de Evaluación — Todos los Experimentos

> Actualizado: 2026-04-29 · Scripts 07 (val estándar), 08 (mex-validation), 09 (análisis de desempeño), 10 (análisis por rango de edad)
> Val estándar: 1,393 imágenes · Mex-validation: 98 imágenes

---

## 1. Resultados globales

| Exp | Tipo | Dataset | CV MAE | Val MAE | Mex MAE | Fusión MAE | ±12m | ±24m | Sesgo |
|-----|------|---------|--------|---------|---------|-----------|------|------|-------|
| 33 | `simple_cnn` | recortado | 26.4 m | 39.2 m | 35.9 m | 34.5 m | 11.8% | 28.6% | −31.6 m |
| 36 | `simple_cnn` | completo | 34.3 m | 30.2 m | 22.2 m | 25.5 m | 22.5% | 42.7% | −20.2 m |
| 39 | `simple_cnn` | balanceado | 29.6 m | 43.5 m | 35.9 m | 41.8 m | 9.2% | 21.7% | −35.8 m |
| **42** | **`simple_cnn`** | **completo+libre** | **33.1 m** | **24.1 m** | **20.0 m** | **20.1 m** | **30.7%** | **55.5%** | **−11.0 m** |
| **34** | **`backbone`** | **recortado** | **27.6 m** | **14.6 m** | **17.6 m** | **9.2 m** | **50.9%** | **78.8%** | **+0.8 m** |
| **37** | **`backbone`** | **completo** | **29.7 m** | **15.4 m** | **16.7 m** | **6.8 m** | **48.9%** | **77.0%** | **+1.3 m** |
| **40** | **`backbone`** | **balanceado** | **26.2 m** | **15.1 m** | **13.9 m** | **9.0 m** | **47.1%** | **79.1%** | **−0.7 m** |
| 35 | `bbone_vec` | recortado | 26.6 m | 36.7 m | 23.4 m | 27.6 m | 14.7% | 31.0% | −28.0 m |
| 38 | `bbone_vec` | completo | 31.4 m | 40.0 m | 28.0 m | 30.3 m | 13.4% | 28.8% | −31.6 m |
| 41 | `bbone_vec` | balanceado | 26.7 m | 35.0 m | 27.3 m | 26.9 m | 15.5% | 32.9% | −26.5 m |
| **43** | **`bbone_vec`** | **completo+libre** | **31.5 m** | **23.0 m** | **18.5 m** | **17.3 m** | **28.8%** | **58.5%** | **−12.8 m** |
| **44** | **`unified_cnn`** | **recortado** | **19.5 m** | **19.0 m** | **16.9 m** | **16.0 m** | **35.9%** | **68.0%** | **−1.8 m** |
| 45 | `unified_cnn` | completo | 30.0 m | 29.0 m | 21.0 m | 25.6 m | 24.3% | 43.1% | −18.5 m |
| **46** | **`unified_cnn`** | **balanceado** | **23.2 m** | **21.0 m** | **21.9 m** | **16.2 m** | **35.3%** | **63.4%** | **+0.9 m** |

> **Negrita** = mejor resultado por arquitectura · **Sesgo:** + sobreestima, − subestima

---

## 2. Análisis por segmento (script 09)

| Exp | Tipo | Dataset | Mejor segmento | MAE | Peor segmento | MAE |
|-----|------|---------|----------------|-----|---------------|-----|
| 33 | `simple_cnn` | recortado | Pulgar | 18.9 m | Meñique | 33.0 m |
| 36 | `simple_cnn` | completo | Pulgar | 19.7 m | Meñique | 37.7 m |
| 39 | `simple_cnn` | balanceado | Muñeca | 21.4 m | Meñique | 31.1 m |
| 42 | `simple_cnn` | completo+libre | Pulgar | 19.6 m | Meñique | 35.6 m |
| 34 | `backbone` | recortado | Medio | 15.9 m | Meñique | 24.3 m |
| 37 | `backbone` | completo | Medio | 20.5 m | Pulgar | 54.9 m |
| 40 | `backbone` | balanceado | Medio | 17.7 m | Muñeca | 29.5 m |
| 35 | `bbone_vec` | recortado | Meñique | 15.4 m | Muñeca | 23.2 m |
| 38 | `bbone_vec` | completo | Pulgar | 14.3 m | Medio | 34.9 m |
| 41 | `bbone_vec` | balanceado | Pulgar | 17.2 m | Muñeca | 19.9 m |
| 43 | `bbone_vec` | completo+libre | Pulgar | 19.8 m | Meñique | 27.4 m |

> `unified_cnn` no aplica — no tiene modelos de segmento separados.
> El **meñique** es el segmento más difícil en la mayoría de arquitecturas.

---

## 3. Análisis por rango de edad (script 10)

> Resultados corregidos 2026-05-28. Fix aplicado en `src/10_age_range_analysis.py`: el dataset MEX ahora usa 100/100 registros con edades correctamente convertidas a meses (antes: 75/100, con conversión errónea de años). Los sesgos combinados (RSNA+MEX) difieren de los sesgos RSNA-only de la sección 1.

| Exp | Tipo | Dataset | Mejor rango | Peor rango | ±12m | Sesgo combinado |
|-----|------|---------|-------------|------------|------|-----------------|
| 33 | `simple_cnn` | recortado | 36–48 m (3–4 a) | 228–240 m (19–20 a) | 12.6% | −37.4 m |
| 36 | `simple_cnn` | completo | 60–72 m (5–6 a) | 228–240 m (19–20 a) | 24.1% | −25.7 m |
| 39 | `simple_cnn` | balanceado | 36–48 m (3–4 a) | 228–240 m (19–20 a) | 9.9% | −41.5 m |
| 42 | `simple_cnn` | completo+libre | 72–84 m (6–7 a) | 228–240 m (19–20 a) | 31.6% | −16.7 m |
| 34 | `backbone` | recortado | 96–108 m (8–9 a) | 0–12 m (0–1 a)¹ | 53.2% | −4.9 m |
| 37 | `backbone` | completo | 96–108 m (8–9 a) | 228–240 m (19–20 a) | 50.9% | −4.4 m |
| 40 | `backbone` | balanceado | 96–108 m (8–9 a) | 228–240 m (19–20 a) | 49.1% | −6.4 m |
| 35 | `bbone_vec` | recortado | 48–60 m (4–5 a) | 228–240 m (19–20 a) | 15.8% | −33.6 m |
| 38 | `bbone_vec` | completo | 36–48 m (3–4 a) | 228–240 m (19–20 a) | 14.3% | −37.2 m |
| 41 | `bbone_vec` | balanceado | 48–60 m (4–5 a) | 228–240 m (19–20 a) | 16.4% | −32.2 m |
| 43 | `bbone_vec` | completo+libre | 12–24 m (1–2 a)² | 228–240 m (19–20 a) | 30.6% | −18.5 m |
| 44 | `unified_cnn` | recortado | 96–108 m (8–9 a) | 228–240 m (19–20 a) | 37.4% | −7.5 m |
| 45 | `unified_cnn` | completo | 0–12 m (0–1 a)¹ | 228–240 m (19–20 a) | 25.4% | −24.1 m |
| 46 | `unified_cnn` | balanceado | 108–120 m (9–10 a) | 228–240 m (19–20 a) | 36.9% | −4.8 m |

> ¹ Bin con solo 2 imágenes (RSNA val: 3 m y 6 m) — estadísticamente no representativo.
> ² Bin con 11 imágenes — muestra pequeña.

### Patrón de sesgo por arquitectura (RSNA+MEX combinados)

| Arquitectura | Sesgo típico | Patrón |
|---|---|---|
| `simple_cnn` (congelado) | −26 a −42 m | Subestima sistemáticamente |
| `simple_cnn` (libre) | — | Pendiente rerun |
| `backbone` | −4 a −7 m | Leve subestimación al combinar con MEX |
| `bbone_vec` (congelado) | −32 a −37 m | Subestima sistemáticamente |
| `bbone_vec` (libre) | — | Pendiente rerun |
| `unified_cnn` (recortado/bal.) | −5 a −8 m | Comparable a `backbone` |
| `unified_cnn` (completo) | −24 m | Dataset completo introduce sesgo |

**Cambio clave vs. análisis anterior:** el peor rango para `backbone` y `unified_cnn` es ahora **228–240 m (adolescentes tardíos)**, no 12–24 m. El resultado anterior estaba distorsionado por el bug del MEX que concentraba casos con edades erróneas en rangos bajos.

---

## 4. Ranking general por Val MAE

| # | Exp | Tipo | Dataset | Val MAE | Mex MAE | Fusión MAE | ±12m | Sesgo |
|---|-----|------|---------|---------|---------|-----------|------|-------|
| 🥇 | **34** | `backbone` | recortado | **14.6 m** | 17.6 m | **9.2 m** | 50.9% | +0.8 m |
| 🥈 | **40** | `backbone` | balanceado | **15.1 m** | **13.9 m** | 9.0 m | 47.1% | −0.7 m |
| 🥉 | **37** | `backbone` | completo | 15.4 m | 16.7 m | **6.8 m** | 48.9% | +1.3 m |
| 4 | **44** | `unified_cnn` | recortado | 19.0 m | **16.9 m** | 16.0 m | 35.9% | −1.8 m |
| 5 | **46** | `unified_cnn` | balanceado | 21.0 m | 21.9 m | 16.2 m | 35.3% | +0.9 m |
| 6 | **43** | `bbone_vec` | completo+libre | 23.0 m | 18.5 m | 17.3 m | 28.8% | −12.8 m |
| 7 | **42** | `simple_cnn` | completo+libre | 24.1 m | 20.0 m | 20.1 m | 30.7% | −11.0 m |
| 8 | 45 | `unified_cnn` | completo | 29.0 m | 21.0 m | 25.6 m | 24.3% | −18.5 m |
| 9 | 36 | `simple_cnn` | completo | 30.2 m | 22.2 m | 25.5 m | 22.5% | −20.2 m |
| 10 | 35 | `bbone_vec` | recortado | 36.7 m | 23.4 m | 27.6 m | 14.7% | −28.0 m |
| 11 | 41 | `bbone_vec` | balanceado | 35.0 m | 27.3 m | 26.9 m | 15.5% | −26.5 m |
| 12 | 33 | `simple_cnn` | recortado | 39.2 m | 35.9 m | 34.5 m | 11.8% | −31.6 m |
| 13 | 38 | `bbone_vec` | completo | 40.0 m | 28.0 m | 30.3 m | 13.4% | −31.6 m |
| 14 | 39 | `simple_cnn` | balanceado | 43.5 m | 35.9 m | 41.8 m | 9.2% | −35.8 m |

---

## 5. Conclusiones

1. **`backbone` domina en precisión** (14–15 m Val MAE) con sesgo prácticamente nulo. La fusión escalar (6–9 m fusión MAE) es la más efectiva.

2. **Descongelar extractores mejora drásticamente** `simple_cnn` (30→24 m, +6 m) y `bbone_vec` (40→23 m, +17 m). Los exps 42 y 43 son el hallazgo más relevante de la Fase 5.

3. **`unified_cnn` es el segundo mejor** en recortado y balanceado (19–21 m), competitivo con `backbone` en sesgo (±2 m) sin necesitar pipeline de dos etapas.

4. **El rango 96–108 m (8–9 años) es universalmente el más predecible** para `backbone` y `unified_cnn`. Los extremos (<24 m y >216 m) son críticos para todas las arquitecturas.

5. **El dataset balanceado no ayuda a `simple_cnn`** pero sí mantiene la calidad en `backbone` y `unified_cnn`, sugiriendo que el balanceo beneficia modelos con mejor calibración.

6. **El meñique es el segmento más difícil** para casi todas las arquitecturas; el medio y el pulgar son los más predecibles.
