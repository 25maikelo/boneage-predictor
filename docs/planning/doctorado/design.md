# Design: Doctorado

> Cómo se piensa satisfacer los requirements de [requirements.md](requirements.md). A diferencia
> de ese documento (que no cambia salvo que cambie el protocolo oficial), este se actualiza cada
> vez que se toma una decisión de diseño nueva. Las decisiones ya tomadas quedan registradas con
> su justificación; las preguntas abiertas quedan listadas explícitamente para no perderlas.

## 1. Arquitectura actual (heredada de la maestría)

```
radiografía raw
     │
     ▼
02_frame_and_zoom.py       Otsu + contorno + minAreaRect → rotación + recorte
     │                     (config/preprocessing.py: blur, dilatación, tolerancia de ángulo,
     │                      área mínima de recorte)
     ▼
03_histogram_equalization.py   CLAHE + mezcla con original + ajuste final de contraste
     │                     (config/preprocessing.py: clipLimit, tileGridSize, mezcla, alpha/beta)
     ▼
01/04: segmentación        U-Net (encoder MobileNetV2) → 4 máscaras (meñique, medio, pulgar,
     │                     muñeca) → recorte de cada región, modo spatial o cropped
     ▼
4 modelos CNN (uno por segmento, DenseNet121/ResNet50/VGG16/InceptionV3)
     │
     ▼
Fusión: concatenación de las 4 salidas + sexo → capas densas → edad ósea (MAE)
```

**Limitación central que motiva el doctorado** (Requirement 4, criterios 4-5): la fusión es
concatenación simple de 4 vectores + una sola variable categórica (sexo). No hay mecanismo de
atención ni de fusión profunda entre segmentos, y no hay espacio para variables clínicas
adicionales (percentiles de crecimiento, etc.). Esto es lo que la hipótesis doctoral propone
reemplazar.

## 2. Arquitectura objetivo: fusión multimodal (Requirement 4, criterios 4-5)

**Aún no implementada. Esta sección documenta las opciones a comparar, no una decisión tomada.**

Tres estrategias candidatas (terminología estándar de fusión multimodal):

| Estrategia | Dónde se fusiona | Ventaja | Riesgo |
|---|---|---|---|
| Temprana (early fusion) | A nivel de entrada/features de bajo nivel, antes de los modelos por segmento | Permite que el modelo aprenda interacciones imagen-clínica desde el inicio | Requiere rediseñar los 4 modelos de segmento, no solo la capa de fusión |
| Intermedia (intermediate fusion) | Combinando los embeddings de cada segmento con las variables clínicas antes de la capa final | Mantiene los 4 modelos de segmento reutilizables, cambia solo la cabeza de fusión | Sigue dependiendo de que los embeddings de segmento ya sean informativos |
| Tardía (late fusion) | Combinando las predicciones finales de cada rama (como hoy, pero con un mecanismo más rico que concatenación+MLP, p. ej. atención o gating) | Cambio incremental sobre el sistema actual, más fácil de comparar de forma controlada | Limitada estructuralmente: no puede capturar interacciones que ya se perdieron en cada rama independiente |

Requirement 4 pide comparar al menos dos de las tres con evidencia cuantitativa antes de elegir.

### Preguntas abiertas (bloquean el diseño detallado de la fusión)

- **¿Qué variables clínicas no identificativas estarán disponibles?** Depende de Requirement 6
  (OP1, la base de datos mexicana ampliada). Sin saber qué variables hay (¿solo sexo y edad
  cronológica? ¿percentiles de crecimiento? ¿antecedentes?), no se puede fijar la dimensión de
  entrada del bloque clínico ni decidir si normalizar/embeber cada variable distinto.
- **¿El modelo de fusión se entrena end-to-end o por etapas?** El sistema actual entrena cada
  segmento por separado y luego fusiona (congelando o no las capas base). Un diseño multimodal
  más profundo podría requerir entrenamiento conjunto desde el inicio, lo que cambia el costo
  computacional y el riesgo de overfitting con un dataset de este tamaño.
- **¿Qué pasa con el género como variable clínica ahora que hay más variables?** Hoy el género
  entra como un escalar binario concatenado al final. Si se añaden más variables clínicas,
  decidir si todas entran igual o si alguna necesita tratamiento especial (p. ej. percentiles
  normalizados por tabla de crecimiento).

## 3. Decisiones de diseño ya tomadas (Requirement 3 / 4, sub-hilo de preprocesamiento y segmentación)

Estas decisiones están implementadas, verificadas y documentadas en detalle en
[`fase_segmentacion.md`](fase_segmentacion.md) y
[`docs/results/doctorado/preprocesamiento_diagnostico/`](../../results/doctorado/preprocesamiento_diagnostico/). Se
listan aquí solo como referencia rápida, sin repetir el detalle:

- **Canvas expandido antes de rotar** en `02_frame_and_zoom.py`, en vez de un canvas del mismo
  tamaño que la imagen original. Evita perder contenido cuando la mano está lejos del centro.
  Verificado: descartes 5.6% → 2.1% sobre 1,500 imágenes.
- **Tolerancia de 1° en el ángulo de frontera** (`cv2.minAreaRect` devuelve `angle=90` para
  rectángulos ya alineados; sin la tolerancia, esos casos se rotaban 90° de más).
- **`clipLimit=4.0` en CLAHE** en vez del default de OpenCV (2.0). Validado sobre las 12,811
  imágenes completas: 85.4% ganan contraste con el cambio.
- **`np.clip` en vez de `cv2.convertScaleAbs`** para el ajuste final de contraste/brillo, porque
  `convertScaleAbs` refleja en vez de saturar los valores negativos (afecta el fondo oscuro).
- **Parámetros centralizados en `config/preprocessing.py`**: todos los valores anteriores (más
  kernel de blur, kernel/iteraciones de dilatación, área mínima de recorte) son constantes
  nombradas y documentadas, no literales dentro de las funciones. Esto es lo que permite
  ablacionarlos sistemáticamente (Requirement 4, criterio 3), motivado directamente por la
  pregunta de la defensa de maestría sobre los parámetros de CLAHE/Otsu.
- **`equalizeHist` y sustracción de fondo morfológica, descartadas** como reemplazo de Otsu para
  el ~5% de casos de bajo contraste global: ninguna mejora el caso base sin introducir
  regresiones en otras imágenes (ver `preprocesamiento_diagnostico/diagnostico.md`).

## 4. Decisiones de diseño pendientes, por requirement

| Requirement | Decisión pendiente |
|---|---|
| 4 (grilla de segmentación) | Qué arquitecturas/modos de recorte entran en la grilla rehecha, y con qué criterio de selección combinado (Dice/IoU, MAE downstream, Dice en MEX) se elige el ganador. Ver `tasks.md` 3.4. |
| 4 (fusión multimodal) | Las tres preguntas abiertas de la sección 2. |
| 5 (evaluación estadística) | Qué prueba no paramétrica exacta (Wilcoxon pareado es la opción natural dado que `11_paired_validation.py` ya existe para comparaciones pareadas, pero no está decidido formalmente para este contexto) y contra qué baseline exacto (candidato: Exp 57, ver `docs/results/maestria/2026-07-22_experiments_master.md`). |
| 6 (base de datos mexicana) | De qué instituciones, cuántas imágenes, qué variables clínicas exactas, y si requiere un trámite de aprobación ética propio (distinto del protocolo general del doctorado). |
| 8 (prototipo) | Formato: CLI, servicio web, notebook interactivo. Ninguno decidido todavía. |

## 5. Principios de diseño heredados (siguen vigentes para todo el doctorado)

De `fase_segmentacion.md`, estos principios de reproducibilidad aplican a todo trabajo nuevo, no
solo a la etapa de preprocesamiento donde se originaron:

- Semilla fija (`random_state=42`) en todo split/muestreo.
- Cada corrida en su propio directorio versionado con `config.json`.
- Comando exacto de SLURM documentado por experimento.
- Datasets de comparación pineados (mismas imágenes para todas las variantes comparadas).
- Métricas crudas (`metrics.json`/`plot_data.json`) como fuente de verdad, nunca transcritas a
  mano.
- Mismo entorno conda (`master_env` según los scripts de `slurm/`) para todas las corridas
  comparables.
