# Análisis de Comentarios — Segunda Ronda de Revisión

> Fuentes: [`Reviewers.docx.pdf`](Reviewers.docx.pdf) (Reviewer 1, 20 comentarios, algunos con borrador de respuesta ya empezado) y [`technologies-4414175-limpio.docx (1).pdf`](technologies-4414175-limpio.docx%20%281%29.pdf) (estado actual del manuscrito, con cambios rastreados).
>
> Por cada comentario: **qué pide** el revisor, **qué ya tenemos** (en el manuscrito y/o en el proyecto), y **qué haría falta** (análisis nuevo, texto, o experimento). No se ejecutó ningún cambio de código ni de entrenamiento en este documento — es solo el mapeo, para decidir después qué se ataca primero.

---

## ⚠️ Hallazgos críticos que hay que ver antes que nada

### 1. La validación mexicana estaba comparando contra la variable equivocada (ya corregido en código, falta re-ejecutar)

Revisé `src/08_mex_validation.py` y encontré que, aunque el script carga y parsea correctamente `bone_age` (la edad ósea TW3, usada incluso para el histograma de edades del propio script), la comparación real contra las predicciones del modelo — MAE, scatter, muestras con saliencia — usaba `real_age` (edad cronológica) en su lugar. **Ya corregí el bug en el código** (línea 312 y siguientes, ahora usa `row["bone_age"]`), pero **todos los números de validación mexicana que aparecen hoy en el manuscrito (Tabla 9, Figura 8, y cualquier `plot_data.json` ya generado) siguen calculados contra la variable equivocada** y deben regenerarse volviendo a correr `08_mex_validation.py` para cada experimento antes de poder citarlos.

Esto conecta directamente con el **Comentario 4** (abajo): la Respuesta 4 en el borrador afirma *"we completely removed any mentions of chronological age"*, pero la Figura 8 del manuscrito actual (pág. 18) sigue tituladas *"Dispersion Real Age vs Prediction"* con eje *"Real Age (months)"* — es decir, la respuesta afirma algo que el manuscrito todavía no refleja, y además los datos subyacentes de esa figura pueden venir del bug de arriba.

### 2. Las Respuestas 13 y 14 están cruzadas en el borrador

En `Reviewers.docx.pdf`, la **Respuesta 13** (que debería responder al Comentario 13, sobre la definición matemática de la función de pérdida) está **vacía**. La **Respuesta 14** sí tiene contenido, pero ese contenido responde textualmente al Comentario 13 (habla de la función de pérdida y de por qué el MAE de entrenamiento excede al de validación) — **no** responde al Comentario 14 (que es sobre control de la comparación de arquitecturas: LR, pesos preentrenados, capas congeladas, medición empírica de tiempo/latencia/memoria). Antes de reenviar hay que mover esa respuesta al comentario correcto y redactar una respuesta real para el Comentario 14.

### 3. Ejemplo en vivo de artefacto de edición (para el Comentario 19)

En la página 3 del manuscrito actual: *"a fusion convolutional architecture for bone age estimation, **inspired based by theon the** Tanner–Whitehouse 3 (TW3) technique"* — es exactamente el tipo de texto fusionado por edición ("resultssaliency", "areshow", etc.) que el Comentario 19 pide eliminar. Confirma que la limpieza editorial completa sigue pendiente.

---

## Comentario por comentario

### Comentario 1 — Limpieza general / versión sin tracked changes

**Pide:** una versión del manuscrito sin cambios rastreados visibles, texto suavizado en las conclusiones sobre validez externa.

**Tenemos:** la Respuesta 1 en el borrador ya declara que se hizo una limpieza y que se "atenuaron" las conclusiones. Pero el PDF que tenemos (`technologies-4414175-limpio.docx (1).pdf`) **todavía muestra tachados y texto en rojo** en varias secciones (Sección 2.1, Tabla 10, Conclusiones) — es decir, la limpieza no está terminada pese a lo que dice la respuesta.

**Falta:** aceptar todos los cambios en Word y exportar una versión verdaderamente limpia. No es análisis nuevo, es un paso mecánico — pero bloqueante, porque el propio Comentario 1 dice que sin esto "no puede continuar la revisión científica".

---

### Comentario 2 — Conciliación de datos (14,236 registros)

**Pide:** que los 14,236 registros de RSNA cuadren exactamente contra 11,783 (balanceado) + 9,426/2,357 (split interno) + 1,425 (validación oficial), con diagrama de flujo de participantes y terminología consistente.

**Tenemos:** el texto limpio (no tachado) de la Sección 2.1 **ya reconcilia todo correctamente**: 12,611 (train oficial) = 11,783 (balanceado) + 828 (excluidas por filtro de edad); 11,783 → 9,426/2,357 (split 80/20); + 1,425 (validación oficial) + 200 (test oficial, no usado) = 14,236. La aritmética cierra. También ya usa consistentemente "10.8% RSNA validation subset" en varios lugares (reemplazando "test set"/"hold-out test set" anteriores).

**Falta:**
- Un **diagrama de flujo de participantes** (figura) — no existe todavía como imagen en el manuscrito, solo como prosa.
- Revisar el documento completo por si queda alguna instancia residual de "test set"/"hold-out test set" sin actualizar a la terminología consistente (no se puede confirmar al 100% sin búsqueda de texto completo sobre el `.docx`).

---

### Comentario 3 — Metodología de balanceo (distribución antes/después)

**Pide:** distribución de edad y sexo antes y después del filtro de balanceo, justificación del umbral de 50, y edades excluidas específicas.

**Tenemos:** Respuesta 3 en el borrador es solo una nota interna ("Agregar las gráficas de distribución de antes y después") — no implementado. El dato SÍ existe en el proyecto, ya calculado y verificado contra los CSV reales en una sesión anterior: raw = 12,611 imgs, 6,833♂/5,778♀ (54.2%/45.8%), 160 edades, rango 1–228m; balanceado = 11,783 imgs, 6,313♂/5,470♀ (53.6%/46.4%), 36 edades, rango 24–216m; 124 de 160 edades excluidas (−828 imágenes, 6.6% del total). Ver `docs/data/dataset_report.md`.

**Falta:** transcribir esa tabla/gráfica al manuscrito y redactar la justificación del umbral de 50 (criterio: mínimo una carga de batch por edad dado `BATCH_SIZE=32`). No requiere reentrenar ni recalcular nada — los números ya están verificados.

---

### Comentario 4 — Consistencia edad cronológica vs. TW3

**Pide:** que todo el manuscrito use consistentemente "TW3-assigned bone age" como variable objetivo, y que Figura 8 y todo análisis del dataset mexicano compare contra TW3, no contra edad cronológica.

**Tenemos:** Respuesta 4 afirma que ya se corrigió todo y se "eliminaron completamente las menciones a edad cronológica". **Esto no es cierto en el PDF actual** — Figura 8 (pág. 18) sigue diciendo *"scatter plots comparing chronological and predicted ages"*, eje *"Real Age (months)"*, título *"Dispersion Real Age vs Prediction"*. Ver también el Hallazgo Crítico #1 arriba: el bug de `08_mex_validation.py` puede significar que los datos subyacentes de Figura 8 y Tabla 9 se calcularon contra la variable equivocada.

**Falta:**
1. Re-ejecutar `08_mex_validation.py` (ya corregido) para regenerar los datos correctos.
2. Regenerar la Figura 8 con los títulos/ejes corregidos ("TW3 Bone Age" en vez de "Real Age"/"chronological").
3. Revisar el resto del manuscrito por más instancias de "chronological" (no se puede garantizar exhaustividad sin búsqueda de texto completo sobre el `.docx`).

---

### Comentario 5 — Estándar de referencia mexicano (confiabilidad inter-observador)

**Pide:** al menos dos lectores TW3 independientes, con reporte de calificaciones, cegamiento, adjudicación, acuerdo intra/inter-observador con IC 95%. Pide quitar el número de cédula profesional.

**Tenemos:** avance parcial — el manuscrito ya dice "assigned by two clinical experts using the TW3 method" (antes era un solo médico nombrado) y el número de cédula profesional **ya fue eliminado** del texto limpio. Respuesta 5 en el borrador confirma: "no formal interobserver or intraobserver reliability analysis was available for this revision (radiologo y médico)".

**Falta:** lo sustancial del comentario sigue sin resolverse — no hay análisis de concordancia (kappa, ICC), ni IC 95%, ni descripción de calificaciones/cegamiento/adjudicación. Esto requiere que los dos lectores existentes (radiólogo y médico, ya mencionados) califiquen independientemente al menos un subconjunto y se calcule el acuerdo — trabajo clínico real, no analítico.

---

### Comentario 6 — Representatividad de la cohorte mexicana

**Pide:** sustituir "Mexican population" por "single-center Mexican clinical cohort" en todo el documento; especificar fechas de inclusión, criterios de selección, indicaciones clínicas, etc.; no atribuir diferencias a etnia.

**Tenemos:** el título y varias secciones ya usan "single-center Mexican clinical cohort"/"external Mexican clinical cohort". Respuesta 6 reconoce la limitación real: *"No se cuenta con la información de todo el expediente clínico, solo tenemos la edad y sexo."*

**Falta:** dado que no hay más datos clínicos disponibles que edad y sexo, la respuesta honesta es declarar esa limitación explícitamente en el manuscrito (Sección 3.5/Limitaciones) en vez de dejarlo implícito, y hacer una pasada final por el texto completo buscando "Mexican population" residual (misma limitación de búsqueda que en Comentarios 2 y 4 — no hay acceso de texto completo al `.docx`).

---

### Comentario 7 — Análisis estadístico externo incompleto

**Pide:** IC 95% por bootstrap a nivel paciente, mediana AE, RMSE, sesgo, DE del error, proporciones dentro de 6/12 meses, pruebas pareadas entre modelos, estratificación por sexo y edad.

**Tenemos:** Respuesta 7 dice *"Falta anexar los datos, pero si se tienen"* — y es cierto: `docs/results/ablacion_backbones/2026-07-22_ablacion_backbones.md` ya tiene IC 95% bootstrap + Wilcoxon con corrección de Holm para las comparaciones entre los 4 backbones, tanto en RSNA (n=1,393) como en MEX (n=98). Esto **sobrevivió intacto** a la sesión anterior (no se tocó en el revert).

**Falta:**
1. Transcribir esas tablas al manuscrito (Tabla 8/9).
2. **Recalcular todo lo de MEX** una vez corregido el bug del Hallazgo Crítico #1 — los números actuales de `ablacion_backbones` para MEX pueden estar afectados por el mismo problema de `real_age` vs `bone_age`, hay que verificar contra qué versión del script se generaron.
3. Agregar lo que aún no existe: mediana AE, RMSE, sesgo, DE, proporciones ≤6m/≤12m, y estratificación por sexo/edad — esto es cómputo nuevo pero reutiliza los mismos `plot_data.json` una vez regenerados.

---

### Comentario 8 — Gráficos de dispersión insuficientes + colapso de F-VGG16

**Pide:** líneas de identidad, ecuación de regresión, bandas de confianza, Bland-Altman, pendiente/intercepto de calibración; investigar la causa del colapso de VGG16 (falla de optimización, fuga de datos, etc., no solo describirlo).

**Tenemos:** Respuesta 8 es solo una nota de intención: *"Si se va a modificar la fig. 8"*. Nada implementado todavía. El diagnóstico de VGG16 (estancamiento desde la primera época, no por falta de épocas; ausencia de BatchNorm/skip-connections al entrenar desde cero) se había armado en la sesión anterior con evidencia real del `training_history` — esa evidencia sigue disponible sin tocar en `experiments/26/training_history/` (no se modificó en el revert), solo hay que rehacer el análisis.

**Falta:** generar la Figura 8 mejorada (identidad, regresión, IC, Bland-Altman) **después** de corregir el Hallazgo Crítico #1, y redactar el diagnóstico de VGG16 con la evidencia del `training_history` real.

---

### Comentario 9 — Validación de segmentación (split y Dice por clase)

**Pide:** conteo exacto de imágenes de train/val/test de segmentación, confirmación de independencia de pacientes, Dice/IoU por clase (thumb, middle, pinky, wrist), no solo accuracy global.

**Tenemos:** Respuesta 9 tiene placeholders sin rellenar: *"the W manually annotated RSNA radiographs were divided into X training images, Y validation images, and Z testing images"*. Los números reales **ya se determinaron en la sesión anterior y siguen siendo válidos** (no dependen de nada revertido): el dataset anotado real tiene **379 imágenes** (no 200, como todavía dice el manuscrito en la Sección 2.3), confirmado contra el log de entrenamiento original del modelo activo (`models/hand-detector/hand-detector_00/`), con split reproducible 303 train / 76 val (`train_test_split(test_size=0.2, random_state=42)`). También ya se había calculado el Dice/IoU por clase recalculando sobre ese split.

**Falta:**
1. Corregir "200 radiographs" → "379 radiographs" en la Sección 2.3, con el split 303/76.
2. Rehacer el cálculo de Dice/IoU por clase (recalculable directamente desde `hand-detector_00`, sin reentrenar) y agregarlo a la Sección 3.1.
3. Aclarar el procedimiento de revisión de máscaras (un solo revisor clínico, sin segunda opinión — limitación a declarar explícitamente).

---

### Comentario 10 — Segmentación no probada en imágenes mexicanas

**Pide:** un subconjunto cegado de imágenes mexicanas segmentado manualmente y comparado contra las máscaras automáticas, con ejemplos de éxito/fallo y correlación entre calidad de segmentación y error final.

**Tenemos:** Respuesta 10 vacía. Nada hecho.

**Falta:** trabajo real de anotación manual (≈20–30 imágenes mexicanas en LabelMe) — no es analítico, requiere una persona anotando. Es el ítem más costoso en tiempo de todo el documento (día(s) de trabajo manual), igual que en la ronda anterior.

---

### Comentario 11 — Asociación TW3 exagerada

**Pide:** describir el diseño como "inspirado en" TW3, no como que lo implementa o automatiza, a menos que reconstruya el score oficial.

**Tenemos:** ya corregido en gran parte — el manuscrito cambió "employs the TW3 technique" por "inspired by the TW3 framework" (Sección 2, pág. 3–4) y "adopts an anatomical segmentation strategy inspired by the TW3 framework" (pág. 4). Respuesta 11 confirma: *"Corregí las definiciones para aclarar que es un método inspirado."*

**Falta:** solo limpieza editorial — hay un artefacto de fusión de texto sin corregir: *"inspired based by theon the Tanner–Whitehouse 3 (TW3) technique"* (pág. 3) — mezcla de dos versiones del texto que nunca se resolvió. Ver Hallazgo Crítico #3.

---

### Comentario 12 — Fuga de datos en el entrenamiento de fusión (stacked-model leakage)

**Pide:** aclarar si la fusión se entrenó con predicciones de segmentos ajustados sobre los mismos casos (fuga), usar predicciones out-of-fold, o un tuning set claramente separado; reportar reglas de early stopping, criterio de selección, número de corridas repetidas; una sola semilla no es suficiente para comparar arquitecturas inestables.

**Tenemos:** Respuesta 12 reconoce el problema sin resolverlo: *"Este experimento haría que tuviéramos que actualizar todos los demás, ya que esta modificación se hace en el script de entrenamiento."* — correcto: es un cambio en el pipeline compartido que obliga a reentrenar todo lo que lo usa.

**Contexto importante:** en la sesión anterior se implementó y se probó una corrección real (split de 3 vías `segment_train`/`fusion_train`/`holdout_val` + semilla configurable) y se llegó a lanzar el reentrenamiento de los 4 backbones más la ablación de fusión del manuscrito (10 experimentos en total) en el clúster. **Ese trabajo fue revertido explícitamente a petición tuya** ("vamos a volver a empezar") — el código, los experimentos archivados y todo quedaron restaurados a su estado original. La corrección en sí (el diseño del split de 3 vías) sigue siendo válida conceptualmente si se decide retomarla, pero no hay nada implementado ahora mismo.

**Falta:** decidir si se retoma la corrección (implica modificar `src/06_training.py` y reentrenar potencialmente 10 experimentos, varios días de cómputo GPU) o si se responde solo con una limitación textual reconociendo el riesgo como trabajo futuro — la misma decisión que ya se tomó y se deshizo la vez pasada.

---

### Comentario 13 — Función de pérdida sin definición matemática

**Pide:** definición matemática formal de `attention_weighted loss`, explicar por qué MAE train > MAE val, aclarar si dropout/augmentación estaban activos al calcular la métrica de entrenamiento, reportar métricas en modo inferencia, replicar con múltiples semillas.

**Tenemos:** Respuesta 13 está **vacía** en el borrador (ver Hallazgo Crítico #2 — el contenido que la responde está mal colocado en Respuesta 14). La ecuación real de `attention_loss` está en `src/utils/losses.py` (no se tocó en el revert):

```python
loss = mean(|y_true - y_pred| * |(y_pred/y_true) - 1|)
```

que se simplifica algebraicamente a `mean((y_pred - y_true)^2 / y_true)` — un MSE ponderado por el inverso de la edad real (penaliza más los errores en pacientes jóvenes). La explicación de MAE train > MAE val (dropout activo durante el cómputo de la métrica de entrenamiento + augmentación solo en train) también se había redactado antes con evidencia del código de `06_training.py`.

**Falta:**
1. Mover la respuesta correcta al Comentario 13 (está en el lugar de la 14 en el borrador).
2. Agregar la ecuación formal y su simplificación al manuscrito (Sección 2.5).
3. Lo de "múltiples semillas" queda ligado al Comentario 12 — si no se reentrena con semilla variable, no se puede responder esta parte con datos reales, solo declararlo como limitación.

---

### Comentario 14 — Comparación de arquitecturas no controlada

**Pide:** definir explícitamente LR, optimizador, pesos preentrenados, capas congeladas, paciencia de early stopping, weight decay, normalización por backbone; sustentar la afirmación de que DenseNet121 es el mejor balance con medición empírica de tiempo de entrenamiento, latencia de inferencia, memoria — no solo conteo de parámetros.

**Tenemos:** Respuesta 14 en el borrador **no responde esto** — responde al Comentario 13 por error (ver Hallazgo Crítico #2), así que este comentario está efectivamente sin respuesta. Los hiperparámetros SÍ están confirmados idénticos entre los 4 backbones (verificado antes contra los `config.py` reales: `WEIGHTS=None`, `NUM_LAYERS_UNFREEZE=10`, sin early stopping explícito con paciencia definida más allá de lo genérico, sin weight decay). Los tiempos de entrenamiento reales por backbone también están documentados en `docs/results/ablacion_backbones/2026-07-22_ablacion_backbones.md` (ResNet50 ~12h, VGG16 ~7h, DenseNet121 ~10h, InceptionV3 ~14h) — esto no se tocó en el revert.

**Falta:**
1. Redactar la respuesta real al Comentario 14.
2. Agregar al manuscrito la tabla de hiperparámetros por backbone (ya verificados) y los tiempos de entrenamiento reales (ya documentados).
3. Latencia de inferencia y memoria — no hay una medición limpia todavía (el intento anterior fue en CPU compartido, poco confiable); si se quiere un número presentable, hay que medirlo de forma aislada.

---

### Comentario 15 — Ablación insuficiente (falta baseline whole-hand)

**Pide:** comparar contra un modelo de mano completa sin segmentar, mismo split/protocolo/backbone; comparaciones adicionales sobre el efecto de segmentación, fusión de 4 regiones, variable de sexo, CLAHE, y regiones individuales.

**Tenemos:** Respuesta 15 vacía. En la sesión anterior se implementó un nuevo `MODEL_TYPE="whole_hand"` en el pipeline y se llegó a lanzar el entrenamiento (experimento 60) — **revertido** junto con todo lo demás a petición tuya. El diseño (mismo split que el experimento de fusión de referencia, mismo backbone DenseNet121, mismo protocolo de dos fases) sigue siendo válido conceptualmente.

**Falta:** igual que el Comentario 12 — decidir si se retoma (nuevo experimento + entrenamiento, ~10h de GPU) o si se responde con limitación textual. Las comparaciones adicionales (efecto de sexo, CLAHE, regiones individuales) son experimentos completamente nuevos que no se habían empezado ni antes.

---

### Comentario 16 — Mapas de saliencia sin reproducibilidad

**Pide:** especificar capa objetivo, normalización, semilla, criterio de interpretación; ampliar de 3 ejemplos a una muestra predefinida más grande, o retirar las afirmaciones de interpretabilidad.

**Tenemos:** Respuesta 16 vacía. En la sesión anterior se había documentado la metodología real (gradiente vainilla respecto a la entrada, sin capa objetivo específica ya que no es Grad-CAM, normalizado a [0,1] por imagen, umbral del percentil 97 para el overlay — todo esto se lee directamente del código en `src/07_validation.py`/`08_mex_validation.py`, que **no se tocó** en el revert) y se habían generado 12 ejemplos nuevos estratificados por edad — esos 12 ejemplos específicos sí se perdieron al borrar `review2/`, pero se pueden regenerar fácilmente porque el código fuente que los generó sigue intacto.

**Falta:**
1. Redactar la metodología (ya se puede extraer del código sin reentrenar nada).
2. Regenerar los ejemplos ampliados (requiere cargar el modelo de fusión ya entrenado y correr inferencia sobre ~12 imágenes — minutos, no hay que reentrenar).

---

### Comentario 17 — Tabla 10 no comparable

**Pide:** que la Tabla 10 no presente 13.70 meses como si compitiera con 4.2–6.2 meses de la literatura bajo protocolos distintos; reconocer que el resultado propio es sustancialmente inferior si no se usa el protocolo oficial idéntico.

**Tenemos:** esto **ya está resuelto en el manuscrito** — el texto limpio (no tachado) dice explícitamente: *"Table 10 should be interpreted only as contextual literature background... the proposed F-DenseNet121 result (13.70 months) is clearly less favorable than the best values previously reported in the literature and therefore should not be presented as directly competitive."* Coincide con lo que pide el revisor casi palabra por palabra.

**Falta:** solo escribir la Respuesta 17 en el borrador confirmando que ya se hizo (la respuesta está vacía aunque el trabajo en el manuscrito ya existe) — nada de análisis nuevo.

---

### Comentario 18 — Declaración de ética inaceptable

**Pide:** identificar el comité aprobador específico, nombre del comité, fecha de aprobación, título del estudio; declarar explícitamente si hubo aprobación y si se exentó el consentimiento informado; articular la relación entre el acuerdo de 2019 y este análisis secundario.

**Tenemos:** parcialmente — la Sección 2.1 (cuerpo del texto, no tachada) ya tiene el detalle bueno: *"registered with the Research Committee, Research Ethics Committee, and Biosafety Committee of the University of Guadalajara (UDG) under file number 26-110"* + agreement RG/ACC/75/2019. **Pero** la sección formal **"Institutional Review Board Statement"** al final del manuscrito (pág. 20) todavía tiene el texto genérico viejo: *"we did not have direct contact with patients... individual informed consent forms are not available"* — sin mencionar el comité, la fecha, ni el expediente 26-110 que sí aparecen en la Sección 2.1. Response 18 está vacía.

**Falta:** sincronizar la sección formal "Institutional Review Board Statement" con el detalle que ya existe en la Sección 2.1 — es una inconsistencia interna del propio manuscrito, fácil de corregir, no requiere información nueva.

---

### Comentario 19 — Limpieza editorial completa

**Pide:** eliminar cambios rastreados, palabras fusionadas ("disciplinemedicine", "areshow", "4four", etc.), DOIs duplicados; estandarizar abreviaturas/decimales; el abstract no debe afirmar mejora de generalización cuando el hallazgo central es un déficit de generalización.

**Tenemos:** Response 19 vacía. Confirmado en vivo (Hallazgo Crítico #3) que **todavía queda al menos un artefacto de fusión de texto** sin limpiar ("inspired based by theon the"). El abstract actual (pág. 1) ya es razonablemente cauteloso ("Rather than positioning these results as evidence of state-of-the-art accuracy, this study highlights that internal validation performance can substantially overestimate real-world reliability...") — no afirma mejora, así que esa parte específica ya está bien.

**Falta:** pasada completa de limpieza sobre el `.docx` real (no tenemos acceso de búsqueda de texto completo desde el PDF) — aceptar cambios, buscar los términos fusionados específicos que cita el revisor, revisar DOIs duplicados en referencias.

---

### Comentario 20 — Decisión general

**Pide:** resumen de que el manuscrito no debe aceptarse en su forma actual; requiere revisión exhaustiva de todos los puntos anteriores.

**Tenemos:** Response 20 vacía — es la respuesta de cierre, se redacta al final una vez resueltos (o explícitamente diferidos) los puntos 1–19.

**Falta:** nada de análisis nuevo — es la síntesis final.

---

## Resumen por tipo de esfuerzo

| Categoría | Comentarios | Notas |
|---|---|---|
| **Ya resuelto en el manuscrito, solo falta confirmarlo en la respuesta** | #11, #17 | Revisar que la limpieza final no borre estos cambios por accidente |
| **Dato ya existe en el proyecto, solo falta transcribir al manuscrito** | #3, #9 (split real), #13 (ecuación), #14 (hiperparámetros/tiempos) | Sin cómputo nuevo |
| **Requiere re-ejecutar scripts ya corregidos (sin reentrenar modelos)** | #4, #7, #8, #9 (Dice por clase), #16 | Depende de #08_mex_validation.py ya arreglado |
| **Inconsistencia interna del manuscrito, corrección directa** | #1, #2 (terminología residual), #18 | Editorial, no analítico |
| **Requiere trabajo clínico/manual real** | #5 (dos lectores, sin acuerdo calculado), #10 (segmentación mexicana) | Los más lentos de todos |
| **Requiere decisión sobre reentrenar (revertido la sesión pasada)** | #12, #15 | El diseño ya existe, la ejecución se deshizo a petición tuya |
| **Bloqueado sin el `.docx` fuente** | #1 (mecánica), #19 (mecánica) | Solo tenemos el PDF exportado |
| **Corrección de fondo recién encontrada, afecta a otros comentarios** | Hallazgo Crítico #1 (real_age vs bone_age) | Repercute en #4, #7, #8 |
| **Error de numeración en el borrador de respuestas** | Hallazgo Crítico #2 | Repercute en #13, #14 |
