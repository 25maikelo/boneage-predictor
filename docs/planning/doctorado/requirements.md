# Requirements: Doctorado

> Transcripción funcional de `Protocolo.pdf` (propuesta de proyecto de investigación doctoral,
> CUCEI-UdeG, entregada por el usuario el 2026-10-25). Cada una de las 11 actividades del
> cronograma oficial (columna "Actividad" del cronograma 2026-2029) se traduce aquí como un
> requirement con una historia de usuario y criterios de aceptación en formato EARS
> (WHEN/IF ... SHALL ...). El periodo oficial de cada requirement viene del cronograma del
> protocolo; si hay duda sobre una fecha exacta, verificar contra el PDF original.

## Contexto: por qué existen estos requirements

**Hipótesis del proyecto**: la integración de un modelo multimodal que combine información
radiográfica con variables clínicas no identificativas del paciente permitirá capturar de
manera más completa los patrones del desarrollo óseo, incrementando significativamente
precisión, robustez y consistencia frente a los enfoques unimodales actuales.

**Objetivo general**: fortalecer el sistema automatizado de estimación de la edad ósea mediante
un enfoque multimodal que integre información radiográfica y variables clínicas no
identificativas, junto con la corrección del preprocesamiento, la detección/corrección de
fallos de segmentación y la integración profunda de los modelos de predicción.

Estos dos puntos motivan todos los requirements siguientes: no son tareas aisladas, son pasos
hacia validar o refutar la hipótesis con evidencia.

---

## Requirement 1: Revisión de la literatura (A)

**Periodo oficial**: 2026 ene-jun → 2029 ene-jun (actividad continua, no un hito puntual).

**Historia de usuario**: Como investigador doctoral, quiero mantener una revisión de literatura
vigente y específica a cada subtema técnico del proyecto, para fundamentar las decisiones
metodológicas y justificar la novedad del trabajo frente al estado del arte.

**Criterios de aceptación**:
1. WHEN se aborde una técnica nueva no cubierta en la literatura de la maestría (p. ej. fusión
   multimodal, pruebas estadísticas no paramétricas aplicadas a BAA) THEN el proyecto SHALL
   contar con una revisión de literatura específica de esa técnica antes de implementarla.
2. CADA afirmación técnica relevante en los documentos de diseño SHALL estar respaldada por una
   fuente citable (DOI o referencia completa), no por memoria o suposición.
3. WHEN se redacte el artículo JCR doctoral (Requirement 9) THEN la revisión de literatura SHALL
   no tener huecos pendientes abiertos en `tasks.md` para los temas que el artículo cubre.

---

## Requirement 2: Registro de Protocolo de Investigación (B)

**Periodo oficial**: 2026 ene-jun.

**Historia de usuario**: Como aspirante a doctorado, quiero que el protocolo de investigación
esté formalmente registrado y aprobado por la institución, para poder proceder con las etapas
subsecuentes del programa con respaldo institucional.

**Criterios de aceptación**:
1. EL protocolo SHALL ser aprobado formalmente por el comité/institución correspondiente.
2. IF el comité solicita ajustes al protocolo THEN el cambio y su justificación SHALL quedar
   documentados (en este repositorio o en el expediente institucional, según corresponda).

---

## Requirement 3: Análisis de Procedimiento Existente (C)

**Periodo oficial**: 2026 jul-dic.

**Historia de usuario**: Como investigador, quiero un diagnóstico exhaustivo y reproducible del
sistema de la maestría (preprocesamiento, segmentación, estrategia de fusión), para identificar
con evidencia empírica qué corregir en los Requirements 4 y 5, en vez de asumir dónde están los
problemas.

**Criterios de aceptación**:
1. WHEN se audite una etapa del pipeline (preprocesamiento o segmentación) THEN el diagnóstico
   SHALL basarse en métricas propias de la imagen o del modelo (fill ratio, Dice/IoU, contraste,
   etc.), no solo en inspección visual.
2. EL diagnóstico SHALL cuantificar la tasa de fallo observada y su causa raíz para cada defecto
   identificado, no solo reportar que "existe un problema".
3. EL diagnóstico SHALL ser reproducible: script versionado, datos de entrada identificados,
   semilla fija cuando aplique muestreo aleatorio.
4. EL diagnóstico de la estrategia de fusión actual (concatenación + sexo) SHALL cuantificar la
   pérdida de información señalada en la tesis de maestría con una metodología propia, no solo
   citar la limitación ya reconocida ahí.

---

## Requirement 4: Corrección y mejora del flujo de procesamiento (D)

**Periodo oficial**: 2027 ene-jun → 2028 ene-jun.

**Historia de usuario**: Como investigador, quiero corregir los defectos identificados en el
Requirement 3 y rediseñar la integración de los modelos de predicción, para eliminar la pérdida
de información que limitaba el sistema de la maestría y mejorar la coherencia con los criterios
del método TW3.

**Criterios de aceptación**:
1. WHEN se corrija un defecto del pipeline THEN SHALL verificarse con al menos un caso de prueba
   conocido que el comportamiento post-corrección es el esperado, antes de aplicarlo a todo el
   dataset.
2. EL pipeline corregido SHALL regenerarse sobre el dataset completo (no solo una muestra) antes
   de usarse como base para el Requirement 5.
3. LOS parámetros de preprocesamiento y segmentación SHALL quedar centralizados en archivos de
   configuración versionados, no hardcodeados dentro de las funciones, para permitir ablaciones
   sistemáticas.
4. WHEN se diseñe el esquema de fusión multimodal (ver también Requirement 4, objetivos OP4/OP5
   del protocolo) THEN SHALL compararse al menos dos de las tres estrategias de integración
   candidatas (temprana, intermedia, tardía) con evidencia cuantitativa antes de elegir una.
5. LA nueva estrategia de fusión SHALL incorporar variables clínicas no identificativas además
   de la imagen, no solo las salidas de los 4 modelos de segmento.

---

## Requirement 5: Evaluación del modelo con nuevos criterios (E)

**Periodo oficial**: 2028 ene-jun.

**Historia de usuario**: Como investigador, quiero evaluar el sistema corregido con pruebas
estadísticas rigurosas (paramétricas y no paramétricas), para determinar si las mejoras
observadas son estadísticamente significativas y no producto del azar o de la selección de
métricas favorables.

**Criterios de aceptación**:
1. LA evaluación SHALL incluir al menos una prueba estadística no paramétrica comparando el
   sistema corregido contra el mejor baseline de la maestría (ver `design.md` para cuál).
2. LA comparación SHALL usar el mismo conjunto de validación y la misma semilla para el modelo
   nuevo y el baseline.
3. IF una mejora observada no resulta estadísticamente significativa THEN SHALL reportarse así
   explícitamente en la documentación de resultados, no omitirse.
4. LA evaluación SHALL generar un reporte por imagen que documente el comportamiento del modelo
   en los mejores y peores escenarios, y por grupo de edad (objetivo OP7 del protocolo).

---

## Requirement 6: Aplicación y validación en población mexicana (F)

**Periodo oficial**: 2028 jul-dic.

**Historia de usuario**: Como investigador, quiero validar el sistema sobre una base de datos
mexicana ampliada (con variables clínicas) y comparar contra evaluación de especialistas, para
medir la generalización real del sistema a la población objetivo del proyecto, más allá de la
validación de 100 imágenes IMSS ya hecha en la maestría.

**Criterios de aceptación**:
1. LA base de datos mexicana SHALL incluir radiografías de mano y variables clínicas no
   identificativas de varias instituciones de salud, con anonimización y aprobación ética
   documentadas (objetivo OP1 del protocolo).
2. LA validación SHALL desagregar el desempeño por rango de edad y por sexo.
3. LAS predicciones del sistema SHALL compararse contra evaluaciones clínicas realizadas por
   especialistas, no únicamente contra la etiqueta `bone_age` de un dataset.
4. EL sistema SHALL evaluarse también en la etapa de segmentación (no solo en la edad final
   predicha) sobre un subconjunto mexicano anotado, retomando explícitamente el Comentario #10
   de la segunda ronda de revisión del artículo de maestría ("segmentación no probada en
   imágenes mexicanas"), que quedó cerrado sin resolver.

---

## Requirement 7: Comparación entre poblaciones y análisis final (G)

**Periodo oficial**: 2028 jul-dic.

**Historia de usuario**: Como investigador, quiero comparar el desempeño del sistema entre la
base RSNA y la población mexicana, para identificar diferencias de comportamiento, sesgos
potenciales y limitaciones de generalización antes de dar por cerrado el desarrollo del modelo.

**Criterios de aceptación**:
1. LA comparación SHALL reportar diferencias de desempeño (MAE para edad final, Dice/IoU para
   segmentación) entre poblaciones con una medida de incertidumbre (intervalo de confianza o
   prueba de significancia), no solo el valor puntual.
2. EL análisis SHALL identificar explícitamente qué subgrupos (por edad, por sexo) muestran
   mayor degradación de desempeño al pasar de RSNA a población mexicana.
3. EL análisis SHALL concluir explícitamente sobre la confiabilidad y aplicabilidad clínica del
   sistema, no solo presentar las cifras comparativas.

---

## Requirement 8: Desarrollo del prototipo final (H)

**Periodo oficial**: 2028 jul-dic.

**Historia de usuario**: Como investigador, quiero un prototipo funcional que integre todo el
flujo (preprocesamiento, segmentación, predicción multimodal, reporte), para demostrar la
aplicabilidad clínica del sistema más allá de scripts de experimentación.

**Criterios de aceptación**:
1. EL prototipo SHALL aceptar una radiografía de mano como entrada y producir una estimación de
   edad ósea.
2. EL prototipo SHALL generar el reporte por imagen definido en el Requirement 5 (mejores/peores
   escenarios, interpretación por grupo de edad).
3. EL prototipo SHALL estar documentado lo suficiente para que alguien fuera del proyecto pueda
   ejecutarlo sin depender de contexto no escrito.
4. EL formato del prototipo (CLI, servicio web, notebook interactivo, etc.) SHALL decidirse
   explícitamente y quedar registrado en `design.md`; no hay un formato asumido por defecto.

---

## Requirement 9: Escritura de artículo JCR doctoral (I)

**Periodo oficial**: 2028 jul-dic.

**Historia de usuario**: Como investigador, quiero publicar los resultados del sistema
multimodal doctoral en una revista indexada en JCR, para que el trabajo sea validado por pares y
visible a la comunidad científica.

**Criterios de aceptación**:
1. EL artículo SHALL reportar resultados posteriores a la finalización de los Requirements 4 a 7
   (sistema corregido, evaluado y validado en ambas poblaciones).
2. EL artículo SHALL ser un trabajo distinto del artículo ya publicado en *Technologies* (MDPI),
   que corresponde a la maestría, no al doctorado.

---

## Requirement 10: Escritura de la Tesis doctoral (J)

**Periodo oficial**: 2029 ene-jun → jul-dic.

**Historia de usuario**: Como aspirante a doctorado, quiero documentar el trabajo completo en el
documento de tesis, para cumplir el requisito de titulación del programa y dejar un registro
exhaustivo del proceso.

**Criterios de aceptación**:
1. LA tesis SHALL cubrir los Requirements 1 a 9 con el mismo nivel de rigor y verificación
   empírica usado en los documentos técnicos de `docs/results/` a lo largo del proyecto.
2. LA tesis SHALL estar aprobada por el comité antes de programar la defensa (Requirement 11).

---

## Requirement 11: Defensa de tesis doctoral (K)

**Periodo oficial**: 2029 jul-dic.

**Historia de usuario**: Como aspirante a doctorado, quiero defender el trabajo ante el comité,
para obtener el grado de Doctor en Ciencias de la Electrónica y la Computación.

**Criterios de aceptación**:
1. LA defensa SHALL programarse solo después de que el comité haya aprobado la versión final de
   la tesis.
2. No debe confundirse con la defensa de la **maestría**, ya ocurrida (la pregunta sobre los
   parámetros de CLAHE/Otsu que motivó centralizarlos en `config/preprocessing.py` vino de esa
   defensa de maestría, no de una defensa doctoral, que todavía no ha ocurrido).
