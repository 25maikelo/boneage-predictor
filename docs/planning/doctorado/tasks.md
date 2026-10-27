# Tasks: Doctorado

> Checklist accionable. Cada tarea referencia el requirement de
> [requirements.md](requirements.md) que satisface. Marcar `[x]` al cerrar una tarea y añadir
> una nota breve de evidencia (commit, documento, resultado) en vez de solo tachar. No reordenar
> ni borrar tareas cerradas; si una tarea deja de ser necesaria, tachar el texto y anotar por
> qué, no eliminarla (mismo criterio que la bitácora de `fase_segmentacion.md`).

## 1. Revisión de la literatura (Requirement 1)

- [x] 1.1 Revisión general para el protocolo doctoral (bibliografía de `Protocolo.pdf`)
- [x] 1.2 Revisión de literatura de preprocesamiento: Otsu, CLAHE, BoneXpert
      (`docs/results/doctorado/preprocesamiento_diagnostico/revision_literatura.md`)
- [ ] 1.3 Revisión de literatura de fusión multimodal (imagen + variables clínicas tabulares)
- [ ] 1.4 Revisión de literatura de pruebas estadísticas no paramétricas aplicadas a BAA
- [ ] 1.5 Revisión de literatura de segmentación de mano en poblaciones no occidentales/no-RSNA

## 2. Registro de Protocolo de Investigación (Requirement 2)

- [ ] 2.1 Protocolo aprobado formalmente ante el comité/institución (trámite administrativo, no
      requiere trabajo de código en este repositorio)

## 3. Análisis de Procedimiento Existente (Requirement 3)

- [x] 3.1 Diagnóstico de fallos de segmentación de `hand-detector_00` con matriz de confusión
      por píxel y peores casos por clase
      (`docs/results/doctorado/segmentacion_diagnostico/diagnostico.md`)
- [x] 3.2 Diagnóstico de preprocesamiento con métricas propias (fill ratio, contraste,
      saturación), sin depender del segmentador
      (`docs/results/doctorado/preprocesamiento_diagnostico/diagnostico.md`)
- [x] 3.3 Revisión de `04_segment_images.py` con el mismo nivel de rigor que `02`/`03` (sin bugs
      equivalentes encontrados)
- [ ] 3.4 Cuantificar con metodología propia la pérdida de información de la estrategia de
      fusión actual (concatenación + sexo), más allá de lo ya reconocido en la tesis de maestría

## 4. Corrección y mejora del flujo de procesamiento (Requirement 4)

### 4.1 Preprocesamiento y segmentación (sub-hilo D1 de `design.md`)

- [x] 4.1.1 Bug de desincronización imagen/máscara en augmentation de
      `01_train_hand_detector.py` (dice 0.1867 → 0.9073)
- [x] 4.1.2 Bug de `ENCODER_WEIGHTS=None` que colapsaba el entrenamiento
- [x] 4.1.3 Bug de corrimiento de píxeles / canvas recortado en la rotación de
      `02_frame_and_zoom.py` (descartes 5.6% → 2.1%)
- [x] 4.1.4 Bug de rotación espuria de 90° en manos ya alineadas (`02_frame_and_zoom.py`)
- [x] 4.1.5 Bug de `convertScaleAbs` reflejando tonos oscuros (`03_histogram_equalization.py`)
- [x] 4.1.6 Parámetros de preprocesamiento centralizados en `config/preprocessing.py`
- [x] 4.1.7 Dataset completo (12,811 imágenes) regenerado en disco con los fixes aplicados
- [ ] 4.1.8 **Siguiente tarea concreta del proyecto**: rehacer la grilla de segmentación con el
      código corregido, agregando `spatial`/`cropped` como eje adicional. Evaluar por Dice/IoU
      por clase + MAE downstream + Dice en subconjunto mexicano anotado (Etapa 2 de
      `fase_segmentacion.md`)
- [ ] 4.1.9 Investigar la causa de fondo del ~1.7% de imágenes que siguen fallando el recorte
      (bajo contraste global, hallazgo sin fix adoptado; ver
      `preprocesamiento_diagnostico/diagnostico.md`)

### 4.2 Fusión multimodal (sub-hilo D2 de `design.md`, no iniciado)

- [ ] 4.2.1 Resolver las preguntas abiertas de `design.md` sección 2 (variables clínicas
      disponibles, entrenamiento end-to-end vs. por etapas, tratamiento del género)
- [ ] 4.2.2 Implementar y comparar al menos 2 de las 3 estrategias de fusión (temprana,
      intermedia, tardía) con evidencia cuantitativa
- [ ] 4.2.3 Elegir la estrategia ganadora con justificación documentada

## 5. Evaluación del modelo con nuevos criterios (Requirement 5)

- [ ] 5.1 Definir el protocolo de comparación formal: qué prueba estadística, baseline exacto
      (candidato: Exp 57, ver `docs/results/maestria/2026-07-22_experiments_master.md`), mismo split/seed
- [ ] 5.2 Ejecutar la evaluación no paramétrica del sistema corregido vs. baseline
- [ ] 5.3 Generar el reporte por imagen (mejores/peores escenarios, por grupo de edad)

## 6. Aplicación y validación en población mexicana (Requirement 6)

- [ ] 6.1 Gestionar la obtención de la base de datos mexicana ampliada (múltiples instituciones,
      variables clínicas, anonimización); probablemente requiere trámite de ética propio, a
      confirmar
- [ ] 6.2 Evaluar el sistema sobre la nueva base, desagregado por edad y sexo
- [ ] 6.3 Comparar predicciones contra evaluación clínica de especialistas
- [ ] 6.4 Evaluar segmentación (no solo edad final) en subconjunto mexicano anotado, retomando
      el Comentario #10 de la revisión del artículo de maestría

## 7. Comparación entre poblaciones y análisis final (Requirement 7)

- [ ] 7.1 Comparar MAE/Dice entre RSNA y población mexicana con medida de incertidumbre
- [ ] 7.2 Identificar subgrupos (edad, sexo) con mayor degradación entre poblaciones
- [ ] 7.3 Análisis integral de confiabilidad y aplicabilidad clínica

## 8. Desarrollo del prototipo final (Requirement 8)

- [ ] 8.1 Decidir el formato del prototipo (CLI, servicio web, notebook interactivo)
- [ ] 8.2 Implementar el flujo completo: entrada → preprocesamiento → segmentación → predicción
      multimodal → reporte por imagen
- [ ] 8.3 Documentar el prototipo para uso fuera del proyecto

## 9. Escritura de artículo JCR doctoral (Requirement 9)

- [ ] 9.1 Redactar artículo sobre los resultados del sistema multimodal doctoral (distinto del
      artículo de *Technologies* ya publicado, que es de la maestría)

## 10. Escritura de la Tesis doctoral (Requirement 10)

- [ ] 10.1 Esqueleto de capítulos
- [ ] 10.2 Redacción
- [ ] 10.3 Aprobación del comité

## 11. Defensa de tesis doctoral (Requirement 11)

- [ ] 11.1 Programar fecha
- [ ] 11.2 Defender
