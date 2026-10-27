# Doctorado: índice

> Documentación del proyecto doctoral organizada estilo Kiro (spec-driven): `requirements.md`
> define el qué y el por qué, `design.md` el cómo, `tasks.md` el checklist accionable. Lee los
> tres en ese orden si es la primera vez; si ya conoces el proyecto, ve directo a la sección
> "Dónde estamos ahora" más abajo y a `tasks.md`.

## Los tres documentos

| Archivo | Para qué sirve | Cuándo actualizarlo |
|---|---|---|
| [requirements.md](requirements.md) | Los 11 objetivos del cronograma oficial (A–K), cada uno como un requirement con historia de usuario y criterios de aceptación. Es la traducción funcional de `Protocolo.pdf`. | Solo si el protocolo oficial cambia (poco frecuente) |
| [design.md](design.md) | Arquitectura actual del sistema, arquitectura objetivo (fusión multimodal), decisiones de diseño ya tomadas y preguntas abiertas | Cada vez que se tome una decisión de diseño nueva o se resuelva una pregunta abierta |
| [tasks.md](tasks.md) | Checklist granular de tareas, cada una referenciando el requirement que satisface. Es el documento de trabajo del día a día | Cada vez que se empiece o cierre una tarea |
| [referencias.md](referencias.md) | Glosario de las referencias bibliográficas citadas en `Tesis.pdf` y `Protocolo.pdf`, con resumen y etiquetas por técnica para búsqueda | Solo si se comparten nuevos documentos fuente con citas |

## Contexto en una página

Este es un doctorado en Ciencias de la Electrónica y la Computación (CUCEI, UdeG) que
**continúa** el trabajo de una maestría ya concluida y publicada: *Propuesta de un Sistema
Automatizado para la Estimación de la Edad a partir de Radiografías de Mano* (nov. 2025),
publicada como artículo JCR en *Technologies* (MDPI). Ese sistema (pipeline de preprocesamiento
+ segmentación U-Net + 4 modelos CNN por segmento + fusión) alcanzó MAE = 5.7 meses (validación
interna) y 13.70 meses (validación externa RSNA), pero la propia tesis reconoció limitaciones:
preprocesamiento sin explorar alternativas, fallos de segmentación sin analizar, y **pérdida de
información al fusionar las 4 salidas de segmento de forma simple** (concatenación + sexo).

El doctorado ataca esas tres limitaciones, con el eje central siendo la hipótesis de que un
**modelo multimodal** (imagen + variables clínicas no identificativas, con una fusión
profunda en vez de concatenación simple) mejora significativamente precisión, robustez y
consistencia. Ver [requirements.md](requirements.md) para el detalle completo objetivo por
objetivo y [design.md](design.md) para cómo se piensa lograr.

## Dónde estamos ahora (actualizar en cada sesión relevante)

**Última actualización**: 2026-10-25.

- Objetivo activo: **C** (Análisis de Procedimiento Existente), transicionando a **D**
  (Corrección y mejora del flujo de procesamiento). El protocolo (**B**) sigue en trámite de
  registro, sin bloquear el trabajo técnico.
- El sub-hilo de preprocesamiento de D (D1 en `design.md`) está en buen estado: bugs
  corregidos, parámetros centralizados en `config/preprocessing.py`, dataset completo
  regenerado.
- El sub-hilo de fusión multimodal de D (D2, el corazón de la hipótesis doctoral) **no ha
  empezado**.
- **Siguiente tarea concreta**: ver `tasks.md`, tarea 3.4 (rehacer la grilla de segmentación).

## Documentos fuente

- `Protocolo.pdf`: propuesta de proyecto de investigación doctoral, entregada por el usuario el
  2026-10-25. Recibida como texto en la conversación, no versionada como archivo en el repo.
  `requirements.md` es su transcripción funcional; si se necesita el documento original completo
  (diagramas, formato exacto, cronograma con fechas precisas), pedirlo de nuevo al usuario.
- `Tesis.pdf`: tesis de maestría ya mencionada arriba. Mismo caso: recibida como texto, no
  versionada como archivo.

## Relación con el resto de `docs/`

Este índice es el punto de entrada de más alto nivel del proyecto (enlazado desde
[`docs/README.md`](../../README.md)). El detalle técnico día a día de cada objetivo sigue
viviendo donde ya vivía:

- [`fase_segmentacion.md`](fase_segmentacion.md): plan y bitácora técnica de C/D1
  (preprocesamiento y segmentación).
- [`docs/results/doctorado/segmentacion_diagnostico/`](../../results/doctorado/segmentacion_diagnostico/) y
  [`docs/results/doctorado/preprocesamiento_diagnostico/`](../../results/doctorado/preprocesamiento_diagnostico/):
  los hallazgos y la evidencia cruda detrás de C/D1.
