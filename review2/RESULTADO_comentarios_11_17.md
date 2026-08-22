# Comentarios 11 y 17 — Ya resueltos en el manuscrito, solo falta la respuesta

Ambos comentarios ya están atendidos en el cuerpo del manuscrito (texto limpio, no tachado). No requieren análisis ni cómputo nuevo — solo redactar la respuesta formal al revisor confirmándolo, y en el caso de #11, señalar un artefacto de edición pendiente que se resuelve junto con el Comentario 19.

---

## Comentario 11 — Asociación TW3 exagerada

**Pide:** describir el diseño como "inspirado en" TW3, no como que lo implementa o automatiza, a menos que se reconstruya el score oficial.

**Ya en el manuscrito:**
- Sección 2 (pág. 3–4): *"...a fusion convolutional architecture for bone age estimation, inspired [based on] the Tanner–Whitehouse 3 (TW3) technique"* (ver nota de artefacto abajo).
- Pág. 4: *"...adopts an anatomical segmentation strategy inspired by the TW3 framework"*.

Ambas instancias ya usan "inspired by/based on", no "implements" ni "automates" — el cambio de fondo que pide el revisor ya está hecho.

**📝 Respuesta al revisor (Respuesta 11):**
> "We revised the manuscript's terminology throughout Section 2 to consistently describe the proposed architecture as *inspired by* the Tanner–Whitehouse 3 (TW3) framework, rather than as an implementation or automation of the official TW3 scoring system. The model does not reconstruct or reproduce the official TW3 score; it borrows only the anatomical segmentation strategy (four hand regions) that TW3 uses as a conceptual basis for region-wise skeletal maturity assessment."

**Pendiente (no es parte de este comentario, es del #19):** queda un artefacto de fusión de texto sin limpiar en la pág. 3 — *"inspired based by theon the Tanner–Whitehouse 3 (TW3) technique"* — mezcla de dos versiones de la misma frase que nunca se resolvió al aceptar cambios en Word. Se corrige junto con la limpieza editorial completa del Comentario 19 (bloqueado sin el `.docx` fuente).

---

## Comentario 17 — Tabla 10 no comparable

**Pide:** que la Tabla 10 no presente 13.70 meses como si compitiera con los 4.2–6.2 meses reportados en la literatura bajo protocolos oficiales distintos; reconocer explícitamente que el resultado propio es inferior si no se usa el protocolo idéntico.

**Ya en el manuscrito** (texto limpio, no tachado, inmediatamente después de la Tabla 10):
> *"Table 10 should be interpreted only as contextual literature background... the proposed F-DenseNet121 result (13.70 months) is clearly less favorable than the best values previously reported in the literature and therefore should not be presented as directly competitive."*

Esto coincide casi palabra por palabra con lo que pide el revisor — ya reconoce explícitamente que 13.70 meses no es competitivo frente al 4.2–6.2 de los protocolos oficiales, y ya reencuadra la tabla como contexto, no como comparación directa.

**📝 Respuesta al revisor (Respuesta 17):**
> "We agree. We revised the paragraph following Table 10 to explicitly state that the table should be interpreted only as contextual literature background, not as a direct performance comparison, since the cited studies use the official RSNA test protocol while our result (13.70 months) is obtained under a different, non-identical evaluation protocol. We now explicitly acknowledge that our result is clearly less favorable than the best values reported in the literature and should not be presented as competitive with them."

---

## Nota de proceso

Ambas respuestas quedan listas para copiar directamente al documento de respuestas al revisor. No modifican el manuscrito (ya estaba correcto); solo cierran el ciclo de "qué se cambió → por qué está resuelto" que pide el formato de respuesta punto por punto.
