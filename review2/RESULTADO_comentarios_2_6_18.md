# Comentarios 2, 6, 18 — Diagrama de flujo, representatividad, ética

---

## Comentario 2 — Diagrama de flujo de participantes

El texto de la Sección 2.1 ya reconcilia correctamente los 14,236 registros (verificado ya en `ANALISIS_COMENTARIOS.md`), pero solo en prosa. El revisor pide explícitamente un diagrama.

**Imagen:** [`figures/participant_flow_diagram.png`](figures/participant_flow_diagram.png). Script: [`scripts/flow_diagram.py`](scripts/flow_diagram.py).

**📝 Agrega esto** en la Sección 2.1, justo después del párrafo que reconcilia los 14,236 registros ("...The official 200-image RSNA test set was not used in this work..."):
> "Figure X summarizes the participant flow across all dataset partitions, from the original 14,236 RSNA radiographs to the final training, internal validation, and external evaluation sets used in this study."

---

## Comentario 6 — Representatividad de la cohorte mexicana

Ya resuelto en gran parte (título y varias secciones ya dicen "single-center Mexican clinical cohort"). Lo único pendiente es declarar explícitamente la limitación de datos clínicos, en vez de dejarla implícita.

**📝 Agrega esto** en la Sección 3.5 (Limitations):
> "Beyond age and sex, no additional clinical data were available for the external Mexican cohort (e.g., inclusion dates, clinical indications, sampling method, or health/endocrine status), precluding a fuller characterization of this single-center convenience sample. Consequently, observed performance differences between the RSNA and Mexican cohorts cannot be attributed to any single demographic or clinical factor, and should not be interpreted as evidence of ethnicity-driven differences, since acquisition equipment, protocol, referral patterns, and reference labeling also differ concurrently between the two datasets."

---

## Comentario 18 — Sincronizar la declaración de ética

La Sección 2.1 (cuerpo del texto) ya tiene el detalle completo: comité UDG (Research Committee, Research Ethics Committee, Biosafety Committee), expediente 26-110, acuerdo RG/ACC/75/2019. Pero la sección formal **"Institutional Review Board Statement"** al final del manuscrito (pág. 20) sigue con el texto genérico viejo, sin ese detalle — inconsistencia interna, no falta de información.

**📝 Reemplaza esto** (sección "Institutional Review Board Statement", al final del manuscrito):
> "With respect to ethical considerations, due to the nature and design of our research project, we did not have direct contact with patients, and all data used in the study were previously collected and fully anonymized in accordance with institutional and national regulations. Therefore, individual informed consent forms are not available."

**Por esto:**
> "This study was conducted under institutional agreement RG/ACC/75/2019 and was registered with the Research Committee, Research Ethics Committee, and Biosafety Committee of the University of Guadalajara (UDG) under file number 26-110. Data were collected retrospectively, and the authors had no direct contact with patients; all data used in the study were previously collected and fully anonymized in accordance with institutional and national regulations. Given the retrospective, fully anonymized nature of the data collection, the approving committee waived the requirement for individual informed consent; consequently, individual informed consent forms are not available. This secondary analysis of the Mexican clinical cohort falls within the scope of the 2019 agreement, which covers the retrospective use of anonymized pediatric hand radiographs for bone age assessment research at the study site."

*(Nota: esto asume que el acuerdo RG/ACC/75/2019 efectivamente cubre este análisis secundario — si no es así, hay que ajustar la última frase con la relación real entre el acuerdo y este estudio, que es justo lo que pide el comentario aclarar.)*
