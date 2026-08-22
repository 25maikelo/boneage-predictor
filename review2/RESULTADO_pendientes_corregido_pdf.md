# Correcciones puntuales pendientes en `corregido.pdf`

---

### 1. Abstract (pág. 1-2) (Comentarios 7 y 8, Hallazgo Crítico #1)

**Cambia esto:**
> "with F-InceptionV3 achieving the most consistent external performance among the evaluated backbones"

**Por esto:**
> "with F-DenseNet121 and F-InceptionV3 achieving statistically indistinguishable external performance among the evaluated backbones"

---

### 2. Conclusiones (pág. 27): párrafo del MAE externo (Comentarios 7, 8, 12, 13, 15)

**Cambia esto:**
> "Although F-DenseNet121 achieved the lowest internal validation MAE (5.77 months), it performed worse on the independent 10.8% RSNA validation subset, with a higher MAE (13.70 months). This discrepancy underscores the need to report independent validation subset performance, as validation metrics monitored during training may overestimate real-world generalization capability. When the models were evaluated on Mexican patient data, MAE increased across all architectures, indicating limited transferability beyond the RSNA development setting. In this external single-center cohort, F-InceptionV3 produced the numerically lowest external MAE (15.98 months), but no formal statistical superiority can be claimed from the current analysis."

**Por esto:**
> "Although F-DenseNet121 achieved the lowest internal validation MAE (5.77 months), it performed worse on the independent 10.8% RSNA validation subset, with a higher MAE (13.70 months). This discrepancy underscores the need to report independent validation-subset performance, as validation metrics monitored during training may overestimate real-world generalization capability. When the models were evaluated on Mexican patient data, MAE increased across all architectures, indicating limited transferability beyond the RSNA development setting. In this external single-center cohort, F-DenseNet121 produced the numerically lowest MAE (16.38 months), closely followed by F-InceptionV3 (17.28 months); paired statistical testing found no significant difference between these two architectures on either the RSNA or Mexican validation sets, so backbone selection in this study is better justified by training-time efficiency (Section 3.2) than by external MAE alone. F-VGG16 failed to generalize under every configuration tested, a collapse attributable to the absence of batch normalization or residual/dense connections when trained from scratch rather than to insufficient training or data leakage. A dedicated whole-hand baseline, trained under an identical protocol but without anatomical segmentation, showed that the four-region fusion strategy provides only a modest advantage on internal validation but a statistically significant and substantially larger one under external evaluation, most notably on the Mexican cohort, reinforcing that internal validation alone is an unreliable predictor of a design choice's real-world value."

**Agrega esto** como párrafo nuevo, justo después del párrafo anterior (todavía dentro de "4. Conclusions", antes de "These findings demonstrate that..."):
> "This study has methodological limitations that should guide its interpretation. The fusion stage was trained on the same data split used to fit the upstream segment models rather than on out-of-fold predictions or a disjoint tuning set, which may make internal validation MAE an optimistic estimate; each configuration was trained once, without repeated random seeds, so comparisons between closely performing architectures should be read as point estimates. The segmentation model's internal validation split was also used for its own checkpoint selection and is therefore not a blind test set. Saliency maps (Figure 7) are provided solely as a qualitative sanity check that model attention falls within the segmented anatomical regions, not as evidence of clinically meaningful interpretability."

---

### 3. Institutional Review Board Statement (pág. 27) (Comentario 18)

**Cambia esto:**
> "With respect to ethical considerations, due to the nature and design of our research project, we did not have direct contact with patients, and all data used in the study were previously collected and fully anonymized in accordance with institutional and national regulations. Therefore, individual informed consent forms are not available."

**Por esto:**
> "This study was conducted under institutional agreement RG/ACC/75/2019 and was registered with the Research Committee, Research Ethics Committee, and Biosafety Committee of the University of Guadalajara (UDG) under file number 26-110. Data were collected retrospectively, and the authors had no direct contact with patients; all data used in the study were previously collected and fully anonymized in accordance with institutional and national regulations. Given the retrospective, fully anonymized nature of the data collection, the approving committee waived the requirement for individual informed consent; consequently, individual informed consent forms are not available. This secondary analysis of the Mexican clinical cohort falls within the scope of the 2019 agreement, which covers the retrospective use of anonymized pediatric hand radiographs for bone age assessment research at the study site."

---

### 4. Página 3: artefacto de fusión de texto (Comentarios 11 y 19)

**Cambia esto:**
> "inspired based by theon the Tanner–Whitehouse 3 (TW3) technique"

**Por esto:**
> "inspired by the Tanner–Whitehouse 3 (TW3) technique"

---

### 4b. Página 5: falta el Participant Flow Diagram (Comentario 2)

**Agrega esto** (imagen) justo después del párrafo que termina en "...to avoid confusion between the RSNA-provided partitions and the internal development splits employed in this study." (el párrafo que reconcilia los 14,236 registros), y antes de la tabla de distribución antes/después del balanceo (futura Table 1):
- Imagen: `figures/participant_flow_diagram.png`
- Pie de figura: "**Figure 2.** Participant flow diagram summarizing the partitioning of the RSNA dataset (official training, validation, and test sets; age-frequency balancing into a balanced subset; internal training/validation split) and the parallel Mexican clinical dataset branch, each converging into its respective independent/external validation."

**Agrega esto** como párrafo nuevo, justo después del párrafo de reconciliación y antes de la imagen (los nombres coinciden exactamente con los términos que ya usa el artículo: "official training/validation/test set", "balanced subset", "internal training/validation subset" del mismo párrafo de reconciliación; "Mexican clinical dataset" del párrafo siguiente; "independent RSNA validation subset" y "external validation" del Abstract; "Mexican Validation" del título de la Sección 3.5):
> "Figure 2 summarizes this participant flow across all dataset partitions. The RSNA Pediatric Bone Age Challenge 2017 dataset (14,236 radiographs) was divided by the challenge organizers into an official training set (12,611 images), an official validation set (1,425 images, held out and never used for model tuning or selection), and an official test set (200 images), which was not used in this study. From the official training set, an inclusion rule based on age frequency (retaining only ages with at least 50 images per month) excluded 828 images, yielding a balanced subset of 11,783 images; this subset was further split 80/20 into an internal training subset (9,426 images) and an internal validation subset (2,357 images, used for model fitting and early stopping). Separately, the official validation set and the Mexican clinical dataset (100 radiographs) were each held out for independent evaluation, the independent RSNA validation and the external Mexican validation, reported by architecture in Section 3.5."

*(Nota: esto recorre en +1 la numeración de figuras a partir de aquí; ver tabla de renumeración de figuras en el punto 11.)*

---

### 5. Nueva sección "3.6 Limitations" (después de "3.5 Mexican Validation", antes de la Tabla 10/"Table 10 contextualizes...") (Comentarios 9 y 12)

**Agrega esto** como sección nueva completa:
> "3.6 Limitations
>
> The fusion stage was trained using regional predictions computed on the same data split used to fit the four upstream segment models, rather than out-of-fold predictions or a separate, disjoint tuning set. During the frozen-extractor phase of fusion training, the fusion head is therefore exposed to segment-model outputs on cases those models were themselves fitted on; this risk is compounded during fine-tuning, when the segment extractors are unfrozen and updated again on the same split. This does not affect the external RSNA and Mexican evaluations, which use entirely held-out data never seen during training, but may make the internal validation MAE (Tables 6-9) an optimistic estimate of the fusion model's true generalization. Relatedly, the checkpoint saved to disk for every trained model corresponds to the final training epoch rather than necessarily the epoch with the best validation loss: early stopping restores the best-validation-loss weights in memory at the end of training, but this occurs after the last on-disk checkpoint has already been written. Each architecture/dataset configuration in this study was trained once, with a single random initialization and no repeated seeds; comparisons between closely performing backbones (Section 3.5) should therefore be interpreted as point estimates. The segmentation model's 76-image internal validation split (Table 5) was also used by early stopping and learning-rate reduction for its own checkpoint selection during training, and is therefore not a blind test set independent of model selection. Training the fusion stage on out-of-fold segment predictions or a disjoint fusion-tuning set, repeating training across multiple seeds, and evaluating segmentation on a genuinely held-out split are identified as necessary future work."

---

### 6. Figura 8: falta insertar Bland-Altman (Comentario 8)

**Agrega esto** (imagen) justo después de la Figura 8 actual (identidad+regresión, pág. 25), como nueva figura, con el número que le corresponda según la tabla de renumeración del punto 11 (marcado abajo como Figura N):
- Imagen: `figures/figura8_bland_altman.png`
- Pie de figura: "**Figure N.** Bland-Altman analysis of predicted vs. TW3 bone age on the external Mexican validation cohort, by backbone. Each blue point represents one patient (mean of TW3 bone age and prediction vs. their difference); the gray line marks zero difference (no bias); solid and dashed red lines mark the mean bias and 95% limits of agreement, respectively. Overlapping points appear as darker regions where multiple patients share similar mean/difference values, most visible for F-VGG16, consistent with its regression collapse (Section 3.3)."

**Agrega esto** como párrafo nuevo, justo después de la imagen y su pie de figura:
> "Figure N presents the Bland-Altman analysis for each backbone on the external Mexican cohort. F-DenseNet121 showed a mean bias of +5.68 months (SD = 21.45, 95% limits of agreement [−36.3, +47.7]), systematically overestimating bone age, while F-InceptionV3 showed a smaller-magnitude bias of −2.37 months (SD = 22.87, 95% limits of agreement [−47.2, +42.4]). F-ResNet50 showed a mean bias of +2.08 months (SD = 24.91, 95% limits of agreement [−46.7, +50.9]). F-VGG16 showed the widest limits of agreement by a large margin (mean bias = −14.81 months, SD = 43.98, 95% limits of agreement [−101.0, +71.4]), consistent with its regression collapse. Although F-DenseNet121 and F-InceptionV3 do not differ significantly in MAE (Table 15), their bias profiles differ considerably, indicating that aggregate MAE alone does not capture systematic over- or under-estimation patterns that may be clinically relevant."

---

### 7. Diagnóstico de F-VGG16: falta el párrafo con evidencia (Comentario 8)

**Agrega esto** justo después del párrafo que describe el colapso de F-VGG16 en la Sección 3.5 (pág. 25-26, el que termina en "...thereby failing to capture the chronological variability of the target population." / "...These results underscore that internal validation metrics frequently underestimate actual clinical error..."):
> "The collapse of F-VGG16 was investigated using its full training history. Early stopping halted training at epoch 7/20 (fusion phase) and epoch 5/10 (fine-tuning), but validation loss and MAE plateaued from the first recorded epoch onward (val MAE oscillating between 34.6 and 40.5 months with no improving trend across 12 total epochs), indicating that additional training would not have resolved the issue. Data leakage is unlikely, as it would be expected to produce artificially strong, not collapsed, performance. The most plausible explanation is an architecture-specific optimization failure: unlike the other three backbones, VGG16 was trained from scratch (WEIGHTS = None, identical across all four backbones) without batch normalization or residual/dense connections, mechanisms present in ResNet50, DenseNet121, and InceptionV3, respectively, known to substantially ease gradient-based optimization from random initialization. This is confirmed quantitatively by the near-zero regression slope (−0.0046) and near-zero correlation (r = −0.06, Figure 8) between predicted and true bone age for F-VGG16, indicating the optimizer converged to a trivial solution near the mean training age rather than learning a genuine image-to-age relationship."

---

### 8. Página 22: referencia sin resolver (Comentario 15)

**Cambia esto:**
> "that comparison is reported separately in Section 3.X (whole-hand baseline)"

**Por esto:**
> "that comparison is reported separately in Section 3.4 (whole-hand baseline)"

---

### 9. Página 23: referencia rota tras la reestructuración (Comentario 8)

**Cambia esto:**
> "consistent with the regression collapse described in Section 3.4.1"

**Por esto** (ajustar al número real de la Sección 3.5 tras insertar el párrafo del punto 7, confirmar contra la numeración final):
> "consistent with the regression collapse described earlier in this section"

---

### 10. Renumeración de tablas: aplicar en este orden exacto (mecánico, resultado de insertar tablas de los Comentarios 3, 7, 9 y 14)

| Rótulo actual | Cambiar a |
|---|---|
| Table Y (pág. 5, distribución antes/después del balanceo) | Table 1 |
| Table 1 (pág. 13, Segmentation Architecture Validation) | Table 2 |
| Table 2 (pág. 15, Experimental Setup) | Table 3 |
| Table 3 (pág. 16, Data Augmentation) | Table 4 |
| Table X (pág. 17, Dice/IoU por clase) | Table 5 |
| Table 4 (pág. 18, F-ResNet50) | Table 6 |
| Table 5 (pág. 19, F-VGG16) | Table 7 |
| Table 6 (pág. 19, F-DenseNet121) | Table 8 |
| Table 7 (pág. 19, F-InceptionV3) | Table 9 |
| Table Z (pág. 18, tiempo de entrenamiento) | Table 10 |
| Table A (pág. 18, latencia/memoria) | Table 11 |
| Table 8 (pág. 20, Results by Model) | Table 12 |
| Table 9 (pág. 23, Mexican Dataset Comparison) | Table 13 |
| Table B (pág. 24, estadísticas RSNA/MEX) | Table 14 |
| Table C (pág. 24, pruebas pareadas) | Table 15 |
| Table 10 (pág. 26, comparación con literatura) | Table 16 |

**Además, corrige esta referencia cruzada rota** (pág. 24, no coincide con el rótulo real de la tabla):
> "Wilcoxon signed-rank tests with Holm correction for multiple comparisons (Table Y)"

Por:
> "Wilcoxon signed-rank tests with Holm correction for multiple comparisons (Table 15)"

---

### 11. Renumeración de figuras: aplicar en este orden exacto (mecánico, resultado de insertar las figuras de los Comentarios 2 y 8)

| Rótulo actual | Cambiar a |
|---|---|
| Figure 1 (pág. 4, pipeline metodológico) | Figure 1 (sin cambio) |
| Nuevo Participant Flow Diagram (punto 4b) | Figure 2 |
| Figure 2 (pág. 7-8, caracterización demográfica, 6 paneles) | Figure 3 |
| Figure 3 (pág. 10, pipeline de preprocesamiento) | Figure 4 |
| Figure 4 (pág. 11, regiones de segmentación) | Figure 5 |
| Figure 5 (pág. 12, modelo U-Net) | Figure 6 |
| Figure 6 (pág. 14, esquema de fusión) | Figure 7 |
| Figure 7 (pág. 21, ejemplos de saliencia) | Figure 8 |
| Figure 8 (pág. 25, dispersión TW3 vs. predicción) | Figure 9 |
| Nueva figura Bland-Altman (punto 6) | Figure 10 |

Todas las referencias en el texto a "Figure N" deben actualizarse conforme a esta tabla, en ese
orden, para no pisar números ya usados a medio camino (igual que con las tablas del punto 10).

---

### 12. Abbreviations (pág. 28): faltan 11 términos usados en el cuerpo del texto

**Agrega esto**, en la lista de abreviaciones, en la posición alfabética correspondiente:

| Abreviatura | Significado |
|---|---|
| CI | Confidence Interval |
| CLAHE | Contrast Limited Adaptive Histogram Equalization |
| DR | Digital Radiography |
| GPU | Graphics Processing Unit |
| HPC | High-Performance Computing |
| IoU | Intersection over Union |
| LR | Learning Rate |
| MAD | Mean Absolute Distance |
| MLP | Multi-Layer Perceptron |
| SD | Standard Deviation |
| UDG | University of Guadalajara |

*(Nota: "HGR" (Hospital General Regional, "HGR 46") también aparece sin expandir, pero es parte
de un nombre propio de institución, similar a "RSNA" mismo; se deja fuera de la lista principal
por ser de menor prioridad.)*
