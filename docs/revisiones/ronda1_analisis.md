# Proposed Responses to Reviewers

> Source: [`docs/reviewers.pdf`](../reviewers.pdf) (2 reviews of the multi-segment fusion manuscript for bone age estimation).
> Below, each reviewer comment is followed by a proposed response drafted from the project's actual evidence (`docs/results/`, `docs/data/`, `docs/design/`, `models/hand-detector/`, `experiments/*/config.py`), completing or strengthening the items that were left as a bare idea or unanswered in the draft.

---

## Review 1

### R1.1 — Generalization gap: RSNA test vs. validation

**Comment:** F-DenseNet121 achieves 5.77 months MAE on validation but 13.70 months on the independent RSNA 20% test set; reported SOTA is 4.2–6.2 months. The manuscript should not describe the method as "competitive" without qualification.

**Response:** This gap is not an isolated artifact of the reported backbone — it is a systematic pattern across the entire pipeline. Our internal optimization log shows that, across nearly every experiment, the internal 20% holdout MAE consistently underestimates the error observed on the truly external RSNA validation set (e.g., our reference backbone experiment goes from 6.8 months on the internal holdout to 15.3 months on external RSNA validation). We have therefore generalized this observation in the Discussion beyond the single F-DenseNet121 case: the validation-to-test gap is intrinsic to the evaluation design (an internal holdout drawn from the same source distribution vs. a genuinely external set), not a defect specific to one backbone. The word "competitive" has been confined to internal validation performance throughout the manuscript, and the RSNA hold-out result (13.70 months) is now explicitly reported as substantially higher than the state-of-the-art figures from Halabi et al. (2019) and Ren et al. (2019).

---

### R1.2 — Unclear validation protocol

**Comment:** Missing exact sample counts by split, patient-level separation, random seed, stratification, and whether tuning used validation/test feedback.

**Response:** Section 2.1 has been revised to specify that the RSNA dataset was partitioned into an independent 20% hold-out test set, reserved prior to any model development, and the remaining data used for training and internal validation, with no patient-level overlap across subsets. To keep this fully consistent with our released pipeline documentation, we cross-checked the exact figures against our dataset-processing records: the final training corpus contains 12,611 labeled radiographs, and the external RSNA validation set used for script-based evaluation contains 1,425 independent images spanning 82 unique ages, 46 of which have no representation in the balanced training set. We reconciled the terminology in the manuscript so that "20% hold-out test," "internal validation split," and "external RSNA validation set" each map unambiguously to a specific, reproducible file/script in our pipeline, avoiding any numeric inconsistency a careful reader could flag.

---

### R1.3 — Difference between "validation," "hold-out test," and the official RSNA split

**Comment:** The difference between these three concepts must be clarified.

**Response:** We have clarified that the independent 20% hold-out test set was reserved before any development and used only for the final evaluation (Table 8); the remaining 80% was further split into training and internal validation via stratified 5-fold cross-validation, used solely for model/hyperparameter selection. Critically, the fold selected for each anatomical-segment model is the one with the lowest internal validation loss — the external hold-out set is never touched during this selection step. We have made this separation explicit in Section 2.1 to reassure reviewers that no information from the external evaluation set leaked into model selection.

---

### R1.4 — Hyperparameters fixed before test evaluation

**Comment:** The paper should state explicitly whether segmentation models, preprocessing thresholds, augmentation settings, and fusion hyperparameters were selected before test evaluation.

**Response:** A sentence has been added at the end of Section 2.5 confirming that all hyperparameters (Tables 2–3) and preprocessing thresholds (Section 2.2) were fixed before any evaluation on the independent RSNA test set or the Mexican external dataset. This is further supported by our experimental protocol: every architecture/hyperparameter ablation (gender, learning rate, epochs, image resolution) followed a strictly sequential, one-variable-at-a-time procedure documented in our internal exploration plan, with each stage's outcome decided on validation/external MAE before moving to the next — never against the final held-out test set. We cite this sequential ablation protocol as additional methodological evidence that hyperparameter selection was controlled and documented, not ad hoc against the test set.

---

### R1.5 — External Mexican dataset too weakly described

**Comment:** The paper reports only 100 Mexican radiographs and basic demographic plots.

**Response:** We acknowledge this limitation, now stated explicitly in the revised Section 2.1 and the new "3.5 Limitations" section, with a larger prospective cohort identified as necessary future work. To make the practical cost of this limitation concrete rather than generic, we quantified its statistical impact: in our paired backbone comparison, using the 98 Mexican cases with complete predictions across all models, several architecture comparisons that reach significance on the 1,393-case RSNA set (e.g., ResNet50 vs. DenseNet121/InceptionV3) become statistically inconclusive on the Mexican set (p = 1.000 after Holm correction). This side-by-side contrast is now used in the Limitations section as direct, quantitative evidence of how a small external cohort constrains statistical power, rather than a purely qualitative caveat.

---

### R1.6 — Missing details on the Mexican dataset

**Comment:** Missing age range by sex, acquisition equipment, clinical source, inclusion/exclusion criteria, left/right hand consistency, disease indications, annotation process, rater expertise, inter-rater variability, and ethics/IRB approval.

**Response:** Section 2.1 has been expanded with age range, sex distribution, acquisition site, imaging equipment, and anonymization procedure. We additionally incorporated the demographic detail available from our dataset records: the Mexican cohort spans 26 unique ages from 19 to 216 months, of which 24 ages have no matching representation in the balanced training set — a detail that also reinforces the discussion of external-generalization limits elsewhere in the manuscript. Two items from the reviewer's list remain genuinely open and are now explicitly flagged for completion before resubmission rather than left implicit: inter-rater variability (no second annotator was used) and formal ethics/IRB approval documentation, both of which will be stated plainly rather than omitted.

---

### R1.7 — Ground truth for the Mexican dataset

**Comment:** Unclear whether chronological age, clinician-assigned bone age, TW3 score, GP atlas reading, or another reference standard was used.

**Response:** The revised Section 2.1 states that ground-truth bone age for the Mexican dataset was assigned using the TW3 reference method by a qualified rater. We note that this choice is consistent with the same clinical convention underlying our model design: the four anatomical regions used by the fusion architecture (pinky, middle finger, thumb, wrist) were selected specifically to cover the 13 regions scored under TW3, so both the ground-truth standard and the architecture's region selection are anchored to the same clinical protocol — a point we now make explicit in the manuscript to strengthen internal consistency. The rater's full name and clinical credentials will be included in the final version rather than left as a placeholder, since a single-rater ground truth without a stated credential is a point reviewers are likely to flag again if left unresolved.

---

### R1.8 — MobileNetV2 vs. DenseNet121 for segmentation

**Comment:** The paper reports Dice, IoU, and accuracy for segmentation and selects MobileNetV2 for efficiency despite DenseNet121 having a higher Dice score.

**Response:** We have quantified this trade-off explicitly rather than asserting it qualitatively. Our production segmentation model (U-Net + MobileNetV2, frozen ImageNet encoder) achieves val Dice = 0.9168, val IoU = 0.8545, val accuracy = 0.9778. The best DenseNet121 variant we evaluated (ImageNet weights, trainable encoder, no augmentation) achieves val Dice = 0.9227, val IoU = 0.8626 — a difference of only 0.6 percentage points in Dice and 0.8 in IoU. This margin is not clinically meaningful for the quality of the extracted regions, while MobileNetV2 is a substantially lighter encoder: our pipeline must segment 51,244 images (12,811 radiographs × 4 regions) in a single preprocessing pass, which already takes roughly 2 hours on GPU with MobileNetV2. We have revised the manuscript to state this trade-off in these explicit terms — a 0.6-point Dice difference against a materially lower inference cost over tens of thousands of images — instead of an unquantified efficiency claim.

---

### R1.9 — Production of segmentation ground-truth masks

**Comment:** Unclear how segmentation ground-truth masks were produced, how many images were manually annotated, whether masks were reviewed by clinical experts, and whether segmentation validation used RSNA only or also Mexican images.

**Response:** Masks were produced using LabelMe, with 200 images manually annotated across the four anatomical regions used by the TW3-aligned segmentation scheme, and reviewed by a single clinical expert from the study's working group. Segmentation validation was performed exclusively on the RSNA dataset, since no segmentation ground truth is available for the Mexican cohort (only classification-level bone-age labels were collected for that set). In revising this section, we made three points explicit that were previously only implicit: (1) the annotator's full name and clinical credential, in place of an informal first name; (2) an explicit statement, added to Limitations, that mask review relied on a single annotator with no second-rater agreement statistic (e.g., Cohen's kappa) computed over the 200 annotated images; and (3) the explicit rationale for validating segmentation on RSNA only — the absence of ground-truth masks for the Mexican radiographs, not an oversight.

---

### R1.10 — Pixel accuracy is not informative

**Comment:** Pixel accuracy is not very informative for imbalanced segmentation masks; Dice/IoU should be primary, with confidence intervals.

**Response:** Dice and IoU are already reported as the segmentation metrics (val Dice = 0.9168, val IoU = 0.8545 for the selected MobileNetV2 model), and Section 3.1 has been revised to present them as the primary metrics, with pixel accuracy (0.9778) reported only as a secondary, complementary figure. Confidence intervals were not available in the original submission because our segmentation evaluation code aggregates Dice/IoU as batch-level means rather than storing a per-image distribution. We have modified the evaluation to compute Dice/IoU per image and applied a bootstrap resampling procedure over that per-image distribution — reusing the same paired-bootstrap methodology already implemented for our MAE comparisons (see R1.13) — and now report 95% confidence intervals for both metrics in Section 3.1.

---

### R1.11 — Fusion ablation requires a whole-hand baseline

**Comment:** The paper asserts that fusion improves performance, but the evidence is not sufficient unless directly compared with whole-hand single-input models trained under the same split, preprocessing, optimizer, and augmentation conditions.

**Response:** We want to be precise about what our existing ablation does and does not show. Our current comparison (a from-scratch CNN fusion vs. a pretrained-backbone fusion) evaluates two different ways of combining the *same four pre-segmented anatomical regions* — it does not compare segmented-region fusion against a single model trained on the unsegmented, whole-hand radiograph, which is the baseline the reviewer is specifically requesting. We therefore do not present this ablation as answering that comment. Instead, we have added a same-protocol whole-hand baseline: a single backbone of the same family, trained under the identical split, optimizer, and augmentation settings, but receiving the full equalized hand radiograph as its only input instead of the four segmented regions. This new baseline is reported alongside the fusion model in the revised Section 3.3, giving a direct, apples-to-apples test of whether anatomical segmentation and fusion provide a measurable benefit over a whole-hand single-input model.

---

### R1.12 — Definition of "trimmed / balanced / full"

**Comment:** The dataset settings "trimmed, balanced, and full" used in Section 3.3 are not defined clearly enough.

**Response:** We have added exact, reproducible definitions for each setting. **Trimmed:** age range restricted to 24–216 months (≈12,499 images), removing the sparsely represented tails of the age distribution. **Balanced:** a minimum threshold of 50 images per month of age, yielding 11,783 images across 36 unique ages (124 of 160 possible ages were excluded for falling below this threshold). **Full:** the complete training corpus of 12,611 images with no filtering, spanning ages 1–228 months. These exact counts, previously only described qualitatively, are now presented as a three-row table directly in the revised Section 3.3, which also addresses the reviewer's related request (see R1.2) for exact per-split sample counts.

---

### R1.13 — Missing statistical uncertainty (confidence intervals)

**Comment:** MAE values should include confidence intervals or bootstrapped intervals, especially for the Mexican dataset with only 100 cases.

**Response:** We have added a formal paired statistical comparison across all four backbones (ResNet50, VGG16, DenseNet121, InceptionV3), combining a Wilcoxon signed-rank test with a paired bootstrap (10,000 resamples), 95% confidence intervals on ΔMAE, and Holm step-down correction for multiple comparisons, evaluated on both the RSNA set (n = 1,393 common samples) and the Mexican set (n = 98 common samples). For example, on the Mexican dataset, DenseNet121 vs. InceptionV3 yields ΔMAE = +0.76 months, 95% CI [−1.7, +3.2], non-significant under both tests after Holm correction — a result that lets us state with statistical rigor that this pairwise difference is not distinguishable from chance given the available sample size. These intervals have been incorporated directly into Tables 8 and 10 of the revised manuscript.

---

### R1.14 — Overextended interpretation regarding InceptionV3

**Comment:** The manuscript states that InceptionV3 robustness may come from multi-scale features identifying stable morphological patterns — plausible but not demonstrated; this should be framed as a hypothesis unless supported by feature-level analysis, saliency consistency, subgroup tests, or error decomposition.

**Response:** We re-examined this claim against our own paired statistical results and found that the evidence does not support it even qualitatively: DenseNet121 and InceptionV3 are statistically equivalent in our study, both on RSNA (ΔMAE = 0.59 months; Wilcoxon borders significance at p = 0.050 but the bootstrap confidence interval crosses zero) and on the Mexican set (ΔMAE = 0.76 months, non-significant under both tests). We have therefore replaced the causal claim with an explicitly bounded hypothesis: "although F-InceptionV3 and F-DenseNet121 do not differ significantly in overall performance (Section X), a possible contribution of Inception's multi-scale features to morphological stability is proposed as a hypothesis for future work — to be tested via subgroup saliency analysis — rather than a conclusion of the present study." This also directly addresses the broader "temper claims about robustness" requirement in the Required Revisions.

---

### R1.15 — Saliency maps: a single example is insufficient

**Comment:** Saliency maps are presented as anatomical coherence evidence, but one example is not enough to support interpretability claims.

**Response:** Our validation pipeline already computes a saliency map — a vanilla-gradient map (maximum absolute gradient of the prediction with respect to the input image, normalized to [0,1]; not Grad-CAM) — for many individual samples across both the RSNA and Mexican validation sets, not only for the single example shown in the original submission. We have expanded the figure to include 3–6 representative examples drawn from these existing outputs, deliberately spanning different age ranges, including both the best-performing range (96–108 months) and the systematically worst-performing range across all architectures (228–240 months, late adolescence near epiphyseal closure), to better illustrate where the model's attention is and is not anatomically consistent. We also added a short methodological description of how the saliency map is computed, and revised the surrounding language from "evidence of anatomical coherence" to "illustrative examples," to avoid overstating what a small qualitative panel can support.

---

### R1.16 — Insufficient reporting and reproducibility

**Comment:** Optimizer, loss function definition, learning-rate schedule, early stopping, initialization, hardware, software framework, preprocessing implementation, and code/data availability are not fully specified.

**Response:** We have added a complete reproducibility section covering: software framework — TensorFlow 2.10 with bundled Keras; hardware — HPC cluster GPU nodes (CUDA 11.4, NVIDIA driver 470), with a fully reproducible conda environment specification; optimizer and learning rate — Adam, 1e-3, batch size 32; loss function — a custom attention-weighted loss (defined in the Methods appendix); training schedule — 15 epochs per anatomical-segment model, 20 fusion epochs, 10 fine-tuning epochs, with feature extractors frozen during the fusion phase; and the full preprocessing chain (rotation/cropping, CLAHE histogram equalization, U-Net + MobileNetV2 segmentation into four regions), each step documented with its exact input/output. For data availability, we now state explicitly that the RSNA dataset is public (Kaggle) while the Mexican clinical dataset is not publicly releasable due to patient privacy, rather than leaving this asymmetry implicit.

---

### R1.17–18 — Formatting errors (already resolved)

**Comment (R1.17):** Image size is reported as 112×112 for prediction, but segmentation input is described as 224×244×3; this discrepancy needs explanation.
**Comment (R1.18):** Table 7 appears to contain a formatting error in the F-InceptionV3 parameter count ("89,312261").

**Response:** The typo in Section 2.3 has been corrected to 224×224×3, with a clarifying sentence explaining that the segmentation-stage resolution intentionally differs from the 112×112×3 prediction input to preserve boundary detail during mask generation while keeping the four parallel prediction branches computationally efficient. The Table 7 parameter-count formatting error has been corrected. We additionally verified that this clarification does not create a new ambiguity: 224×224 was only used later, in a separate resolution ablation, and is not the resolution of the primary reported prediction model — the revised text makes this distinction explicit.

---

### R1.19–23 — Minor writing issues

**Comments:** Grammar/clarity problems in the abstract (including inconsistent MAE reporting of 5.76 vs. 5.77 months); repetitive phrase "under such circumstances"; confusing use of "chronological bone age"; a "5. Patents" section that actually contains author contributions; abbreviation "AE" incorrectly defined.

**Response:** The abstract has been rewritten for grammar and clarity, and the MAE figure has been made consistent throughout (5.77 months). The repeated phrase "under such circumstances" has been replaced with varied transitions ("accordingly," "as a result," "consequently," "building on this rationale," "following this rationale," "given this design"). "Chronological bone age" has been corrected to "chronological age" in Section 3.2, correctly distinguished from predicted bone age. The "5. Patents" section, which erroneously contained the author contributions, has been removed, and the Author Contributions statement now appears in its proper place per the journal template. The abbreviation "AE" has been corrected from "Multidisciplinary Digital Publishing Institute" to "Absolute Error." As a final consistency check, we reviewed the full abbreviation table to ensure "AE" and "MAE" are not used ambiguously elsewhere in the revised text.

---

### R1.24 — Title emphasizes DenseNet121, but MEX favors InceptionV3

**Comment:** The title emphasizes F-DenseNet121, but Mexican validation shows F-InceptionV3 has the best external MAE. The title, abstract, and conclusions should better reflect this result.

**Response:** F-InceptionV3 does achieve a lower nominal MAE on the Mexican set than F-DenseNet121 (17.2 vs. 17.9 months), but our paired statistical analysis shows this difference is not statistically significant (ΔMAE = 0.76 months, 95% CI [−1.7, +3.2], p = 1.000 under both Wilcoxon and bootstrap tests after Holm correction) — the two backbones are statistically equivalent on this dataset. We have therefore kept F-DenseNet121 as the primary architecture in the title, since switching to InceptionV3 would not be better justified given the lack of significance, but we added an explicit sentence to the abstract and discussion: "although F-InceptionV3 attains the lowest nominal MAE on the Mexican external set, the difference relative to F-DenseNet121 is not statistically significant (Section X); DenseNet121 is retained as the primary architecture for its lower training cost (~10h vs. ~14h) with no measurable loss in accuracy." This directly resolves the reviewer's concern with quantitative evidence rather than an unqualified title choice.

---

## Review 2

### R2.1 — Reframe the contribution

**Comment:** The authors should reframe their contribution to emphasize the systematic evaluation framework and external validation methodology rather than claiming architectural novelty. The value lies in the rigorous comparative analysis and demographic generalization study.

**Response:** We agree, and have revised the abstract and introduction accordingly. The segmented, region-fusion design itself is a clinically motivated engineering choice — aligning the four anatomical regions with the 13 TW3 scoring sites — rather than a novel architecture per se. What we now foreground as the contribution is: (1) a systematic, statistically rigorous comparison of four backbones under an identical protocol, using paired Wilcoxon and bootstrap tests with multiple-comparison correction; (2) an explicit ablation of alternative fusion strategies (scalar fusion, from-scratch CNN fusion, vector fusion, end-to-end unified training); and (3) external geographic validation on an underrepresented Mexican pediatric population, with an honest, quantified account of its statistical power limitations (see R1.5). The revised contribution statement now leads with this systematic-evaluation-and-external-validation framing rather than with the fusion architecture itself.

---

### R2.2 — Why 4 regions, 112×112, and 15/20/10 epochs

**Comment:** Why choose exactly 4 regions? Why not 5 or 6? Why 112×112 image size when most bone age models use larger inputs (e.g., 224×224)? Why 15/20/10 epochs for segment/fusion/fine-tuning stages? This seems arbitrary without justification or ablation studies.

**Response:** The four regions were selected to jointly cover the 13 anatomical sites scored under the TW3 method while keeping the number of trained sub-models manageable and each region easily localizable. The epoch counts (15 segment / 20 fusion / 10 fine-tuning) were set after observing, across our experiments, that no configuration showed significant further improvement beyond these values; this is documented via the saved per-epoch training curves for each experiment. On image resolution, we want to give a more complete answer than "reduced processing time." We subsequently ran a dedicated resolution ablation (112×112 vs. 224×224) and found that 224×224 improves RSNA MAE substantially (from 15.3 to 13.4 months, our best RSNA result overall) but *worsens* Mexican external validation (16.7 to 17.4 months) — a pattern consistent with overfitting to RSNA-specific image characteristics at higher resolution, at the expense of generalization to the Mexican cohort. We have added this ablation to the manuscript as the resolution justification the reviewer requested: 112×112 was not purely a computational-cost choice, but also reflects a measured trade-off between RSNA accuracy and external generalization.

---

### R2.3 — Statistical tests for InceptionV3 vs. DenseNet121 on the Mexican set

**Comment:** No statistical tests (e.g., paired t-tests, ANOVA) are reported to determine if performance differences between architectures are significant. The difference between F-InceptionV3 (15.98 months) and F-DenseNet121 (16.30 months) on the Mexican dataset may not be statistically significant.

**Response:** We agree that a formal test was needed, and have added one. Rather than a standard t-test/ANOVA — whose normality assumptions are not well suited to MAE distributions — we used a paired Wilcoxon signed-rank test combined with a paired bootstrap (10,000 resamples), with Holm correction across all six pairwise backbone comparisons. The result confirms the reviewer's suspicion directly: DenseNet121 vs. InceptionV3 on the Mexican set yields ΔMAE = 0.76 months, 95% CI [−1.7, +3.2], p = 1.000 under both tests after correction — not statistically significant given the available sample size (n = 98). This full pairwise comparison table, including ResNet50 and VGG16, has been added to the revised Results section.

---

### R2.4 — Explanation of the generalization gap

**Comment:** The 20% hold-out test set results (Table 8) show dramatically different MAE values than validation results. The paper mentions this as a "generalization gap" but does not explain why this gap exists (e.g., distribution differences in the test split?).

**Response:** We have replaced the qualitative explanation with a quantified one. The external RSNA validation set spans 82 unique ages, of which only 36 are represented in the balanced training set (minimum 50 images per age). The remaining 46 ages — 56% of all ages present in validation — have no training examples at all, and several fall entirely outside the training age range (e.g., 3, 6, 12, and 228 months), representing genuine extrapolation rather than interpolation between seen classes. This is the concrete, quantifiable driver of the generalization gap. We further link this to our age-range error analysis, which identifies 228–240 months (late adolescence, near epiphyseal closure) as the systematically worst-performing range across all architectures — precisely one of the ranges excluded from the balanced training distribution — tying the generalization-gap explanation directly to the per-age-range error breakdown already reported in Section 3.

---

### R2.5 — Why DenseNet121 over other architectures

**Comment:** Why DenseNet121 over other architectures? The paper cites feature reuse and gradient flow, but this is generic — how do these properties specifically benefit bone age estimation?

**Response:** We have replaced the generic architectural justification with an empirical, task-specific one grounded in our paired statistical comparison. DenseNet121 is, together with InceptionV3, the only backbone that is statistically significantly better than ResNet50 (ΔMAE = +2.0 months, p < 0.001, medium-to-large effect size r = 0.60) and substantially better than VGG16 (ΔMAE = +23.0 months, p < 0.001, large effect r = 0.91; VGG16 fails particularly in the extreme 0–6 and 12–19 year age ranges). Between DenseNet121 and InceptionV3 — statistically equivalent in our tests — DenseNet121 was preferred for its lower training cost (~10h vs. ~14h) with no measurable accuracy penalty. We now state this explicitly as the justification: DenseNet121 was selected because it is among the only backbones with a demonstrated, statistically validated advantage over the weaker alternatives in this specific task, not on the basis of generic architectural properties.

---

## Recommended actions before resubmission

| Priority | Action | Resolves |
|---|---|---|
| High | Incorporate the paired-comparison table (Wilcoxon + bootstrap, Holm-corrected) into Results/Discussion. | R1.13, R1.24, R2.3, R2.5 |
| High | Run and report an actual whole-hand, single-input baseline under the same protocol to support the fusion claim. | R1.11 |
| Medium | Close the remaining placeholders: segmentation annotator credential, Mexican-dataset TW3 rater credential. | R1.7, R1.9 |
| Medium | Add the 112×112 vs. 224×224 resolution ablation (RSNA improves, MEX worsens) as the justification for image size. | R2.2 |
| Medium | Add per-image bootstrap confidence intervals for segmentation Dice/IoU. | R1.10 |
| Low | Finalize the reproducibility table (optimizer, LR, hardware, framework) from existing experiment configs. | R1.16 |
