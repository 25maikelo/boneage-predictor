# Comentario 12 — Fuga de datos en el entrenamiento de fusión (stacked-model leakage)

> Decisión: no se reentrena (ya se implementó y se revirtió por completo en una sesión anterior,
> a petición explícita). Se responde con declaración honesta de la limitación.

## 📝 Respuesta al revisor

> We confirm this is the case: the fusion stage was trained on regional predictions computed on
> the same split used to fit the regional models, not on out-of-fold predictions or a separate
> tuning set. We have added this as an explicit limitation in Section 3.5, noting it does not
> affect the external RSNA/Mexican evaluations (fully held-out data), but may make the internal
> validation MAE optimistic; out-of-fold fusion training is identified as future work. Early
> stopping (patience = 4 on validation loss) and learning-rate reduction on plateau (factor =
> 0.5, patience = 3) were used identically across all training stages and are now reported in
> Section 2.5/Table 2. Each configuration was trained once, with no repeated seeds; this is also
> now acknowledged as a limitation in Section 3.5.

## Evidencia técnica detrás de la respuesta (por si hace falta consultarla después)

- Fuga confirmada en código: `src/06_training.py:605` (split de segmentos) y `:789` (split de
  fusión) usan `train_test_split(df, test_size=cfg.TEST_SPLIT, random_state=42)` sobre el mismo
  `df` — split idéntico, no independiente.
- Early stopping/LR: `EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)` +
  `ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7)`, idénticos en las
  4 fases (segmento, fusión, fine-tuning, folds de CV).
- Hallazgo adicional (no pedido explícitamente, pero relevante a "checkpoint selection"): el
  checkpoint guardado en disco es el de la última época, no necesariamente el de mejor
  `val_loss` — `SaveModelCallback` guarda en cada `on_epoch_end`, y `restore_best_weights` solo
  actúa en memoria en `on_train_end`, después del último guardado. Verificado empíricamente en
  el experimento whole-hand de esta sesión (mejor época 9, guardada época 10). Aplica a todos los
  modelos del pipeline, no solo a la fusión.
- Una sola corrida por configuración, sin semillas repetidas — confirmado, no hay forma de
  desmentirlo sin reentrenar.
