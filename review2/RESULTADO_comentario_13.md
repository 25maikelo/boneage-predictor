# Comentario 13 — Función de pérdida sin definición matemática

> Fuente de la ecuación: `src/utils/losses.py` (código, no depende de ningún experimento). Fuente de la explicación MAE train &gt; val: `src/06_training.py` (`train_fusion`, `custom_data_generator`, `fusion_data_generator`).

## Definición matemática exacta

El manuscrito usa `LOSS_FUNCTION_NAME = "attention_loss"` (confirmado idéntico en los 4 backbones vía `config.py` de cada experimento). Implementación exacta:

```python
def attention_loss(y_true, y_pred):
    error = tf.abs(y_true - y_pred)
    alpha = tf.abs((y_pred / y_true) - 1)
    return tf.reduce_mean(alpha * error)
```

En notación formal, para un lote de $N$ muestras con edad real $y_i$ y predicción $\hat{y}_i$:

$$\mathcal{L}_{\text{attention}} = \frac{1}{N}\sum_{i=1}^{N} \underbrace{\left|\hat{y}_i - y_i\right|}_{\text{error absoluto}} \cdot \underbrace{\left|\frac{\hat{y}_i}{y_i} - 1\right|}_{\alpha_i \text{ — peso de atención}}$$

**Simplificación algebraica no mencionada en el manuscrito, y clínicamente relevante:** como $\alpha_i = \left|\frac{\hat{y}_i}{y_i}-1\right| = \frac{|\hat{y}_i - y_i|}{y_i}$ (válido para $y_i>0$, siempre cierto aquí ya que `AGE_RANGE` empieza en 24 meses), la función se reduce a:

$$\mathcal{L}_{\text{attention}} = \frac{1}{N}\sum_{i=1}^{N} \frac{(\hat{y}_i - y_i)^2}{y_i}$$

Es decir, **`attention_loss` es un MSE ponderado por el inverso de la edad real** ($w_i = 1/y_i$): para el mismo error absoluto, penaliza mucho más fuerte a las edades pequeñas que a las grandes. Un error de 10 meses en un paciente de 24 meses aporta $100/24 \approx 4.17$ a la pérdida; el mismo error en un paciente de 216 meses aporta solo $100/216 \approx 0.46$ — casi 9 veces menos.

**📝 Reemplaza esto** (Sección 2.5, Experimentation):
> "The loss function was defined as a custom attention-weighted formulation."

**Por esto:**
> "The loss function was defined as $\mathcal{L}_{\text{attention}} = \frac{1}{N}\sum_{i=1}^{N} |\hat{y}_i - y_i| \cdot \left|\frac{\hat{y}_i}{y_i} - 1\right|$, where $y_i$ and $\hat{y}_i$ denote the true and predicted bone age (months) for sample $i$. Algebraically, this is equivalent to a mean squared error weighted by the inverse of the true age, $\mathcal{L}_{\text{attention}} = \frac{1}{N}\sum_i (\hat{y}_i - y_i)^2 / y_i$, which assigns proportionally larger loss to a fixed absolute error at younger ages than at older ages — a deliberate design choice to counteract the lower representation of younger age groups in the training distribution (Section 2.1), stated here explicitly rather than left implicit in the reported loss values."

## Por qué MAE Train > MAE Val (Tablas 4–7)

El manuscrito ya muestra el patrón — p. ej. Tabla 4 (F-ResNet50): MAE Train = 15.6425 > MAE Val = 9.1098; Tabla 6 (F-DenseNet121): MAE Train = 11.1484 > MAE Val = 5.7683 — sin explicarlo. Revisando `src/06_training.py` hay **dos causas concretas y verificables en el código**:

1. **Dropout activo durante el cómputo de la métrica de entrenamiento.** La cabeza de fusión tiene `Dropout(0.5)`. Keras calcula `mae`/`loss` de entrenamiento como el promedio corrido por batch **durante** el forward pass con `training=True` (dropout activo), mientras que `val_mae` se calcula al final de cada época con `training=False` (dropout desactivado, red completa). Esto hace la métrica de entrenamiento sistemáticamente más ruidosa/peor, incluso si el modelo generaliza igual o mejor — es un artefacto de cómo Keras reporta las métricas, no una señal de mal ajuste.
2. **Aumentación de datos activa solo en entrenamiento.** Confirmado en `train_fusion`: `custom_data_generator(train_df, cfg, augment=cfg.USE_AUGMENTATION)` (rotación ±20°, brillo 0.8–1.2×, zoom 0.2) vs. `custom_data_generator(val_df, cfg, augment=False)`. El modelo entrena sobre imágenes deliberadamente perturbadas y valida sobre imágenes limpias — la tarea de entrenamiento es, por diseño, más difícil.

Sobre "report metrics obtained after each final model was set to inference mode": **`val_mae`/`val_loss` en las Tablas 4–8 ya están en modo inferencia** (`training=False`, sin dropout ni augmentación) — es el número que corresponde citar como desempeño real del modelo. El `MAE Train` de esas mismas tablas es la métrica de entrenamiento con dropout/augmentación activos y debería aclararse como tal.

**📝 Agrega esto** como nota al pie de las Tablas 4–7, o justo después de la Tabla 3 (Data Augmentation Configuration):
> "Reported 'MAE Train' values are computed during the forward pass with dropout (rate = 0.5 in the fusion layer) and data augmentation active, whereas 'MAE Val' values are computed in inference mode (dropout disabled, no augmentation) at the end of each epoch. This explains why MAE Train exceeds MAE Val in Tables 4–7: the two metrics are not measured under the same conditions, and MAE Val — not MAE Train — reflects the model's actual inference-time performance."

## Lo que el comentario pide y que sigue sin poder responderse con datos

- **"Performance should be replicable across multiple random seeds and summarized using the mean with either standard deviation or confidence interval."** Esto está directamente ligado al Comentario 12 (una sola semilla, arquitecturas potencialmente inestables). Sin reentrenar con semillas distintas no hay forma de dar una media±DE real — solo se puede declarar como limitación, la misma decisión pendiente que en #12/#15.

**📝 Agrega esto** como limitación explícita (Sección 3.5):
> "All results in this study are reported from a single training run per configuration. Given known training instability in deep learning architectures (most visibly observed in F-VGG16, Section 3.4), performance replicated across multiple random seeds — reported as mean ± standard deviation or confidence interval — is identified as necessary future work."
