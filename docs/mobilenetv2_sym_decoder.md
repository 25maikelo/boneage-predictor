# Decoder simétrico a MobileNetV2 — `mobilenetv2_sym`

## 1. El bloque base de MobileNetV2: Inverted Residual

MobileNetV2 construye su encoder con un único tipo de bloque repetido llamado
**inverted residual** (residual invertido). A diferencia de los bloques residuales
clásicos (wide → narrow → wide), este hace lo opuesto: narrow → wide → narrow.

```
Entrada  (narrow)
    │
    ▼
[Conv2D 1×1]  ← Expand: multiplica canales × t (expansion factor, t=6 en MNv2)
[BN + ReLU6]
    │
    ▼ (wide)
[DepthwiseConv2D 3×3, stride=s]  ← Filtra espacialmente canal a canal
[BN + ReLU6]                       stride=2 → reduce resolución ÷2
    │
    ▼
[Conv2D 1×1]  ← Project: comprime canales al tamaño de salida
[BN]          ← Sin activación en el proyecto (diseño original del paper)
    │
    ▼ (narrow)  [+ residual si stride=1 y in_ch == out_ch]
Salida
```

El paso **Expand → Depthwise → Project** es el corazón de MobileNetV2.
La idea es que las convoluciones depthwise son baratas computacionalmente,
pero necesitan operar en un espacio de alta dimensión para ser expresivas —
por eso se expande antes y se comprime después.

---

## 2. El schedule del encoder

MobileNetV2 reduce la resolución espacial progresivamente aplicando `stride=2`
en el depthwise de ciertos bloques, mientras aumenta el número de canales:

| Etapa | Bloque Keras | Resolución | Canales | Expansión | Stride |
|---|---|:---:|:---:|:---:|:---:|
| Stem | `Conv1` | 112×112 | 32 | — | 2 |
| B0 | `expanded_conv` | 112×112 | 16 | 1 | 1 |
| B1–B2 | `block_1–2` | 56×56 | 24 | 6 | 2→1 |
| B3–B5 | `block_3–5` | 28×28 | 32 | 6 | 2→1→1 |
| B6–B9 | `block_6–9` | 14×14 | 64 | 6 | 2→1→1→1 |
| B10–B12 | `block_10–12` | 14×14 | 96 | 6 | 1 |
| B13–B15 | `block_13–15` | 7×7 | 160 | 6 | 2→1→1 |
| B16 | `block_16` | 7×7 | 320 | 6 | 1 |

El bottleneck final es un tensor de **7×7×320**.

---

## 3. El bloque decoder simétrico

El decoder de `mobilenetv2_sym` replica la estructura del bloque encoder
**invirtiendo el stride**: donde el encoder usaba `stride=2` para reducir,
el decoder usa `UpSampling2D(2×2) + DepthwiseConv2D` para aumentar.

```
Encoder                              Decoder (simétrico)
──────────────────────────           ──────────────────────────────────────
[Conv2D 1×1] Expand  ×t             [Conv2D 1×1] Expand  ×t
[BN + ReLU6]                        [BN + ReLU6]
                                          │
[DepthwiseConv2D stride=2]  ←→     [UpSampling2D ×2]           ← invierte
[BN + ReLU6]                        [DepthwiseConv2D stride=1]    el stride
                                     [BN + ReLU6]
                                          │
                                     [Concatenate skip]  ← skip connection
                                          │
[Conv2D 1×1] Project                [Conv2D 1×1] Project
[BN]                                [BN]
```

La concatenación del skip se inserta **después del upsample y antes del project**,
no como una operación separada externa al bloque. Esto mantiene la secuencia
expand → operate spatially → project intacta, con la información de skip
disponible en el momento de la proyección.

---

## 4. Schedule de canales: encoder vs decoder

Los canales del decoder siguen el schedule del encoder **en orden inverso**,
tomando como referencia las etapas de transición de resolución:

| Resolución | Encoder (canales) | Decoder (canales) |
|:---:|:---:|:---:|
| 7×7 | 320 (bottleneck) | 320 → entrada |
| 14×14 | 96 (bloques 10–12) | 160 (salida) |
| 28×28 | 64 (bloques 6–9) | 96 |
| 56×56 | 32 (bloques 3–5) | 64 |
| 112×112 | 24 (bloques 1–2) | 32 |
| 224×224 | 16 (bloque 0) | 16 |

El decoder no replica exactamente el número de canales del encoder en cada
nivel porque los skip connections aportan canales adicionales al project step.
Los valores (160, 96, 64, 32, 16) son el output del project en cada etapa del
decoder, elegidos para mantenerse proporcionales al schedule del encoder.

---

## 5. Skip connections

Los skip connections se extraen de los puntos **expand_relu** del encoder —
es decir, de la representación expandida (wide) de cada bloque con `stride=2`,
capturada **antes** del depthwise con stride. Esto significa que el skip
corresponde a la resolución previa al downsampling:

| Skip | Capa Keras | Resolución | Canales |
|---|---|:---:|:---:|
| s1 | `block_1_expand_relu` | 112×112 | 96 |
| s2 | `block_3_expand_relu` | 56×56 | 144 |
| s3 | `block_6_expand_relu` | 28×28 | 192 |
| s4 | `block_13_expand_relu` | 14×14 | 576 |

Estos son los mismos puntos que usa `build_unet_mobilenetv2`, lo que permite
una comparación directa entre los dos decoders.

---

## 6. Diferencias con las otras arquitecturas

| Arquitectura | Encoder | Upsampling | Bloque decoder | Skip |
|---|---|---|---|:---:|
| `unet_mobilenetv2` | MNv2 | Conv2DTranspose | Conv2D estándar | Sí |
| `mobilenetv2` | MNv2 | Conv2DTranspose | `_depthwise_block` | Sí |
| `mobilenetv2_sym` | MNv2 | UpSampling + DepthwiseConv | expand→up→project | Sí |
| `mobilenetv2_blocks` | Desde cero (DW blocks) | Conv2DTranspose | `_depthwise_block` | Sí |

La diferencia principal de `mobilenetv2_sym` respecto a `mobilenetv2`:

- **`mobilenetv2`**: el upsampling (`Conv2DTranspose`) y el procesamiento
  (`_depthwise_block`) son dos operaciones independientes y externas entre sí.
- **`mobilenetv2_sym`**: el upsampling está integrado **dentro** del bloque
  decoder (entre expand y project), igual que el stride está integrado dentro
  del bloque encoder. La estructura expand → operate → project se mantiene en
  ambos sentidos.

---

## 7. Parámetros del modelo (224×224, 5 clases)

| Sección | Parámetros |
|---|---|
| Encoder MobileNetV2 (imagenet, frozen) | ~1.86 M |
| Decoder simétrico | ~1.48 M |
| **Total** | **~3.34 M** |

El decoder es significativamente más ligero que el encoder porque usa
depthwise convolutions en lugar de convoluciones estándar, consistente
con la filosofía de eficiencia de MobileNetV2.
