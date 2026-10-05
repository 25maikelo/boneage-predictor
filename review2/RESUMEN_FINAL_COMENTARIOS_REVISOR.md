# Segunda ronda de revisión: resumen final de los 20 comentarios del Revisor 1

> Documentación de cierre. Consolida `ANALISIS_COMENTARIOS.md` y los 10 documentos
> `RESULTADO_comentario*.md` producidos durante esta ronda en un solo archivo de referencia.
> Manuscrito: *A Multi-Segment Fusion Architecture for Bone Age Estimation* (revista *Technologies*,
> MDPI). Todos los números citados aquí están verificados contra el código y los datos del
> proyecto: no hay cifras inventadas.

## Estado por comentario

| # | Tema | Estado |
|---|---|---|
| 1 | Limpieza general / versión sin tracked changes | 🔒 Bloqueado: requiere el `.docx` fuente |
| 2 | Diagrama de flujo de participantes | ✅ Resuelto |
| 3 | Distribución antes/después del balanceo + umbral de 50 | ✅ Resuelto |
| 4 | Terminología TW3 vs. edad cronológica (Figura 8) | ✅ Resuelto |
| 5 | Confiabilidad inter-observador (estándar mexicano) | ⏳ Requiere trabajo clínico real |
| 6 | Representatividad de la cohorte mexicana | ✅ Resuelto |
| 7 | Estadística externa incompleta (IC, RMSE, sesgo, estratificación) | ✅ Resuelto |
| 8 | Gráficos de dispersión + colapso de F-VGG16 | ✅ Resuelto |
| 9 | Validación de segmentación (split real + Dice/IoU por clase) | ✅ Resuelto (con limitación declarada) |
| 10 | Segmentación no probada en imágenes mexicanas | ⏳ Requiere trabajo manual real |
| 11 | Asociación TW3 exagerada | ✅ Ya resuelto en el manuscrito, solo faltaba la respuesta |
| 12 | Fuga de datos en el entrenamiento de fusión | ✅ Resuelto (declaración de limitación, sin reentrenar) |
| 13 | Función de pérdida sin definición matemática | ✅ Resuelto |
| 14 | Comparación de arquitecturas no controlada | ✅ Resuelto |
| 15 | Ablación insuficiente (falta baseline whole-hand) | ✅ Resuelto: experimento nuevo, datos reales |
| 16 | Mapas de saliencia sin reproducibilidad | ✅ Resuelto (reencuadrado como ilustrativo) |
| 17 | Tabla 10 no comparable con la literatura | ✅ Ya resuelto en el manuscrito, solo faltaba la respuesta |
| 18 | Declaración de ética inaceptable | ✅ Resuelto |
| 19 | Limpieza editorial completa | 🔒 Bloqueado: requiere el `.docx` fuente |
| 20 | Decisión general | ⏳ Se redacta al cerrar 1, 5, 10, 19 |

---

## Hallazgos críticos encontrados durante la revisión

**Hallazgo #1: Bug `real_age` vs. `bone_age`.** `src/08_mex_validation.py` comparaba las
predicciones del modelo (que predice edad ósea TW3) contra `real_age` (edad cronológica) en vez
de `bone_age` (edad ósea asignada por TW3), pese a usar `bone_age` correctamente en otras partes
del mismo script. Corregido en código (commit `9ccd9a6`) y **todos** los números de validación
mexicana citados en este documento ya están recalculados con la corrección. Efecto principal:
**F-DenseNet121 pasa a ser el backbone con el MAE numéricamente más bajo en MEX** (16.38 vs. el
17.28 de F-InceptionV3, que antes figuraba como mejor): aunque la diferencia entre ambos sigue
sin ser estadísticamente significativa.

**Hallazgo #2: Respuestas 13/14 cruzadas** en el borrador de respuestas al revisor (la
Respuesta 14 contestaba al Comentario 13 por error, dejando el 14 efectivamente sin responder).
Corregido al redactar las respuestas de este documento.

**Hallazgo #3: Artefacto de fusión de texto**: *"inspired based by theon the Tanner–Whitehouse 3
(TW3) technique"* (pág. 3): evidencia en vivo de que la limpieza editorial del Comentario 19
sigue pendiente. Bloqueado junto con #1 y #19 por falta del `.docx` fuente.

---

## Comentarios resueltos: detalle

### #2: Diagrama de flujo de participantes
Se construyó un diagrama de flujo (`figures/participant_flow_diagram.png`) que seguía iterando
hasta usar exactamente la terminología del manuscrito ("official training/validation/test set",
"balanced subset", "internal training/validation subset", "Mexican clinical dataset"), sin
conectar visualmente conceptos no relacionados (se corrigió una flecha que implicaba una relación
falsa entre el test set oficial de RSNA, no usado, y el cohorte mexicano).

### #3: Distribución antes/después del balanceo
Verificado contra los CSV reales: raw = 12,611 imágenes (160 edades, 1–228 meses, 54.2%♂/45.8%♀)
→ balanceado = 11,783 imágenes (36 edades, 24–216 meses, 53.6%♂/46.4%♀). El filtro excluye 124 de
160 edades (−828 imágenes, 6.6%), mayormente valores aislados con <10 muestras. Umbral de 50
imágenes/mes justificado como mínimo para un batch completo (`BATCH_SIZE=32`). La Figura 2 del
manuscrito se extendió de 4 a 6 paneles (RSNA raw, México, RSNA balanceado) en vez de crear una
figura nueva separada. Sobre el pedido de "demostrar que el balanceo mejora el desempeño en un
dataset independiente": con los datos de MEX ya corregidos (Hallazgo #1), el dataset balanceado
sigue siendo el mejor de las tres variantes (trimmed/balanced/full) pero por un margen modesto
(16.42 vs. 16.82/18.37 meses, no los ~3-4 meses que sugerían los números con el bug).

### #4, #7, #8: Terminología TW3, estadística externa, dispersión y colapso de VGG16
Con los datos de MEX corregidos: tabla completa de MAE/RMSE/mediana AE/sesgo/DE/±6m/±12m para
los 4 backbones en RSNA (n=1,393) y MEX (n=99); pruebas pareadas (Wilcoxon + bootstrap 10,000
remuestreos, corrección de Holm) confirman que **F-DenseNet121 y F-InceptionV3 siguen siendo
estadísticamente indistinguibles** en ambos datasets (MEX: ΔMAE=−0.90, p=0.70/0.50). Hallazgo
clínico adicional: pese al MAE similar, DenseNet121 sobreestima sistemáticamente (sesgo +5.68m)
mientras InceptionV3 subestima levemente (−2.37m): invisible en el MAE agregado. Nueva Figura 8
(identidad + regresión + IC 95%) y figura Bland-Altman generadas con terminología TW3 corregida.
Diagnóstico del colapso de F-VGG16 con evidencia real de `training_history`: no es falta de
entrenamiento (estancado desde la época 1) ni fuga de datos (produciría desempeño artificialmente
bueno, no un colapso), sino falla de optimización específica de la arquitectura: VGG16 es la
única de las 4 sin BatchNorm ni conexiones residuales/densas al entrenar desde cero.

### #6, #18: Representatividad mexicana y declaración de ética
Terminología "single-center Mexican clinical cohort" ya adoptada en el manuscrito. Limitación de
datos clínicos (solo edad/sexo disponibles) declarada explícitamente. La sección formal
"Institutional Review Board Statement" se sincronizó con el detalle que ya existía en la Sección
2.1 (comité UDG, expediente 26-110, acuerdo RG/ACC/75/2019).

### #9: Validación de segmentación
El manuscrito decía "200 radiografías anotadas"; el dataset real tiene **379** (303 train / 76
val, split reproducible `random_state=42`), confirmado contra el log de entrenamiento del modelo
activo (`hand-detector_00`). Dice/IoU por clase calculado sobre el split real: meñique 0.78,
medio 0.89, pulgar 0.86, muñeca 0.77 (macro 0.83): notablemente más bajo y heterogéneo que el
0.9168 agregado que reporta el manuscrito (calculado en modo "soft"/probabilístico, no por clase).
**Limitación honesta declarada**: las 76 imágenes de "validación" no son un test ciego: son el
mismo conjunto que `EarlyStopping`/`ReduceLROnPlateau` usó para seleccionar el checkpoint del
segmentador. No se entrenó un segmentador nuevo con split de 3 vías porque `hand-detector_00` es
el modelo que generó las máscaras de **todos** los experimentos del estudio; evaluar un modelo
distinto describiría un segmentador que nunca se usó realmente.

### #11, #17: Ya resueltos en el manuscrito
Ambos solo requerían confirmar en la respuesta al revisor que el cambio ya estaba hecho en el
texto (terminología "inspired by" en vez de "implements" para #11; reencuadre de la Tabla 10 como
contexto no competitivo para #17). Sin cambios de contenido.

### #12: Fuga de datos en el entrenamiento de fusión (stacked-model leakage)
**Confirmado real**: `src/06_training.py` entrena el modelo de fusión sobre el mismo split
(`random_state=42`) que los modelos de segmento, así que durante la fase congelada de fusión, la
cabeza de fusión ve salidas de los modelos de segmento calculadas sobre casos en los que esos
modelos fueron ajustados. **Decisión explícita del autor: no reentrenar** (ya se había
implementado y revertido una corrección completa: split de 3 vías: en una sesión anterior).
Se optó por declarar la limitación honestamente en vez de rehacer el pipeline. Hallazgo adicional
encontrado en el camino: el checkpoint guardado en disco es el de la última época, no
necesariamente el de mejor `val_loss`, pese a `restore_best_weights=True`: aplica a todos los
modelos del pipeline, no solo a la fusión.

### #13: Función de pérdida sin definición matemática
Ecuación formal extraída de `src/utils/losses.py`: $\mathcal{L}_{attention} = \frac{1}{N}\sum |\hat{y}_i-y_i|\cdot|\hat{y}_i/y_i - 1|$,
que se simplifica algebraicamente a un MSE ponderado por el inverso de la edad real (penaliza
~9× más fuerte los errores en pacientes jóvenes que en mayores). MAE train > MAE val explicado por
dos causas verificables en el código: dropout activo durante el cómputo de la métrica de
entrenamiento, y augmentación de datos solo en entrenamiento.

### #14: Comparación de arquitecturas no controlada
Hiperparámetros confirmados idénticos entre los 4 backbones (tabla completa: optimizador, LR,
early stopping, pesos iniciales, etc.). Tiempos de entrenamiento reales por backbone
(ResNet50~12h, VGG16~7h, DenseNet121~10h, InceptionV3~14h) y benchmark de latencia/memoria medido
en nodo GPU dedicado: DenseNet121 tiene los menos parámetros y la menor memoria, pero
paradójicamente la **mayor** latencia de inferencia (526.7ms, más lento que ResNet50 pese a tener
3.3× menos parámetros): hallazgo reportado tal cual, sin ocultar que el conteo de parámetros no
predice bien la latencia real.

### #15: Ablación insuficiente (falta baseline whole-hand)
Experimento nuevo (exp. 60, `MODEL_TYPE=whole_hand`): una sola rama DenseNet121 sobre la mano
completa sin segmentar, mismo split/protocolo/hiperparámetros que la fusión de referencia.
Resultado con significancia estadística pareada en las tres condiciones de evaluación:

| Evaluación | ΔMAE (whole-hand − fusión) | Significativo |
|---|---:|:---:|
| Interna (n=2,357) | +0.39m | No (p=0.11) |
| RSNA externa (n=1,393) | +0.86m | Sí (p=0.016) |
| MEX externa (n=99) | +7.67m | Sí, fuerte (p<0.0001) |

La segmentación anatómica aporta poco en validación interna pero se vuelve significativa y
sustancial bajo distribution shift: refuerza la tesis central del artículo sobre la brecha
interna/externa. Nota secundaria: el modelo whole-hand, sin depender de segmentación, procesó el
100% de ambos conjuntos externos, mientras la fusión pierde 32/1,425 (RSNA) y 1/100 (MEX) casos
por fallos de segmentación.

### #16: Mapas de saliencia sin reproducibilidad
Decisión explícita del autor: **no generar ejemplos nuevos**: la saliencia es puramente
ilustrativa. Metodología documentada con precisión (gradiente vainilla vía `tf.GradientTape`, sin
capa objetivo específica, normalización [0,1], umbral percentil 97, determinístico). Texto de
reemplazo refuerza explícitamente que no se afirma interpretabilidad clínica ni anatómica más
allá de que la atención del gradiente cae dentro de las regiones segmentadas: invocando
directamente la alternativa que el propio revisor ofrece ("...or withdraw interpretability
claims").

---

## Comentarios bloqueados o diferidos

- **#1 y #19** (limpieza general/editorial): bloqueados porque solo se dispuso del PDF exportado
  del manuscrito, nunca del `.docx` fuente: no es posible aceptar cambios de Word ni hacer
  búsqueda de texto completo de forma confiable sobre un PDF.
- **#5** (confiabilidad inter-observador): requiere que los dos lectores clínicos existentes
  (radiólogo y médico) califiquen independientemente un subconjunto y se calcule kappa/ICC:
  trabajo clínico real, no analítico.
- **#10** (segmentación en imágenes mexicanas): requiere anotación manual de ~20–30 imágenes
  mexicanas en LabelMe por una persona: el ítem más costoso en tiempo de todo el documento.
- **#20** (decisión general): es la síntesis de cierre, se redacta una vez resueltos o declarados
  como diferidos los puntos 1, 5, 10 y 19.

---

## Más allá de los 20 comentarios numerados

Durante la revisión del manuscrito ya editado (`corregido2.pdf`) se encontraron además ~16
correcciones puntuales (números con el bug viejo aún citados en Abstract/Conclusiones,
referencias cruzadas rotas tras la reestructuración, tablas/figuras con numeración sin resolver,
abreviaciones faltantes, un párrafo redundante, etc.): documentadas por separado, con ubicación
exacta y texto listo para pegar, en
[`RESULTADO_pendientes_corregido_pdf.md`](RESULTADO_pendientes_corregido_pdf.md).

## Inventario de archivos de este análisis

- `ANALISIS_COMENTARIOS.md`: mapeo inicial de los 20 comentarios (qué pide / qué tenemos / qué falta).
- `RESULTADO_comentario_{3,9,12,13,14,15,16}.md`, `RESULTADO_comentarios_{2_6_18,4_7_8,11_17}.md`: detalle completo por comentario, con texto de reemplazo listo para el manuscrito.
- `RESULTADO_texto_backbone_vs_simplecnn.md`: corrección de un párrafo específico señalado por el usuario (comparación backbone-vs-simple_cnn mal etiquetada como "fusion vs. baseline").
- `RESULTADO_pendientes_corregido_pdf.md`: checklist mecánico de 16 puntos sobre el manuscrito ya editado.
- `figures/`: diagrama de flujo, Figura 2 extendida (6 paneles), Figura 8 (identidad+regresión), Bland-Altman.
- `scripts/`: todo el código usado para generar los números y figuras de este análisis, reproducible desde los datos del proyecto.
- `results_json/`: resultados numéricos crudos (Dice/IoU, estadísticas extendidas, latencia, significancia whole-hand, etc.).
