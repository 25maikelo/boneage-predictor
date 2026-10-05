# Experimentos y análisis adicionales para el manuscrito

> **El artículo ya fue publicado.** Este documento queda como registro histórico de cierre de un
> conjunto de experimentos y análisis adicionales realizados sobre el manuscrito *A Multi-Segment
> Fusion Architecture for Bone Age Estimation* (revista *Technologies*, MDPI), en respuesta a los
> 20 comentarios de la segunda ronda de revisión. No describe trabajo pendiente: el ciclo de
> revisión ya cerró con la publicación, independientemente del estado en que haya quedado cada
> punto a continuación.
> Todos los números citados aquí están verificados contra el código y los datos del proyecto: no
> hay cifras inventadas. Material de respaldo (figuras, scripts, resultados numéricos crudos) en
> las subcarpetas de este mismo directorio, listadas al final.

## Estado por comentario del revisor

| # | Tema | Estado |
|---|---|---|
| 1 | Limpieza general / versión sin tracked changes | 🗄️ Cerrado sin aplicar: no se dispuso del `.docx` fuente |
| 2 | Diagrama de flujo de participantes | ✅ Resuelto |
| 3 | Distribución antes/después del balanceo + umbral de 50 | ✅ Resuelto |
| 4 | Terminología TW3 vs. edad cronológica (Figura 8) | ✅ Resuelto |
| 5 | Confiabilidad inter-observador (estándar mexicano) | 🗄️ Cerrado sin realizar: requería trabajo clínico real |
| 6 | Representatividad de la cohorte mexicana | ✅ Resuelto |
| 7 | Estadística externa incompleta (IC, RMSE, sesgo, estratificación) | ✅ Resuelto |
| 8 | Gráficos de dispersión + colapso de F-VGG16 | ✅ Resuelto |
| 9 | Validación de segmentación (split real + Dice/IoU por clase) | ✅ Resuelto (con limitación declarada) |
| 10 | Segmentación no probada en imágenes mexicanas | 🗄️ Cerrado sin realizar: requería trabajo manual real |
| 11 | Asociación TW3 exagerada | ✅ Ya resuelto en el manuscrito, solo faltaba la respuesta |
| 12 | Fuga de datos en el entrenamiento de fusión | ✅ Resuelto (declaración de limitación, sin reentrenar) |
| 13 | Función de pérdida sin definición matemática | ✅ Resuelto |
| 14 | Comparación de arquitecturas no controlada | ✅ Resuelto |
| 15 | Ablación insuficiente (falta baseline whole-hand) | ✅ Resuelto: experimento nuevo, datos reales |
| 16 | Mapas de saliencia sin reproducibilidad | ✅ Resuelto (reencuadrado como ilustrativo) |
| 17 | Tabla 10 no comparable con la literatura | ✅ Ya resuelto en el manuscrito, solo faltaba la respuesta |
| 18 | Declaración de ética inaceptable | ✅ Resuelto |
| 19 | Limpieza editorial completa | 🗄️ Cerrado sin aplicar: no se dispuso del `.docx` fuente |
| 20 | Decisión general | 🗄️ Cerrado por publicación (ver nota arriba) |

---

## Hallazgos críticos encontrados durante el análisis

**Hallazgo #1: Bug `real_age` vs. `bone_age`.** `src/08_mex_validation.py` comparaba las
predicciones del modelo (que predice edad ósea TW3) contra `real_age` (edad cronológica) en vez
de `bone_age` (edad ósea asignada por TW3), pese a usar `bone_age` correctamente en otras partes
del mismo script. Corregido en código (commit `9ccd9a6`) y **todos** los números de validación
mexicana citados en este documento ya están recalculados con la corrección. Efecto principal:
**F-DenseNet121 pasa a ser el backbone con el MAE numéricamente más bajo en MEX** (16.38 vs. el
17.28 de F-InceptionV3, que antes figuraba como mejor), aunque la diferencia entre ambos sigue
sin ser estadísticamente significativa.

**Hallazgo #2: Respuestas 13/14 cruzadas** en el borrador de respuestas al revisor (la
Respuesta 14 contestaba al Comentario 13 por error, dejando el 14 efectivamente sin responder).
Corregido al redactar las respuestas de este análisis.

**Hallazgo #3: Artefacto de fusión de texto**: *"inspired based by theon the Tanner–Whitehouse 3
(TW3) technique"* (pág. 3 del PDF revisado en ese momento), evidencia de que la limpieza
editorial del Comentario 19 estaba incompleta en ese borrador. No se verificó si llegó a
corregirse en el texto final publicado (no se dispuso del `.docx` fuente para aplicarlo
directamente); dado que el artículo ya está publicado, queda como nota histórica, no como
pendiente.

---

## Comentarios resueltos: detalle

### #2: Diagrama de flujo de participantes
Se construyó un diagrama de flujo (`figures/participant_flow_diagram.png`) que usa exactamente
la terminología del manuscrito ("official training/validation/test set", "balanced subset",
"internal training/validation subset", "Mexican clinical dataset"), sin conectar visualmente
conceptos no relacionados (se corrigió una flecha que implicaba una relación falsa entre el test
set oficial de RSNA, no usado, y el cohorte mexicano).

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
mientras InceptionV3 subestima levemente (−2.37m), algo invisible en el MAE agregado. Nueva
Figura 8 (identidad + regresión + IC 95%) y figura Bland-Altman generadas con terminología TW3
corregida. Diagnóstico del colapso de F-VGG16 con evidencia real de `training_history`: no es
falta de entrenamiento (estancado desde la época 1) ni fuga de datos (produciría desempeño
artificialmente bueno, no un colapso), sino falla de optimización específica de la arquitectura:
VGG16 es la única de las 4 sin BatchNorm ni conexiones residuales/densas al entrenar desde cero.

### #6, #18: Representatividad mexicana y declaración de ética
Terminología "single-center Mexican clinical cohort" ya adoptada en el manuscrito. Limitación de
datos clínicos (solo edad/sexo disponibles) declarada explícitamente. La sección formal
"Institutional Review Board Statement" se sincronizó con el detalle que ya existía en la Sección
2.1 (comité UDG, expediente 26-110, acuerdo RG/ACC/75/2019).

### #9: Validación de segmentación
El manuscrito decía "200 radiografías anotadas"; el dataset real tiene **379** (303 train / 76
val, split reproducible `random_state=42`), confirmado contra el log de entrenamiento del modelo
activo (`hand-detector_00`). Dice/IoU por clase calculado sobre el split real: meñique 0.78,
medio 0.89, pulgar 0.86, muñeca 0.77 (macro 0.83), notablemente más bajo y heterogéneo que el
0.9168 agregado que reporta el manuscrito (calculado en modo "soft"/probabilístico, no por clase).
**Limitación honesta declarada**: las 76 imágenes de "validación" no son un test ciego, son el
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
implementado y revertido una corrección completa, un split de 3 vías, en una sesión anterior). Se
optó por declarar la limitación honestamente en vez de rehacer el pipeline. Hallazgo adicional
encontrado en el camino: el checkpoint guardado en disco es el de la última época, no
necesariamente el de mejor `val_loss`, pese a `restore_best_weights=True`; aplica a todos los
modelos del pipeline, no solo a la fusión.

### #13: Función de pérdida sin definición matemática
Ecuación formal extraída de `src/utils/losses.py`: $\mathcal{L}_{attention} = \frac{1}{N}\sum |\hat{y}_i-y_i|\cdot|\hat{y}_i/y_i - 1|$,
que se simplifica algebraicamente a un MSE ponderado por el inverso de la edad real (penaliza
~9× más fuerte los errores en pacientes jóvenes que en mayores). MAE train > MAE val explicado por
dos causas verificables en el código: dropout activo durante el cómputo de la métrica de
entrenamiento, y augmentación de datos solo en entrenamiento.

### #14: Comparación de arquitecturas no controlada
Hiperparámetros confirmados idénticos entre los 4 backbones (optimizador, LR, early stopping,
pesos iniciales, etc.). Tiempos de entrenamiento reales por backbone (ResNet50~12h, VGG16~7h,
DenseNet121~10h, InceptionV3~14h) y benchmark de latencia/memoria medido en nodo GPU dedicado:
DenseNet121 tiene los menos parámetros y la menor memoria, pero paradójicamente la **mayor**
latencia de inferencia (526.7ms, más lento que ResNet50 pese a tener 3.3× menos parámetros),
hallazgo reportado tal cual, sin ocultar que el conteo de parámetros no predice bien la latencia
real.

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
sustancial bajo distribution shift, reforzando la tesis central del artículo sobre la brecha
interna/externa. Nota secundaria: el modelo whole-hand, sin depender de segmentación, procesó el
100% de ambos conjuntos externos, mientras la fusión pierde 32/1,425 (RSNA) y 1/100 (MEX) casos
por fallos de segmentación.

### #16: Mapas de saliencia sin reproducibilidad
Decisión explícita del autor: **no generar ejemplos nuevos**, la saliencia es puramente
ilustrativa. Metodología documentada con precisión (gradiente vainilla vía `tf.GradientTape`, sin
capa objetivo específica, normalización [0,1], umbral percentil 97, determinístico). Texto de
reemplazo refuerza explícitamente que no se afirma interpretabilidad clínica ni anatómica más
allá de que la atención del gradiente cae dentro de las regiones segmentadas, invocando
directamente la alternativa que el propio revisor ofrece ("...or withdraw interpretability
claims").

---

## Comentarios que quedaron cerrados sin aplicar

Con el artículo ya publicado, estos puntos ya no son trabajo pendiente; se documentan solo para
dejar registro de por qué no se llegaron a aplicar durante el análisis:

- **#1 y #19** (limpieza general/editorial): no se dispuso del `.docx` fuente del manuscrito,
  solo del PDF exportado; no era posible aceptar cambios de Word ni hacer búsqueda de texto
  completo de forma confiable sobre un PDF.
- **#5** (confiabilidad inter-observador): hubiera requerido que los dos lectores clínicos
  existentes (radiólogo y médico) califiquen independientemente un subconjunto y se calcule
  kappa/ICC, trabajo clínico real, no analítico.
- **#10** (segmentación en imágenes mexicanas): hubiera requerido anotación manual de ~20–30
  imágenes mexicanas en LabelMe por una persona, el ítem más costoso en tiempo de todo este
  análisis.
- **#20** (decisión general): era la síntesis de cierre que se redactaría una vez resueltos los
  puntos 1, 5, 10 y 19; la publicación del artículo cierra el ciclo sin necesidad de esa síntesis.

---

## Correcciones de texto identificadas sobre el manuscrito editado (histórico)

Al revisar una versión del manuscrito ya editada con los cambios de arriba se identificaron 16
correcciones puntuales adicionales, con ubicación exacta y texto listo para insertar. El artículo
ya se publicó, así que esta lista no es trabajo pendiente; queda como registro de lo que se
detectó en ese momento, sin confirmación de si cada punto llegó a aplicarse en la versión final:

1. **Abstract**: todavía cita a F-InceptionV3 como el de mejor desempeño externo; corregir a
   "statistically indistinguishable" entre F-DenseNet121 y F-InceptionV3.
2. **Conclusiones**: mismo número viejo (15.98 meses) a corregir, más un párrafo nuevo de
   limitaciones metodológicas (fuga de fusión, semilla única, split de segmentación no ciego).
3. **Institutional Review Board Statement**: sincronizar con el detalle ya presente en la Sección
   2.1 (comité UDG, expediente 26-110, acuerdo RG/ACC/75/2019).
4. **Artefacto de texto** ("inspired based by theon the") en página 3: limpiar a "inspired by the".
5. **Participant Flow Diagram**: falta insertar la imagen y el párrafo que la introduce (texto ya
   redactado, usando exactamente la terminología del manuscrito).
6. **Nueva sección "Limitations"**: no existe actualmente ninguna sección de limitaciones en el
   manuscrito; se redactó el contenido completo (fuga de fusión, checkpoint, semilla única, split
   de segmentación no ciego) listo para insertar como nueva subsección.
7. **Figura Bland-Altman**: falta insertarla junto con su párrafo de discusión.
8. **Diagnóstico de F-VGG16**: falta el párrafo con la evidencia real de `training_history` que
   explica el colapso (ya redactado).
9-10. **Referencias cruzadas rotas** tras la reestructuración de secciones (una sección
   inexistente "3.X", una subsección "3.4.1" que ya no existe).
11-12. **Renumeración mecánica** de 16 tablas y 10 figuras, resultado de insertar las tablas y
   figuras nuevas de los comentarios de arriba (tabla de mapeo completa disponible en el historial
   de este análisis).
13. **Abreviaciones faltantes**: 11 términos usados en el cuerpo del texto (IoU, CLAHE, SD, CI,
   GPU, HPC, DR, UDG, MAD, MLP, LR) nunca definidos en la lista de abreviaciones.
14. **Pie de la figura de saliencia**: genérico ("Examples of bone age prediction"), no describe
   que son mapas de saliencia; reescrito para reflejar el contenido real.
15. **Párrafo redundante** en la Sección 3.3 que menciona las Tablas 14/15 antes de que aparezca
   la Tabla 13 en el texto; se identificó como eliminable sin pérdida de información (todo su
   contenido está duplicado, con más precisión, en la Sección 3.5).
16. **Typo** ("reports additional reports additional") y **leyenda faltante** para los símbolos
   ✅/❌ de significancia estadística en la tabla de pruebas pareadas.

## Material de respaldo

- [`figures/`](figures/): diagrama de flujo de participantes, Figura 2 extendida (6 paneles), Figura 8 (identidad+regresión), Bland-Altman.
- [`scripts/`](scripts/): todo el código usado para generar los números y figuras de este análisis, reproducible desde los datos del proyecto.
- [`results_json/`](results_json/): resultados numéricos crudos (Dice/IoU por clase, estadísticas extendidas RSNA/MEX, latencia/memoria, significancia del baseline whole-hand, diagnóstico de fallos de validación).
