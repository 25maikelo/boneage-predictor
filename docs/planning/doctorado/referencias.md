# Referencias: glosario por técnica/metodología

> Todas las referencias bibliográficas citadas en `Tesis.pdf` (maestría) y `Protocolo.pdf`
> (doctorado), los dos documentos que el usuario compartió en la conversación. Cada entrada
> tiene cita completa, un resumen breve de qué propone/aporta, y etiquetas de tema para buscar
> por técnica (`Ctrl+F` o `grep` por `#tag` o por apellido del autor). No son resúmenes de haber
> leído el artículo completo: son los resúmenes que los propios documentos (tesis/protocolo)
> dan de cada referencia, verificados contra el texto citante, no contra el artículo original.

## Índice por tema

| Tema | Referencias |
|---|---|
| TW3 / Greulich-Pyle (métodos clínicos de edad ósea) | [T2](#t2), [T5](#t5), [P3](#p3), [P6](#p6) |
| Dataset RSNA / benchmarks | [T3](#t3) |
| Segmentación de mano / extracción de ROIs | [T5](#t5), [T7](#t7), [T9](#t9--p2), [P4](#p4) |
| Preprocesamiento (contraste, ruido, normalización) | [T6](#t6), [T7](#t7) |
| Arquitecturas CNN (comparativas, backbones) | [T6](#t6), [T8](#t8), [T9](#t9--p2), [T12](#t12) |
| Sesgo / generalización / validación externa | [T4](#t4) |
| Validación clínica / en población específica | [P3](#p3), [P6](#p6) |
| Survey / estado del arte | [P5](#p5) |
| Marco teórico general (edad ósea, pediatría) | [T1](#t1), [P1](#p1) |
| Trabajo previo propio (autorreferencia) | [P7](#p7) |

---

## Referencias citadas en `Tesis.pdf` (maestría)

### T1
Gilsanz V., Ratib O. (2005). *Hand Bone Age: A Digital Atlas of Skeletal Maturity*. Springer
Science & Business Media.

Resumen: atlas de referencia de madurez ósea de mano (plantillas Greulich-Pyle). Es la base
conceptual de qué es "edad ósea", no un método computacional.

`#marco-teorico` `#atlas` `#edad-osea` `#greulich-pyle`

### T2
Aja-Fernández S., De Luis-García R., Martín-Fernández M.Á., Alberola-López C. (2004). "A
computational TW3 classifier for skeletal maturity assessment. A Computing with Words
approach." *Journal of Biomedical Informatics* 37(2), 99-107.
https://doi.org/10.1016/j.jbi.2004.01.002

Resumen: propone un clasificador computacional para TW3 usando lógica difusa ("computación con
palabras") para automatizar la asignación de etapas de madurez ósea a partir de las
descripciones en lenguaje natural del método original.

`#TW3` `#clasificador` `#logica-difusa`

### T3
Halabi S.S., Prevedello L.M., Kalpathy-Cramer J., et al. (2019). "The RSNA Pediatric Bone Age
Machine Learning Challenge." *Radiology* 290(2), 498-503.
https://doi.org/10.1148/radiol.2018180736

Resumen: define el reto RSNA Pediatric Bone Age Challenge (2017) y publica el dataset usado en
todo este proyecto (14,236 radiografías etiquetadas por Greulich-Pyle; 12,611 entrenamiento /
1,425 validación / 200 prueba). Mejor desempeño del reto: MAD = 4.2 meses, con arquitecturas
tipo InceptionV3/ResNet50, género como entrada, aumentación de datos y ensamblado de modelos.

`#dataset` `#RSNA` `#benchmark` `#challenge`

### T4
Beheshtian E., Putman K., Santomartino S.M., Parekh V.S., Yi P.H. (2023). "Generalizability and
Bias in a Deep Learning Pediatric Bone Age Prediction Model Using Hand Radiographs." *Radiology*
306(2). https://doi.org/10.1148/radiol.220505

Resumen: evalúa el modelo ganador "16Bit" del reto RSNA sobre un conjunto de prueba externo y
encuentra errores clínicamente significativos en 16% de las radiografías, con sesgo de género
(mayor MAD en niñas) y variación de desempeño según madurez sexual y edad cronológica. Evidencia
directa de por qué la validación externa/poblacional (objetivo F del doctorado) no es opcional.

`#sesgo` `#generalizacion` `#validacion-externa` `#genero`

### T5
Son S.J., Song Y., Kim N., Do Y., Kwak N., Lee M.S., Lee B. (2019). "TW3-Based Fully Automated
bone age assessment System using deep neural networks." *IEEE Access* 7, 33346-33358.
https://doi.org/10.1109/access.2019.2903131

Resumen: sistema de 3 etapas: (1) extracción de bROIs (muñeca, pulgar, medio, meñique) mediante
rotación de alineación vertical de la muñeca, (2) Faster R-CNN para proponer y recortar las 13
ROIs reales dentro de cada bROI, (3) VGGNet-BA puntúa la madurez de cada ROI y la suma se
traduce en edad vía matriz de correlación. Precisión 97.6% dentro de 1 año. Es la referencia
metodológica más cercana al enfoque de 4 segmentos (meñique/medio/pulgar/muñeca) de este
proyecto: aquí se agrupan las 13 ROIs en 4 regiones en vez de 4 bROIs intermedios.

`#TW3` `#segmentacion` `#Faster-RCNN` `#VGGNet` `#bROI`

### T6
Chen X., Li J., Zhang Y., Lu Y., Liu S. (2019). "Automatic feature extraction in X-ray image
based on deep learning approach for determination of bone age." *Future Generation Computer
Systems* 110, 795-801. https://doi.org/10.1016/j.future.2019.10.032

Resumen: compara SVM con fusión de características multidimensionales contra una CNN profunda;
el preprocesamiento incluye normalización, reducción de ruido y ajuste de contraste para
homogeneizar la distribución de grises entre radiografías. La CNN profunda supera al enfoque
SVM tradicional.

`#preprocesamiento` `#SVM` `#comparacion-arquitecturas`

### T7
Li S., Liu B., Li S. et al. (2022). "A deep learning-based computer-aided diagnosis method of
X-ray images for bone age assessment." *Complex Intelligent Systems* 8, 1929-1939.
https://doi.org/10.1007/s40747-021-00376-z

Resumen: localización de la mano con aprendizaje no supervisado (segmentación por agrupamiento
de características diferenciables), preprocesamiento con eliminación de fondo + recorte por
conectividad + ecualización de histograma, y predicción con MobileNetV3 + MLP con el género
embebido como vector adicional. MAE = 6.2 meses sobre RSNA.

`#segmentacion-no-supervisada` `#MobileNetV3` `#preprocesamiento` `#ecualizacion`

### T8
He J., Jiang D. (2021). "Fully Automatic Model Based on SE-ResNet for Bone Age Assessment."
*IEEE Access* 9, 62460-62466. doi: 10.1109/ACCESS.2021.3074713

Resumen: SE-ResNet (módulo de compresión y excitación) que recorta áreas no informativas de la
radiografía antes de extraer características, con una función de pérdida que incorpora un
término de peso gaussiano para el desbalance de la distribución de edades. Mejora relativa de
5.96 meses en MAE respecto a kNN/SVM, y supera a U-Net/VGG como arquitectura base.

`#SE-ResNet` `#atencion-de-canal` `#balance-de-clases` `#loss-ponderada`

### T9 / P2
Liu Z., Hu Z., Wu T., Ye G., Tang Y., Zeng Z., Ouyang Z., Li Y. (2023). "Bone age recognition
based on mask R-CNN using Xception regression model." *Frontiers in Physiology* 14.
https://doi.org/10.3389/fphys.2023.1062034

Resumen: usa Mask R-CNN para segmentación de regiones óseas y un modelo de regresión Xception
para estimar la edad. MAE reportado entre 5.44 y 8.18 meses según configuración. **Única
referencia citada en ambos documentos** (tabla comparativa de la tesis y marco teórico del
protocolo).

`#Mask-RCNN` `#Xception` `#segmentacion`

### T10
Lee H., Tajmir S., Lee J., Zissen M., Yeshiwas B.A., Alkasab T.K., Choy G., Do S. (2017). "Fully
Automated Deep learning system for bone age assessment." *Journal of Digital Imaging* 30(4),
427-441. https://doi.org/10.1007/s10278-017-9955-8

Resumen: uno de los sistemas pioneros completamente automatizados (de extremo a extremo) de
estimación de edad ósea con deep learning.

`#end-to-end` `#pionero`

### T11
Spampinato C., Palazzo S., Giordano D., Aldinucci M., Leonardi R. (2016). "Deep learning for
automated skeletal bone age assessment in X-ray images." *Medical Image Analysis* 36, 41-51.
https://doi.org/10.1016/j.media.2016.10.010

Resumen: trabajo temprano aplicando deep learning directamente sobre radiografías completas para
BAA, sin ingeniería de características manual previa.

`#pionero` `#medical-image-analysis`

### T12
Hamal S., Mishra B.K., Baldock R., Sayers W., Adhikari T.N., Gibson R.M. (2024). "A comparative
analysis of machine learning algorithms for detecting COVID-19 using lung X-ray images."
*Decision Analytics Journal* 11, 100460. https://doi.org/10.1016/j.dajour.2024.100460

Resumen: **no es sobre edad ósea.** Comparación de arquitecturas (VGG, InceptionV3, DenseNet,
ResNet, entre otras) para clasificar COVID-19 en radiografías de tórax. Citado en la tesis
únicamente como fuente de las descripciones de fortalezas/debilidades de VGG16, InceptionV3 y
DenseNet en el marco teórico (sección de arquitecturas), no por el dominio clínico.

`#arquitecturas-CNN` `#comparativa` `#fuera-de-dominio`

---

## Referencias citadas en `Protocolo.pdf` (doctorado)

### P1
Cavallo F., Mohn A., Chiarelli F., Giannini C. (2021). "Evaluation of bone age in children: A
mini-review." *Frontiers in Pediatrics* 9. doi: 10.3389/fped.2021.580314

Resumen: revisión breve de por qué y cómo se evalúa la edad ósea en pediatría (crecimiento,
trastornos endocrinos, pubertad precoz/retrasada). Fundamenta la introducción del protocolo
doctoral.

`#marco-teorico` `#pediatria` `#mini-review`

### P2
Ver [T9 / P2](#t9--p2) arriba: mismo artículo citado en ambos documentos (Liu et al. 2023, Mask
R-CNN + Xception).

### P3
Martín Pérez S.E. et al. (2025). "The validation of the Tanner-Whitehouse 3 method for
radiological bone assessments in a pediatric population from the Canary Islands." *Osteology*
5(1). doi: 10.3390/osteology5010006

Resumen: validación clínica del método TW3 en una población pediátrica geográficamente
específica (Islas Canarias). **Precedente metodológico directo** para el objetivo F del
doctorado (validación en población mexicana): mismo tipo de pregunta (¿TW3 generaliza a esta
población?), aplicado a otra población distinta de la original.

`#TW3` `#validacion-poblacional` `#precedente-metodologico`

### P4
Li Z. et al. (2023). "Bone age assessment based on deep neural networks with annotation-free
cascaded critical bone region extraction." *Frontiers in Artificial Intelligence* 6.
doi: 10.3389/frai.2023.BBACEFD

Resumen: extracción en cascada de regiones óseas críticas **sin necesidad de anotación manual**
(annotation-free). Alternativa potencial a la extracción manual de ROIs usada en este proyecto
(200 imágenes etiquetadas a mano con LabelMe para entrenar la U-Net de segmentación);
relevante si se busca reducir la dependencia de anotación manual en la ampliación del dataset
mexicano (objetivo F).

`#segmentacion` `#annotation-free` `#ROI`

### P5
Nadeem M.W. et al. (2020). "Bone age assessment empowered with deep learning: A survey, open
research challenges and future directions." *Diagnostics* 10(10). doi: 10.3390/diagnostics10100781

Resumen: survey del estado del arte de BAA con deep learning, con mapeo de desafíos abiertos y
direcciones futuras. Punto de partida natural para ampliar la revisión de literatura del
doctorado (Requirement 1 de `requirements.md`) más allá de lo ya cubierto en la tesis.

`#survey` `#estado-del-arte` `#desafios-abiertos`

### P6
Shin N.-Y. et al. (2020). "Evaluation of the clinical efficacy of a TW3-based fully automated
bone age assessment system using deep neural networks." *Imaging Science in Dentistry* 50(3).
doi: 10.5624/isd.2020.50.3.237

Resumen: evaluación de la **eficacia clínica** (no solo métricas offline) de un sistema TW3
automatizado. Aporta un precedente de cómo medir utilidad clínica real, relevante para los
objetivos F/G del doctorado (comparación contra evaluación de especialistas).

`#TW3` `#validacion-clinica` `#eficacia`

### P7
Lozano López M.Á. (2025). *Propuesta de un sistema automatizado para la estimación de la edad a
partir de radiografías de mano*. Tesis de maestría no publicada, Universidad de Guadalajara.

Resumen: autorreferencia. Es el propio `Tesis.pdf` de este proyecto: el sistema previo que el
doctorado analiza, corrige y extiende (ver `docs/planning/doctorado/README.md`, sección
"Contexto en una página").

`#autorreferencia` `#tesis-maestria` `#sistema-previo`
