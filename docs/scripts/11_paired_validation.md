# Script 11: Validación Pareada / Ablación Estadística

## Descripción

`src/11_paired_validation.py` compara dos o más experimentos **sobre los mismos pacientes** (intersección de IDs en `plot_data.json`), aplicando pruebas estadísticas pareadas sobre el error absoluto por imagen. A diferencia de comparar MAE agregados (scripts 07/08/09), esto controla por dificultad del caso individual: cada paciente se compara contra sí mismo entre arquitecturas.

### Uso

```bash
python src/11_paired_validation.py --experiments 23 26 27 28
python src/11_paired_validation.py --experiments 23 26 27 28 --source mex
python src/11_paired_validation.py --experiments 23 26 27 28 --test bootstrap
python src/11_paired_validation.py --experiments 23 26 27 28 --age-bins 0 72 144 228
python src/11_paired_validation.py --experiments 23 26 27 28 --output results/paired_23_28.json
```

| Argumento | Default | Descripción |
|---|---|---|
| `--experiments` | (requerido) | Lista de IDs de experimento a comparar entre sí (todas las combinaciones por pares) |
| `--base-dir` | `experiments` | Directorio raíz de experimentos |
| `--source` | `rsna` | Dataset a usar: `rsna` (`validation/plot_data.json`) o `mex` (`mex-validation/plot_data.json`) |
| `--test` | `both` | `wilcoxon`, `bootstrap`, o `both` |
| `--n-bootstrap` | `10000` | Remuestreos para el bootstrap pareado |
| `--age-bins` | `0 72 144 228` | Límites de grupos de edad (meses) para la tabla de ablación por rango |
| `--alpha` | `0.05` | Nivel de significancia, antes de corrección de Holm |
| `--output` | ninguno | Ruta opcional para guardar el JSON de resultados |

### Entrada / Salida

| | Ruta |
|---|---|
| Entrada | `experiments/<id>/{validation,mex-validation}/plot_data.json` de cada experimento listado (requiere el campo `scatter.ids`, ver [script 07](07_validation.md)) |
| Salida | Tabla impresa en consola; JSON opcional si se usa `--output` |

### Métodos estadísticos

1. **Intersección de IDs** (`paired_errors`): solo se comparan pacientes presentes en ambos experimentos.
2. **Wilcoxon signed-rank** (`wilcoxon_test`, `zero_method="wilcox"`) sobre la diferencia de error absoluto pareado.
3. **Bootstrap pareado** (`bootstrap_paired_test`): remuestrea con reemplazo los pares (no cada grupo por separado) para estimar el IC de la diferencia de MAE.
4. **Corrección de Holm** (`holm_correction`): ajusta los p-valores cuando se comparan más de 2 experimentos (todas las combinaciones por pares), controlando el error tipo I acumulado.
5. **Rank-biserial correlation** (`rank_biserial`): tamaño de efecto no paramétrico acompañando a Wilcoxon.

### Notas

- Es el script usado para las comparaciones pareadas de `docs/results/ablacion_backbones/` y de `docs/results/experimentos_adicionales/analisis.md` (comentarios #4/#7/#8 del revisor).
- Si algún experimento no tiene `scatter.ids` en su `plot_data.json` (generado por versiones antiguas del script 07), hay que re-ejecutar 07/08 para ese experimento antes de poder emparejarlo.
