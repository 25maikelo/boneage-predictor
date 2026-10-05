# Script 00: Descarga del Dataset

## Descripción

`src/preprocessing/00_download_dataset.py` descarga el dataset RSNA Bone Age (`kmader/rsna-bone-age`) desde Kaggle usando `kagglehub`, y copia todas las imágenes PNG al directorio del proyecto.

### Uso

```bash
python src/preprocessing/00_download_dataset.py
```

Requiere credenciales de Kaggle configuradas (`~/.kaggle/kaggle.json` o variables de entorno `KAGGLE_USERNAME`/`KAGGLE_KEY`).

### Entrada / Salida

| | Ruta |
|---|---|
| Entrada | Dataset remoto `kmader/rsna-bone-age` (Kaggle) |
| Salida | `RAW_IMAGES_DIR` (`config/paths.py`), 13,014 imágenes PNG |

### Notas

- Si una imagen con el mismo nombre ya existe en el destino, se omite (permite re-ejecutar sin duplicar).
- No aplica ningún preprocesamiento; solo descarga y copia. La limpieza manual de calidad (13 imágenes eliminadas, 190 volteadas) se hace fuera de este script, antes del paso 02.
