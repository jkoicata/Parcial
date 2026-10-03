# Detector de fallas de GPU

Proyecto de aprendizaje automático para clasificar el estado de una GPU a partir de ventanas de telemetría. El entrenamiento valida los datos con Pandera, extrae estadísticas por ventanas de 30 segundos y entrena un `RandomForestClassifier`. La API REST de FastAPI recibe lecturas recientes y devuelve el estado estimado y su confianza.

## Estados detectados

- `normal`
- `sobrecalentamiento`
- `degradacion_memoria`
- `falla_alimentacion`

## Requisitos

- Python 3.12 o posterior.
- Docker (opcional, para ejecutar el servicio en un contenedor).
- El archivo `telemetria_publica.csv` para entrenar. Debe contener las columnas que define `src/schema.py`. Por defecto, el script busca el archivo en `data/telemetria_publica.csv` dentro del proyecto o en `data/telemetria_publica.csv` en la carpeta padre del proyecto.

## Entrenamiento del modelo

Desde la carpeta `detector-fallas-gpu`:

```bash
python -m venv .venv
```

Activa el entorno virtual y después instala las dependencias:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
# Linux o macOS
source .venv/bin/activate
pip install -r requirements.txt
```

Entrena el modelo:

```bash
python src/train.py
```

El script valida el CSV, agrupa las lecturas por episodio, las divide en ventanas de 30 segundos, calcula las características y realiza una división de entrenamiento y prueba. Al finalizar, muestra el accuracy y guarda el pipeline serializado en `models/modelo.joblib`.

Si el CSV está en otra ubicación, indícala explícitamente:

```bash
python src/train.py --csv ruta/al/telemetria_publica.csv
```

## Ejecución local

Después de entrenar el modelo, inicia la API desde la carpeta del proyecto:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

La API estará disponible en `http://localhost:8000`. La documentación interactiva de Swagger se encuentra en `http://localhost:8000/docs`.

## Construcción con Docker

Entrena primero el modelo para que `models/modelo.joblib` contenga un artefacto válido. Desde la carpeta `detector-fallas-gpu`, construye la imagen:

```bash
docker build -t detector-fallas-gpu .
```

## Ejecución con Docker

```bash
docker run --rm -p 8000:8000 detector-fallas-gpu
```

La API quedará disponible en `http://localhost:8000`; Swagger, en `http://localhost:8000/docs`.

## Predicción

El endpoint `POST /predecir` requiere al menos diez lecturas. Cada una debe proporcionar `temp_c`, `power_w`, `util_pct`, `clock_mhz` y `ecc_errors` con valores numéricos; `ecc_errors` debe ser entero. Ejemplo con diez lecturas:

```bash
curl -X POST "http://localhost:8000/predecir" \
	-H "Content-Type: application/json" \
	-d '{
		"lecturas": [
			{"temp_c": 56.2, "power_w": 234.5, "util_pct": 76.2, "clock_mhz": 2412.0, "ecc_errors": 0},
			{"temp_c": 57.1, "power_w": 236.0, "util_pct": 77.0, "clock_mhz": 2408.0, "ecc_errors": 0},
			{"temp_c": 58.0, "power_w": 238.1, "util_pct": 78.2, "clock_mhz": 2401.0, "ecc_errors": 0},
			{"temp_c": 59.3, "power_w": 240.4, "util_pct": 79.5, "clock_mhz": 2398.0, "ecc_errors": 0},
			{"temp_c": 60.0, "power_w": 242.0, "util_pct": 80.1, "clock_mhz": 2392.0, "ecc_errors": 0},
			{"temp_c": 61.2, "power_w": 244.3, "util_pct": 81.0, "clock_mhz": 2389.0, "ecc_errors": 0},
			{"temp_c": 62.4, "power_w": 246.5, "util_pct": 82.4, "clock_mhz": 2381.0, "ecc_errors": 0},
			{"temp_c": 63.0, "power_w": 248.0, "util_pct": 83.0, "clock_mhz": 2378.0, "ecc_errors": 0},
			{"temp_c": 64.1, "power_w": 250.2, "util_pct": 84.3, "clock_mhz": 2370.0, "ecc_errors": 0},
			{"temp_c": 65.0, "power_w": 252.1, "util_pct": 85.0, "clock_mhz": 2365.0, "ecc_errors": 0}
		]
	}'
```

Respuesta de ejemplo (la confianza depende del modelo entrenado):

```json
{
	"estado_predicho": "normal",
	"confianza": 0.95
}
```

Las solicitudes con campos faltantes, tipos incorrectos o menos de diez lecturas reciben un error de validación HTTP 422.

## Estructura del proyecto

```text
detector-fallas-gpu/
├── src/
│   ├── api.py          # API FastAPI y endpoint de predicción
│   ├── features.py     # Extracción de estadísticas por ventana
│   ├── schema.py       # Esquema Pandera para los datos de entrenamiento
│   └── train.py        # Validación, entrenamiento y serialización
├── models/
│   └── modelo.joblib   # Pipeline entrenado (generado por train.py)
├── Dockerfile          # Imagen y comando de inicio del servicio
├── .dockerignore       # Archivos excluidos del contexto Docker
├── requirements.txt    # Dependencias Python
└── README.md           # Documentación del proyecto
```
# Extracción de Características (Features)

El modelo no trabaja directamente con cada lectura de telemetría. En lugar de utilizar cientos de registros individuales, las lecturas se agrupan en ventanas de 30 segundos y se resumen mediante características estadísticas. Esta estrategia permite capturar el comportamiento general de la GPU durante un período de tiempo y reducir el ruido presente en mediciones aisladas.

La función extraer_caracteristicas() transforma cada ventana en una única fila que posteriormente será utilizada por el modelo Random Forest.

## Temperatura (temp_c)

Para cada ventana se calculan:

- Media (mean)
- Desviación estándar (std)
- Mínimo (min)
- Máximo (max)

¿Por qué son útiles?

Las fallas por sobrecalentamiento suelen reflejarse en temperaturas persistentemente elevadas. Mientras que la temperatura máxima identifica los momentos más críticos, la temperatura media permite determinar si el calentamiento se mantiene durante toda la ventana.

## Potencia Consumida (power_w)

Para esta variable se calculan:

- Media (mean)
- Desviación estándar (std)
- Mínimo (min)
- Máximo (max)
- Rango (max - min)

¿Por qué son útiles?

Las fallas de alimentación suelen provocar cambios abruptos en la potencia suministrada a la GPU. Por esta razón, además del promedio, se mide la variabilidad de la señal.

## Utilización de GPU (util_pct)

Para esta métrica se calculan:

- Media (mean)
- Desviación estándar (std)
- Mínimo (min)
- Máximo (max)

## Frecuencia del Reloj (clock_mhz)

Para esta variable se calculan:

- Media (mean)
- Desviación estándar (std)
- Mínimo (min)
- Máximo (max)

## Errores ECC (ecc_errors)

Para esta variable se calculan:

- Suma (sum)
- Media (mean)
- Máximo (max)

## ¿Por qué utilizar ventanas de 30 segundos?

Analizar una sola lectura podría resultar engañoso debido al ruido o variaciones momentáneas de la señal. Al agrupar múltiples observaciones en una ventana de tiempo se obtienen características más representativas del comportamiento real de la GPU.

## Relación entre Features y Tipos de Falla

Sobrecalentamiento:
- temp_c_mean
- temp_c_max
- temp_c_std

Degradación de Memoria:
- ecc_errors_suma
- ecc_errors_mean
- ecc_errors_max

Falla de Alimentación:
- power_w_std
- power_w_rango
- clock_mhz_std

Estado Normal:
- Temperaturas moderadas.
- Baja variabilidad de potencia.
- Frecuencias estables.
- Ausencia de errores ECC.

## Justificación Técnica

La selección de estas características permite transformar señales temporales complejas en un conjunto reducido de variables numéricas que representan el comportamiento general de la GPU. En lugar de entrenar el modelo con todas las lecturas originales, se utilizan estadísticas descriptivas que resumen cada ventana de telemetría.

