"""API FastAPI para predecir fallas a partir de telemetría de GPU."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Union

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, StrictFloat, StrictInt

try:
	from .features import extraer_caracteristicas
except ImportError:  # Permite ejecutar también ``python src/api.py``.
	from features import extraer_caracteristicas


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "modelo.joblib"
Numero = Union[StrictInt, StrictFloat]


class Lectura(BaseModel):
	"""Lectura individual con todos los valores numéricos requeridos."""

	temp_c: Numero
	power_w: Numero
	util_pct: Numero
	clock_mhz: Numero
	ecc_errors: StrictInt

	class Config:
		extra = "forbid"


class SolicitudPrediccion(BaseModel):
	"""Solicitud de predicción; requiere al menos diez lecturas."""

	lecturas: list[Lectura] = Field(..., min_items=10)

	class Config:
		extra = "forbid"


@asynccontextmanager
async def lifespan(app: FastAPI):
	"""Carga el modelo una vez al iniciar la aplicación."""
	if not MODEL_PATH.is_file():
		raise RuntimeError(
			f"No se encontró el modelo en {MODEL_PATH}. Entrénalo antes de iniciar la API."
		)
	app.state.modelo = joblib.load(MODEL_PATH)
	yield


app = FastAPI(
	title="Detector de fallas de GPU",
	description="Predice el estado de la GPU usando una ventana de telemetría.",
	lifespan=lifespan,
)


@app.post("/predecir")
def predecir(
	solicitud: SolicitudPrediccion, request: Request
) -> dict[str, str | float]:
	"""Calcula características y retorna el estado predicho y su confianza."""
	lecturas = [
		lectura.model_dump() if hasattr(lectura, "model_dump") else lectura.dict()
		for lectura in solicitud.lecturas
	]
	ventana = pd.DataFrame(lecturas)
	caracteristicas = extraer_caracteristicas(ventana)
	modelo = request.app.state.modelo

	try:
		estado = modelo.predict(caracteristicas)[0]
		probabilidades = modelo.predict_proba(caracteristicas)[0]
		clases = getattr(modelo, "classes_", None)
		if clases is None and hasattr(modelo, "named_steps"):
			clases = modelo.named_steps["clasificador"].classes_
		if clases is None:
			raise AttributeError("El modelo no expone las clases de clasificación")
		indice = list(clases).index(estado)
		confianza = float(probabilidades[indice])
	except (AttributeError, IndexError, KeyError, ValueError) as exc:
		raise HTTPException(
			status_code=500,
			detail="No fue posible obtener la predicción y su confianza del modelo.",
		) from exc

	return {"estado_predicho": str(estado), "confianza": confianza}
