"""Extracción de características estadísticas de ventanas de telemetría."""

import pandas as pd


def extraer_caracteristicas(ventana: pd.DataFrame) -> pd.DataFrame:
	"""Resume una ventana de telemetría en una única fila de características.

	La desviación estándar se calcula con ``ddof=0`` para que también esté
	definida cuando la ventana contiene una sola muestra.

	Args:
		ventana: DataFrame con las columnas temp_c, power_w, util_pct,
			clock_mhz y ecc_errors.

	Returns:
		DataFrame de una fila, con nombres de columnas estables para Scikit-Learn.

	Raises:
		TypeError: Si ``ventana`` no es un DataFrame de pandas.
		ValueError: Si la ventana está vacía o faltan columnas requeridas.
	"""
	if not isinstance(ventana, pd.DataFrame):
		raise TypeError("ventana debe ser un pandas.DataFrame")
	if ventana.empty:
		raise ValueError("ventana no puede estar vacía")

	columnas_requeridas = [
		"temp_c",
		"power_w",
		"util_pct",
		"clock_mhz",
		"ecc_errors",
	]
	faltantes = [col for col in columnas_requeridas if col not in ventana.columns]
	if faltantes:
		raise ValueError(f"Faltan columnas requeridas: {', '.join(faltantes)}")

	caracteristicas: dict[str, float] = {}
	for columna in ("temp_c", "power_w", "util_pct", "clock_mhz"):
		valores = pd.to_numeric(ventana[columna], errors="raise")
		caracteristicas[f"{columna}_mean"] = float(valores.mean())
		caracteristicas[f"{columna}_std"] = float(valores.std(ddof=0))
		caracteristicas[f"{columna}_min"] = float(valores.min())
		caracteristicas[f"{columna}_max"] = float(valores.max())

	caracteristicas["power_w_rango"] = (
		caracteristicas["power_w_max"] - caracteristicas["power_w_min"]
	)
	caracteristicas["clock_mhz_rango"] = (
    	caracteristicas["clock_mhz_max"] - caracteristicas["clock_mhz_min"]
	)
	errores_ecc = pd.to_numeric(ventana["ecc_errors"], errors="raise")
	caracteristicas["ecc_errors_suma"] = float(errores_ecc.sum())
	caracteristicas["ecc_errors_mean"] = float(errores_ecc.mean())
	caracteristicas["ecc_errors_max"] = float(errores_ecc.max())

	return pd.DataFrame([caracteristicas])