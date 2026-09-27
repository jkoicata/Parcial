"""Esquema Pandera para validar telemetría de GPUs."""

import pandera as pa
from pandera import Check, Column, DataFrameSchema


ESTADOS_VALIDOS = [
	"normal",
	"sobrecalentamiento",
	"degradacion_memoria",
	"falla_alimentacion",
]


telemetria_schema = DataFrameSchema(
	{
		"episodio_id": Column(int),
		"segundo": Column(int),
		"temp_c": Column(float, Check.in_range(0, 120)),
		"power_w": Column(float, Check.gt(0)),
		"util_pct": Column(float, Check.in_range(0, 100)),
		"clock_mhz": Column(float, Check.gt(0)),
		"ecc_errors": Column(int, Check.ge(0)),
		"estado": Column(str, Check.isin(ESTADOS_VALIDOS)),
	},
	strict=True,
	coerce=True,
)
