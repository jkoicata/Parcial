"""Entrena y serializa un clasificador de fallas a partir de telemetría GPU."""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

try:  # Permite ejecutar como ``python src/train.py`` o ``python -m src.train``.
    from .features import extraer_caracteristicas
    from .schema import telemetria_schema
except ImportError:
    from features import extraer_caracteristicas
    from schema import telemetria_schema


PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_DIR.parent


def crear_dataset(
    telemetria: pd.DataFrame, duracion_ventana: int = 30
) -> tuple[pd.DataFrame, pd.Series]:
    """Agrupa por episodio y devuelve las características y etiquetas.

    Las ventanas comienzan en el segundo mínimo de cada episodio. Si hay
    varios estados en una ventana, se usa el más frecuente; los empates se
    resuelven de forma determinista por orden alfabético.
    """
    if duracion_ventana <= 0:
        raise ValueError("duracion_ventana debe ser mayor que cero")

    filas_features: list[pd.DataFrame] = []
    etiquetas: list[str] = []

    for _, episodio in telemetria.groupby("episodio_id", sort=False):
        episodio = episodio.sort_values("segundo")
        segundo_inicial = episodio["segundo"].min()
        id_ventana = (episodio["segundo"] - segundo_inicial) // duracion_ventana

        for _, ventana in episodio.groupby(id_ventana, sort=True):
            filas_features.append(extraer_caracteristicas(ventana))
            etiquetas.append(ventana["estado"].mode().iloc[0])

    if not filas_features:
        raise ValueError("No se generaron ventanas de telemetría para entrenar")

    X = pd.concat(filas_features, ignore_index=True)
    y = pd.Series(etiquetas, name="estado")
    return X, y


def resolver_csv(ruta_csv: Path | None) -> Path:
    """Devuelve la ruta indicada o busca el CSV en ubicaciones habituales."""
    if ruta_csv is not None:
        return ruta_csv.expanduser().resolve()

    candidatos = (
        PROJECT_DIR / "data" / "telemetria_publica.csv",
        WORKSPACE_DIR / "data" / "telemetria_publica.csv",
    )
    for candidato in candidatos:
        if candidato.is_file():
            return candidato
    return candidatos[0]


def entrenar(ruta_csv: Path | None = None, ruta_modelo: Path | None = None) -> float:
    """Valida, entrena, evalúa y guarda el pipeline completo del clasificador."""
    csv_path = resolver_csv(ruta_csv)
    model_path = (
        ruta_modelo or PROJECT_DIR / "models" / "modelo.joblib"
    ).expanduser().resolve()

    if not csv_path.is_file():
        raise FileNotFoundError(f"No se encontró el CSV de telemetría: {csv_path}")
    telemetria = pd.read_csv(csv_path)
        # Limpiar registros inválidos
    filas_originales = len(telemetria)
    telemetria = telemetria[telemetria["power_w"] > 0]
    print(f"Se eliminaron {filas_originales - len(telemetria)} filas con power_w negativo")
    telemetria = telemetria_schema.validate(telemetria)
    X, y = crear_dataset(telemetria, duracion_ventana=30)

    if len(X) < 2:
        raise ValueError("Se necesitan al menos dos ventanas para dividir los datos")

    # Solo estratifica cuando hay suficientes ejemplos en ambos conjuntos.
    conteos = y.value_counts()
    cantidad_test = max(1, int(len(y) * 0.2 + 0.999999))
    cantidad_train = len(y) - cantidad_test
    estratificar = (
        y
        if conteos.min() >= 2
        and cantidad_test >= len(conteos)
        and cantidad_train >= len(conteos)
        else None
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=estratificar,
    )

    pipeline = Pipeline(
        steps=[
            (
                "clasificador",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    class_weight="balanced",
                ),
            )
        ]
    )
    pipeline.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, pipeline.predict(X_test))
    print(f"Accuracy: {accuracy:.4f}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Pipeline guardado en: {model_path}")
    return accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Ruta al CSV (por defecto, busca data/telemetria_publica.csv)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Ruta de salida del pipeline serializado",
    )
    args = parser.parse_args()
    entrenar(args.csv, args.model)


if __name__ == "__main__":
    main()
