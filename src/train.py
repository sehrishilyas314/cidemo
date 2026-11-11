from pathlib import Path
import json

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]


def train_model(data_path=None, model_path=None, metrics_path=None, params_path=None):
    data_path = Path(data_path or ROOT / "data" / "iris.csv")
    model_path = Path(model_path or ROOT / "models" / "iris_model.joblib")
    metrics_path = Path(metrics_path or ROOT / "metrics.json")
    params_path = Path(params_path or ROOT / "params.yaml")

    df = pd.read_csv(data_path)
    required = {"sepal_length", "sepal_width", "petal_length", "petal_width", "species"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["species"])

    with params_path.open(encoding="utf-8") as f:
        params = yaml.safe_load(f)["train"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["test_size"],
        random_state=params["random_state"], stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        random_state=params["random_state"]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "encoder": encoder}, model_path)
    metrics_path.write_text(
        json.dumps({"accuracy": round(float(accuracy), 4)}, indent=2),
        encoding="utf-8"
    )
    print(f"Model saved to: {model_path}")
    print(f"Test accuracy: {accuracy:.4f}")
    return accuracy


if __name__ == "__main__":
    train_model()
