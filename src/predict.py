from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

bundle = joblib.load(ROOT / "models" / "iris_model.joblib")

model = bundle["model"]
encoder = bundle["encoder"]

sample = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=[
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ]
)

predicted_class = model.predict(sample)[0]

print("Predicted species:", encoder.inverse_transform([predicted_class])[0])