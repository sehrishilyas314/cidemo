from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_exists_and_has_expected_columns():
    path = ROOT / "data" / "iris.csv"
    assert path.exists()
    df = pd.read_csv(path)
    assert len(df) >= 12
    assert list(df.columns) == [
        "sepal_length", "sepal_width", "petal_length", "petal_width", "species"
    ]


def test_dataset_has_three_species():
    df = pd.read_csv(ROOT / "data" / "iris.csv")
    assert df["species"].nunique() == 3
