# Iris ML + MLOps Starter

A beginner-friendly Machine Learning project for practicing Git, DVC, CI, and GitHub Actions.

## What is included?
- `data/iris.csv`: Iris flower dataset (150 rows when generated from scikit-learn).
- `src/train.py`: trains a Random Forest classifier and saves a model plus metrics.
- `src/predict.py`: loads the saved model and predicts one example.
- `tests/`: basic dataset validation tests.
- `dvc.yaml`: DVC training pipeline.
- `.github/workflows/ci.yml`: GitHub Actions workflow for tests and training.
- `params.yaml`: model training parameters.

## 1. Open the project in VS Code
Extract the ZIP, open the `iris_mlops_starter` folder in VS Code, then open Terminal.

## 2. Create and activate a virtual environment (Windows PowerShell)
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```
If PowerShell blocks activation, run this once in the current terminal:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

## 3. Install dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pyyaml
```

## 4. Run tests and train once
```powershell
pytest -q
python src/train.py
python src/predict.py
```

## 5. Start Git
```powershell
git init
git add .
git commit -m "Initial Iris ML project"
```
Create an empty GitHub repository, then connect it (replace YOUR-USERNAME):
```powershell
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/iris-mlops-practice.git
git push -u origin main
```

## 6. Initialize DVC and track the dataset
```powershell
dvc init
dvc add data/iris.csv
git add data/iris.csv.dvc data/.gitignore .dvc .dvcignore
git commit -m "Track Iris dataset with DVC"
```

Note: after `dvc add`, DVC creates `data/iris.csv.dvc` and a `data/.gitignore`. The dataset itself is stored in DVC's local cache. To share data with another computer, configure a DVC remote (for example, a cloud storage location) and run `dvc push`. Do not commit private credentials.

## 7. Run the DVC pipeline
```powershell
dvc repro
dvc metrics show
git status
```
Commit the pipeline files and generated DVC metadata:
```powershell
git add dvc.yaml params.yaml src tests .github
git add dvc.lock
git commit -m "Add DVC training pipeline"
```

## 8. Push and check GitHub Actions
```powershell
git push
```
Open your GitHub repository → **Actions** tab → open the latest workflow run.

## Practice tasks
1. Change `n_estimators` in `params.yaml` from 100 to 150, then run `dvc repro`.
2. Add a new test in `tests/test_model.py`.
3. Change the training code and push to GitHub. Check whether Actions runs automatically.
4. Compare metrics using `dvc metrics show`.

## Important
- The included GitHub Actions workflow installs dependencies, runs tests, and trains the model.
- It does not download data from a DVC remote; the small starter dataset is included in the repository at first.
- Later, we can improve CI so it restores data from a configured DVC remote and validates the DVC pipeline end-to-end.
