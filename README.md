# Medical Insurance Premium Prediction API

A machine-learning project that predicts an insurance premium category (`Low`, `Medium`, or `High`) from applicant attributes. The final notebook builds engineered features, applies one-hot encoding to categorical variables, and trains a Random Forest classifier inside a scikit-learn `Pipeline`. FastAPI exposes the fitted pipeline through a REST API.

> **Scope:** This is an educational portfolio project, not an underwriting system or financial/insurance advice. The dataset contains 100 observations; reported performance should not be interpreted as production-grade validation.

## Project structure

```text
Medical-Insurance-Premium-Prediction/
├── app.py
├── config/city_tier.py
├── data/insurance.csv
├── docs/
├── model/model.pkl
├── model/predict.py
├── notebooks/final_model.ipynb
├── schemas/output_validation.py
├── schemas/user_input.py
├── tests/test_api.py
├── requirements.txt
└── .gitignore
```

## 1. Environment setup (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Use the Python version and dependency versions recorded in `requirements.txt`. The pickle model must be loaded with a compatible scikit-learn version. Only load model files from trusted sources.

## 2. Run the API

From the project root:

```powershell
uvicorn app:app --reload
```

Open Swagger UI at `http://127.0.0.1:8000/docs`. Health check: `http://127.0.0.1:8000/health`.

## 3. Example request

`POST http://127.0.0.1:8000/predict`

```json
{
  "age": 36,
  "weight": 70,
  "height": 1.75,
  "income_lpa": 12.5,
  "smoker": false,
  "city": "Kolkata",
  "occupation": "private_job"
}
```

The API computes BMI, age group, lifestyle risk, and city tier using the same rules as the notebook before calling the model. The response includes the predicted category, maximum class probability, and probabilities for all classes. A probability is a model score, not a guarantee of outcome or a calibrated risk estimate.

## 4. Run tests

```powershell
pytest -q
```

## 5. Model details

- Estimator: `RandomForestClassifier`
- `n_estimators=200`, `max_depth=5`, `min_samples_split=5`, `min_samples_leaf=3`, `max_features="sqrt"`, `random_state=42`
- Categorical features: age group, lifestyle risk, occupation, city tier
- Numeric features: BMI and income in LPA
- Target: `insurance_premium_category`
- Preprocessing and classifier are persisted together as one scikit-learn pipeline.

The notebook uses an 80/20 random train/test split (`random_state=1`) and 5-fold shuffled `KFold` cross-validation on the training subset (`random_state=42`). Since this is a small dataset, fold and holdout metrics can vary. See the notebook for the actual evaluation outputs; do not treat a single score as proof of real-world performance.

## 6. Push to GitHub

```powershell
git init
git add .
git status
git commit -m "Build insurance premium prediction API"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repository>.git
git push -u origin main
```

Replace the remote URL with your repository URL. Review `git status` before committing; do not commit virtual environments, secrets, personal paths, or unrelated files.

## Reproducibility and limitations

- The notebook is the source of the feature engineering and model design.
- The API reproduces those feature transformations at inference time; keep both implementations synchronized if rules change.
- The bundled `model.pkl` was fitted from the supplied `insurance.csv` using the notebook's feature construction, split seed, and Random Forest configuration. Re-run the notebook and re-export the pipeline whenever the training data or model configuration changes.
- Pickle/joblib artifacts can execute code during loading. Load only trusted artifacts and keep the serving environment's scikit-learn version compatible.
- The dataset is small and may not represent the population or pricing practices of any insurer. Do not use this project for real customer decisions.
