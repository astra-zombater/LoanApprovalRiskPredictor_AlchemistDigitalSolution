# Loan Approval & Credit Risk Prediction System

> **A machine learning project that studies historical loan decisions and predicts whether an application is likely to be approved.**

Built with Python, Pandas, Scikit-learn and Streamlit.

---

## The idea

Banks don't make lending decisions from one number. Income, requested loan amount, repayment period, credit history and other applicant details all play a part.

For this project, I worked with a public loan prediction dataset and built a classification workflow to explore those patterns, compare different ML models, and turn the selected pipeline into a small interactive app.

The goal is straightforward: take applicant information, process it consistently, and estimate the likelihood of the application receiving an approval in the historical dataset.

**A note on the target:** the dataset contains past loan approval decisions (`Loan_Status`). It does not contain verified future repayment/default outcomes. So this project predicts *loan approval*, not whether a borrower will actually default. The phrase “credit risk” in the project title describes the wider use case, not a default model trained on repayment history.

## What it does

- Cleans the dataset and checks for duplicate records.
- Handles missing numerical and categorical values inside Scikit-learn pipelines.
- Encodes categorical features and scales numerical features.
- Engineers additional income and repayment-related features.
- Explores approval patterns through visualizations.
- Trains and compares four classification algorithms.
- Evaluates models with Accuracy, Precision, Recall, F1-score and ROC-AUC.
- Tunes a Random Forest model using `GridSearchCV`.
- Saves the final trained pipeline and loads it in a Streamlit app.

## Dataset

The project uses the classic **Loan Prediction Problem** dataset, containing 614 applications and 13 columns in the supplied training CSV.

- Dataset: [Loan Prediction Problem Dataset — Kaggle](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)
- Target column: `Loan_Status`
- `Y` = approved, `N` = rejected
- `Loan_ID` is an identifier and is removed before modelling.

The target distribution in this dataset is approximately 68.7% approved and 31.3% rejected. That imbalance is one reason the project reports more than accuracy alone.

Please check the dataset page for its original licence and usage terms before redistributing the raw data.

## Feature engineering

Alongside the original applicant fields, the project creates a few features to give the models more useful financial context:

| Feature | How it is calculated | Why it is included |
|---|---|---|
| `Total_Income` | Applicant income + co-applicant income | Combined monthly income |
| `EMI` | Loan amount × 1000 / loan term | Approximate monthly instalment using the dataset's loan-amount scale |
| `Income_to_Loan` | Total income / loan amount | Income relative to the requested amount |
| `EMI_to_Income` | EMI / total income | Approximate instalment burden relative to income |
| `*_log` | `log1p()` of selected income/loan values | Reduces the effect of highly skewed numerical distributions |

The numerical features are imputed with the median and scaled with `StandardScaler`. Categorical features are imputed with the most frequent value and encoded using `OneHotEncoder`. Credit history is imputed separately.

## Models and results

Four models were compared on a stratified 80/20 train-test split (random state 42). The figures below are the results recorded in the current notebook run.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | CV ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.854 | 0.832 | 0.988 | 0.903 | **0.878** | 0.723 |
| Random Forest | **0.870** | **0.888** | 0.929 | **0.908** | 0.874 | **0.773** |
| Gradient Boosting | 0.854 | 0.860 | 0.941 | 0.899 | 0.833 | 0.730 |
| Decision Tree | 0.846 | 0.844 | 0.953 | 0.895 | 0.747 | 0.653 |

In this run, Logistic Regression has the highest test ROC-AUC (0.878), while Random Forest has the highest accuracy and F1-score. The notebook's initial confusion matrix and feature-coefficient analysis are for Logistic Regression. Random Forest is then tuned separately with `GridSearchCV` and saved as the app's final pipeline.

The recorded Random Forest tuning results are:

- Best cross-validation ROC-AUC: **0.780**
- Test ROC-AUC after tuning: **0.880**
- Test F1-score after tuning: **0.913**
- Best parameters: `n_estimators=100`, `max_depth=None`, `min_samples_leaf=1`, `class_weight=None`

These are results from one split and one notebook run, not a guarantee of performance on new applicants. The tuned model's complete five-metric report and confusion matrix should be generated from the tuned pipeline before treating it as the final evaluated model.

## What the results tell us

A high approval recall means the model catches most historically approved applications. But that is only one side of the decision. The initial Logistic Regression report shows 0.99 recall for approved applications and 0.55 recall for rejected applications. In other words, it misses a meaningful share of historically rejected cases.

For a lending workflow, false approvals and false rejections can have different costs. A real deployment would need a clearly defined business objective, threshold analysis, fairness checks, and validation on representative, current data. Model predictions should support human review rather than act as an automatic lending decision.

Feature relationships in this dataset are historical patterns, not proof that a feature causes an approval decision. Demographic attributes also require careful legal and fairness review before any real-world use.

## Project structure

```text
loan-approval-predictor/
├── app.py                         # Streamlit eligibility predictor
├── data/
│   └── train.csv                  # Loan prediction dataset
├── models/
│   ├── loan_pipeline.joblib       # Pipeline loaded by the app
│   ├── best_model.pkl             # Saved model artifact
│   ├── feature_names.pkl          # Saved feature names
│   └── scaler.pkl                 # Saved scaler artifact
├── notebooks/
│   └── loan_approval_analysis.ipynb
├── src/
│   └── features.py                # Shared feature engineering
├── README.md
├── requirements.txt
└── structure.md
```

`loan_pipeline.joblib` is the artifact currently loaded by `app.py`. The other model files are retained from the project work and are not loaded by the Streamlit app.

## Run it locally

### 1. Clone or download the project

```bash
git clone <your-repository-url>
cd loan-approval-predictor
```

If you downloaded the ZIP instead, extract it and open a terminal in the `loan-approval-predictor` folder.

### 2. Create a virtual environment

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Launch the predictor

```bash
python -m streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`). Open it in your browser.

### 5. Open the notebook

```bash
jupyter notebook
```

Then open `notebooks/loan_approval_analysis.ipynb`. The notebook reads the CSV from `../data/train.csv` and saves the final pipeline to `../models/loan_pipeline.joblib` when run from the notebook directory, as intended by its relative paths.

## Tech stack

- **Python** — main language
- **Pandas / NumPy** — data handling and feature calculations
- **Matplotlib / Seaborn** — exploratory visualizations
- **Scikit-learn** — preprocessing, classification, cross-validation, evaluation and tuning
- **Joblib** — model persistence
- **Streamlit** — interactive web interface
- **Jupyter Notebook** — experimentation and analysis

## Current limitations

- The dataset is small and historical; it may not represent current lending populations or policies.
- The target is approval status, not observed loan repayment or default.
- A single train-test split gives only one view of model performance.
- The app displays a model-estimated approval probability; it is not a bank decision or a calibrated guarantee.
- This is an educational ML project, not a production credit-scoring service.

## Next steps

- Evaluate the tuned final pipeline with all five required metrics and its own confusion matrix.
- Align feature-importance reporting with the final deployed Random Forest pipeline.
- Add stronger input validation and clearer decision explanations in the app.
- Test the project in a clean environment and record the exact package versions used to train the saved model.

---

**Built as an applied machine-learning project - from raw applicant data to a working prediction interface.**
