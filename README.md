
# Customer Churn — Flask UI (Demo)

A small demo web app for exploring a customer churn classification model. The app includes:

- Single-record prediction (interactive form)
- Batch predictions from CSV upload and downloadable results
- Simple feature-importance visualization for single predictions
- Training-on-first-run with `model.pkl` persistence to speed subsequent starts

This repository is intentionally lightweight so you can run it locally for demos and iterate quickly.

Features
- Interactive UI with light/dark themes and accent color selection
- Single-prediction explainability via feature importances (Chart.js)
- Example batch CSV at `static/example_batch.csv`

Prerequisites
- Recommended: Miniconda / Anaconda (Windows users: avoids building native wheels). Tested with Python 3.11 in a conda environment.
- Alternatively: a system Python 3.11 installation (pip may need build tools on Windows).

Quick start — Conda (recommended)

1. Open PowerShell or your terminal.
2. Create and activate a Conda environment and install dependencies:

```powershell
conda create -n churn python=3.11 -y
conda activate churn
conda install pandas scikit-learn flask imbalanced-learn xgboost seaborn matplotlib -c conda-forge -y
```

3. Run the app from the project root:

```powershell
cd "D:\project 1\Customer Churn\-Customer-Churn-Detection"
python app.py
# open http://127.0.0.1:5000 in your browser
```

Optional: venv / pip (Python 3.11)

```powershell
python -m venv .venv
\.venv\Scripts\Activate.ps1        # PowerShell on Windows
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python app.py
```

Project structure

- `app.py` — Flask application, preprocessing, training, prediction, batch upload handling
- `Churn_Modelling.csv` — raw dataset used to train the demo model (should be in project root)
- `model.pkl` — persisted model + preprocessing context (created after first run)
- `templates/` — Jinja2 HTML templates (`index.html`, `result.html`)
- `static/` — static assets (`style.css`, `example_batch.csv`)

How the app works

- On first run the app loads `Churn_Modelling.csv`, runs the pipeline (preprocessing + SMOTE + classifier), and saves the training context to `model.pkl` so subsequent starts reuse the trained pipeline.
- The UI offers a **Predict** tab for single records and a **Batch** tab to upload a CSV containing columns:
	`CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary`
- Batch uploads return a CSV with an additional `prediction` column and `probability` where applicable.

Endpoints (for automation)

- `GET /` — HTML UI (index)
- `POST /predict` — form submit for single prediction (redirects to result page)
- `POST /predict_batch` — accepts `multipart/form-data` file upload; returns generated CSV download

Notes on model & reproducibility

- The demo uses `scikit-learn` pipelines and `imbalanced-learn`'s SMOTE during training. The exact training seed, model hyperparameters, and preprocessing steps are in `app.py`.
- `model.pkl` stores the encoders/scalers and the trained estimator so you do not retrain on every restart. Delete `model.pkl` to force retraining.

Batch CSV example

- A small example is included at `static/example_batch.csv`.
- Uploaded CSVs must contain the column header row and use the column names listed above.

Troubleshooting

- Error building `pandas` or other wheels on Windows: use the Conda instructions to install prebuilt binaries from `conda-forge`.
- `FileNotFoundError` for `Churn_Modelling.csv`: ensure the CSV is placed in the project root (same folder as `app.py`).
- Permissions on `uploads/` or `model.pkl`: ensure the process can write to the project folder.

Development & testing

- The project is small and function-focused; consider adding unit tests for `preprocess_input()` and prediction helpers in `app.py`.
- To iterate on UI assets, edit files under `templates/` and `static/` and reload the server (or enable Flask debug mode during development).

Docker (quick notes)

If you want a reproducible single-command run, creating a `Dockerfile` is straightforward. Example (not included):

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5000
CMD ["python", "app.py"]
```

Using Docker on Windows may still require building large wheels; prefer the Conda workflow for local development if you need optimized binary scientific packages.

Ideas & next steps

- Add SHAP explainability to the result page for richer per-sample explanations.
- Add a minimal API with OpenAPI docs so mobile/other services can call `/api/predict`.
- Add Docker + GitHub Actions CI for automated builds and tests.
- Persist models and metrics to a `models/` directory with metadata (timestamp, metrics, hyperparameters).

Contributing

- Fork the repo, create a feature branch, and open a PR. If you add breaking behavior (API changes or new required packages), update `README.md` with the new instructions.

Questions or want me to implement one of the next steps? I can add SHAP explainability or scaffold a `Dockerfile` and sample `docker-compose.yml` next.
# Customer Churn Analysis - EDA and Modeling

## Overview

This project focuses on Exploratory Data Analysis (EDA) and modeling to uncover insights from the data and build predictive models to understand Customer Churn.

Customer churn, also known as customer attrition, refers to the phenomenon where customers stop doing business with a company or service. It is a critical metric for businesses as it directly impacts revenue and profitability. 
High churn rates can indicate dissatisfaction with the product or service, poor customer experience.

## Dataset

The dataset used in this project is [Data Source](https://www.kaggle.com/datasets/rjmanoj/credit-card-customer-churn-prediction/data).

It contains the following features: 

 1. RowNumber
 2. CustomerId
 3. Surname
 4. CreditScore
 5. Geography
 6. Gender
 7. Age
 8. Tenure
 9. Balance
 10. NumOfProducts
 11. HasCrCard
 12. IsActiveMember
 13. EstimatedSalary
 14. Exited

The main variables of interest is **Exited**.

## Requirements

The following libraries are required to run the notebook:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Key Features

1. Handling Imbalanced Data: The project implements techniques to help handle imbalanced data such as SMOTE, ensuring accurate predictions even when the dependent variable is underrepresented.
2. Exploratory Data Analysis (EDA): The project features a stage of Exploratory Data Analysis (EDA), where we examine the data closely to identify trends and understand the reasons behind customer churn.
3. Classification: The project employs a variety of models, including Logistic Regression, Random Forest, K-Nearest Neighbors, Support Vector Machine, XGBoost, and Gradient Boosting, to predict customer churn, with techniques such as class weighting and SMOTE used to handle class imbalance.

## Results

| Model                   | Accuracy | Recall Score | F1 Score | ROC AUC Score |
|-------------------------|----------|--------------|----------|---------------|
| Logistic Regression     | 0.703667 | 0.683219     | 0.473029 | 0.764076      |
| Random Forest           | 0.862000 | 0.414384     | 0.538976 | 0.852447      |
| K-Nearest Neighbors     | 0.752333 | 0.667808     | 0.512147 | 0.776639      |
| Support Vector Machine  | 0.785667 | 0.662671     | 0.546224 | 0.822503      |
| XGBoost                 | 0.833000 | 0.609589     | 0.586974 | 0.841784      |
| Gradient Boosting       | 0.817000 | 0.700342     | 0.598391 | 0.859767      |

From the results of the classification models on the churn prediction dataset, we can infer the following:

1. **Gradient Boosting** has the highest F1 score (0.598391) and the highest ROC AUC score (0.859767) among all the models. This suggests that Gradient Boosting is the most effective model in balancing precision and recall and has the best ability to distinguish between the churned and non-churned customers.

2. **XGBoost** also performs well, with a relatively high F1 score (0.586974) and a good ROC AUC score (0.841784). This indicates that XGBoost is another strong model for this task.

3. **Random Forest** has a high accuracy (0.862000) but a lower F1 score (0.538976) compared to Gradient Boosting and XGBoost. This suggests that while Random Forest is good at predicting the majority class (non-churned customers), it might not be as effective at identifying the minority class (churned customers).

4. **Support Vector Machine** and **K-Nearest Neighbors** have moderate F1 scores and ROC AUC scores. They perform better than Logistic Regression but are not as effective as Gradient Boosting or XGBoost for this dataset.

5. **Logistic Regression** has the lowest accuracy (0.703667), F1 score (0.473029), and ROC AUC score (0.764076) among all the models. This indicates that Logistic Regression is the least effective model for predicting customer churn in this dataset.

#### Overall:

Gradient Boosting appears to be the best model for this churn prediction task, followed closely by XGBoost. These models are able to better handle the class imbalance and provide a good balance between precision and recall. 



