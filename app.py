from flask import Flask, render_template, request
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_pipeline_imb

import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)


def load_and_train():
    df = pd.read_csv('Churn_Modelling.csv')

    # Feature engineering (same as notebook)
    df['CreditUtilization'] = df['Balance'] / df['CreditScore']
    df['InteractionScore'] = df['NumOfProducts'] + df['HasCrCard'] + df['IsActiveMember']
    df['BalanceToSalaryRatio'] = df['Balance'] / df['EstimatedSalary']
    df['CreditScoreAgeInteraction'] = df['CreditScore'] * df['Age']

    # Credit score groups
    bins = [0, 669, 739, 850]
    labels = ['Low', 'Medium', 'High']
    df['CreditScoreGroup'] = pd.cut(df['CreditScore'], bins=bins, labels=labels, include_lowest=True)

    # Encode categorical columns
    cat_col = ['Geography', 'Gender', 'CreditScoreGroup']
    encoders = {}
    for col in cat_col:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Prepare training data
    col_drop = ['Exited', 'RowNumber', 'CustomerId', 'Surname']
    X = df.drop(col_drop, axis=1, errors='ignore')
    y = df['Exited']

    scaling_columns = ['Age', 'CreditScore', 'Balance', 'EstimatedSalary', 'CreditUtilization', 'BalanceToSalaryRatio', 'CreditScoreAgeInteraction']
    scaler = StandardScaler()
    X[scaling_columns] = scaler.fit_transform(X[scaling_columns])

    # Train-test split for metrics
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Model pipeline with SMOTE
    model = make_pipeline_imb(SMOTE(random_state=42), GradientBoostingClassifier(random_state=42))
    model.fit(X_train, y_train)

    # Compute metrics
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    try:
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    except Exception:
        roc_auc = None

    metrics = {
        'accuracy': round(acc, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'roc_auc': round(roc_auc, 4) if roc_auc is not None else None,
    }

    context = {
        'df_sample': df.head(10).to_html(classes='table table-sm table-striped', index=False),
        'metrics': metrics,
        'encoders': encoders,
        'scaler': scaler,
        'features_columns': X.columns.tolist(),
        'full_df': df,
        'model': model,
    }
    return context


ctx = load_and_train()


def preprocess_input(form, ctx):
    # Expected fields from form: CreditScore, Geography, Gender, Age, Tenure, Balance,
    # NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
    df_row = {}
    df_row['CreditScore'] = float(form.get('CreditScore', 650))
    df_row['Geography'] = form.get('Geography', 'France')
    df_row['Gender'] = form.get('Gender', 'Male')
    df_row['Age'] = float(form.get('Age', 40))
    df_row['Tenure'] = int(form.get('Tenure', 3))
    df_row['Balance'] = float(form.get('Balance', 0.0))
    df_row['NumOfProducts'] = int(form.get('NumOfProducts', 1))
    df_row['HasCrCard'] = int(form.get('HasCrCard', 1))
    df_row['IsActiveMember'] = int(form.get('IsActiveMember', 1))
    df_row['EstimatedSalary'] = float(form.get('EstimatedSalary', 50000))

    # Feature engineering
    df_row['CreditUtilization'] = df_row['Balance'] / (df_row['CreditScore'] if df_row['CreditScore'] != 0 else 1)
    df_row['InteractionScore'] = df_row['NumOfProducts'] + df_row['HasCrCard'] + df_row['IsActiveMember']
    df_row['BalanceToSalaryRatio'] = df_row['Balance'] / (df_row['EstimatedSalary'] if df_row['EstimatedSalary'] != 0 else 1)
    df_row['CreditScoreAgeInteraction'] = df_row['CreditScore'] * df_row['Age']

    # CreditScoreGroup
    cs = df_row['CreditScore']
    if cs <= 669:
        cs_group = 'Low'
    elif cs <= 739:
        cs_group = 'Medium'
    else:
        cs_group = 'High'
    df_row['CreditScoreGroup'] = cs_group

    # Create DataFrame with same columns order as training
    features = ctx['features_columns']
    row = pd.DataFrame([df_row], columns=features)

    # Apply encoders
    encoders = ctx['encoders']
    # Geography and Gender and CreditScoreGroup were label encoded
    for col in ['Geography', 'Gender', 'CreditScoreGroup']:
        le = encoders[col]
        # transform expects the same labels used during fit; map if needed
        row[col] = le.transform(row[col].astype(str))

    # Scale numeric columns
    scaling_columns = ['Age', 'CreditScore', 'Balance', 'EstimatedSalary', 'CreditUtilization', 'BalanceToSalaryRatio', 'CreditScoreAgeInteraction']
    row[scaling_columns] = ctx['scaler'].transform(row[scaling_columns])

    return row


@app.route('/', methods=['GET'])
def index():
    metrics = ctx['metrics']
    sample_html = ctx['df_sample']
    return render_template('index.html', metrics=metrics, sample_table=sample_html)


@app.route('/predict', methods=['POST'])
def predict():
    form = request.form
    row = preprocess_input(form, ctx)
    model = ctx['model']
    pred = model.predict(row)[0]
    try:
        prob = model.predict_proba(row)[0][1]
    except Exception:
        prob = None

    return render_template('result.html', prediction=int(pred), probability=prob)


if __name__ == '__main__':
    app.run(debug=True)
