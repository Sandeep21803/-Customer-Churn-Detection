from flask import Flask, render_template, request, send_file
from flask import Response
import pandas as pd
import numpy as np
import os
import pickle

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_pipeline_imb

import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def load_and_train(persist_path='model.pkl'):
    # Load persisted context if available
    if os.path.exists(persist_path):
        with open(persist_path, 'rb') as f:
            context = pickle.load(f)

        # If an older persisted context did not include test split, try to reconstruct it
        if 'X_test' not in context or 'y_test' not in context:
            try:
                # Prefer embedded full_df when available
                if 'full_df' in context and context['full_df'] is not None:
                    df = context['full_df']
                else:
                    df = pd.read_csv('Churn_Modelling.csv')

                # Recreate the same feature engineering steps used during training
                df = df.copy()
                df['CreditUtilization'] = df['Balance'] / df['CreditScore']
                df['InteractionScore'] = df['NumOfProducts'] + df['HasCrCard'] + df['IsActiveMember']
                df['BalanceToSalaryRatio'] = df['Balance'] / df['EstimatedSalary']
                df['CreditScoreAgeInteraction'] = df['CreditScore'] * df['Age']
                bins = [0, 669, 739, 850]
                labels = ['Low', 'Medium', 'High']
                df['CreditScoreGroup'] = pd.cut(df['CreditScore'], bins=bins, labels=labels, include_lowest=True)

                # Apply stored encoders if present
                encoders = context.get('encoders')
                if encoders:
                    for col in ['Geography', 'Gender', 'CreditScoreGroup']:
                        if col in encoders and col in df.columns:
                            le = encoders[col]
                            df[col] = le.transform(df[col].astype(str))

                # Build X and y
                col_drop = ['Exited', 'RowNumber', 'CustomerId', 'Surname']
                X = df.drop(col_drop, axis=1, errors='ignore')
                y = df['Exited']

                # Apply stored scaler if present
                scaler = context.get('scaler')
                scaling_columns = ['Age', 'CreditScore', 'Balance', 'EstimatedSalary', 'CreditUtilization', 'BalanceToSalaryRatio', 'CreditScoreAgeInteraction']
                if scaler is not None:
                    X[scaling_columns] = scaler.transform(X[scaling_columns])

                # recreate test split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                context['X_test'] = X_test
                context['y_test'] = y_test
            except Exception:
                # If reconstruction fails, leave context as-is and ROC endpoint will return 404
                pass

        return context

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
        'X_test': X_test,
        'y_test': y_test,
        'model': model,
    }

    # Persist context for faster subsequent startups
    with open(persist_path, 'wb') as f:
        pickle.dump(context, f)

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

    # Compute local feature importances (if available)
    importances = None
    try:
        base_model = None
        # if pipeline, find last estimator with feature_importances_
        if hasattr(model, 'named_steps'):
            # get last step
            last = list(model.named_steps.items())[-1][1]
            base_model = last
        else:
            base_model = model

        if hasattr(base_model, 'feature_importances_'):
            fi = base_model.feature_importances_
            features = ctx['features_columns']
            imp_df = pd.DataFrame({'feature': features, 'importance': fi})
            imp_df = imp_df.sort_values('importance', ascending=False).head(10)
            importances = imp_df.to_dict(orient='records')
    except Exception:
        importances = None

    return render_template('result.html', prediction=int(pred), probability=prob, importances=importances)


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    file = request.files.get('file')
    if not file:
        return 'No file uploaded', 400

    path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(path)

    df_in = pd.read_csv(path)
    # Expect same input columns as used in form; try to map
    required = ['CreditScore','Geography','Gender','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']
    missing = [c for c in required if c not in df_in.columns]
    if missing:
        return f'Missing columns in upload: {missing}', 400

    rows = []
    for _, r in df_in.iterrows():
        form = r.to_dict()
        row = preprocess_input(form, ctx)
        pred = ctx['model'].predict(row)[0]
        try:
            prob = ctx['model'].predict_proba(row)[0][1]
        except Exception:
            prob = None
        rows.append({'prediction': int(pred), 'probability': float(prob) if prob is not None else None})

    out = pd.concat([df_in.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    out_path = path.replace('.csv', '_predictions.csv')
    out.to_csv(out_path, index=False)

    return send_file(out_path, as_attachment=True)


@app.route('/plots/roc.png')
def plot_roc():
    # generate ROC curve PNG using stored test set
    try:
        X_test = ctx.get('X_test')
        y_test = ctx.get('y_test')
        model = ctx.get('model')
        if X_test is None or y_test is None:
            return 'No test data available', 404

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc

        y_score = None
        try:
            y_score = model.predict_proba(X_test)[:, 1]
        except Exception:
            y_score = model.decision_function(X_test) if hasattr(model, 'decision_function') else None

        if y_score is None:
            return 'Probability scores not available for ROC', 400

        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(fpr, tpr, color='tab:blue', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower right')
        fig.tight_layout()

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        return Response(buf.getvalue(), mimetype='image/png')
    except Exception as e:
        return f'Error generating ROC plot: {e}', 500


@app.route('/plots/dist.png')
def plot_dist():
    # generate class distribution plot using seaborn/matplotlib
    try:
        full_df = ctx.get('full_df')
        if full_df is None:
            return 'No dataset available', 404

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x='Exited', data=full_df, palette='pastel', ax=ax)
        ax.set_xlabel('Exited (1 = churn)')
        ax.set_ylabel('Count')
        fig.tight_layout()

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        return Response(buf.getvalue(), mimetype='image/png')
    except Exception as e:
        return f'Error generating distribution plot: {e}', 500


if __name__ == '__main__':
    app.run(debug=True)
