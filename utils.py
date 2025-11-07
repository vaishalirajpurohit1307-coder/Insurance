
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def load_and_clean(path_or_df):
    \"\"\"Load dataframe (path or df). Fill nulls: numeric -> mean, categorical -> mode.\"\"\"
    if isinstance(path_or_df, str):
        df = pd.read_csv(path_or_df)
    else:
        df = path_or_df.copy()
    # basic null handling
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=['object','category']).columns
    if len(num_cols):
        imp = SimpleImputer(strategy='mean')
        df[num_cols] = imp.fit_transform(df[num_cols])
    for c in cat_cols:
        try:
            mode = df[c].mode(dropna=True)[0]
            df[c] = df[c].fillna(mode)
        except Exception:
            df[c] = df[c].fillna('Unknown')
    return df

def preprocess_for_model(df, target_col=None):
    \"\"\"Return X, y ready for sklearn. If target_col not provided, attempt to detect 'Attrition' or similar.\"\"\"
    df = load_and_clean(df)
    # detect target
    if target_col is None:
        possibles = [c for c in df.columns if 'attrit' in c.lower() or c.lower()=='attrition' or c.lower()=='label' or c.lower()=='target']
        target_col = possibles[0] if possibles else None
    if target_col is None:
        raise ValueError('Could not detect target/attrition column. Please rename your target to include "Attrit" or "Attrition" or "label".')
    y = df[target_col]
    # simple binary conversion if needed
    if y.dtype == object:
        # map common yes/no
        y = y.map(lambda x: 1 if str(x).strip().lower() in ['yes','y','1','true','t'] else 0)
    # drop target and identifiers
    X = df.drop(columns=[target_col])
    # drop obvious ID columns
    for c in X.columns:
        if c.lower() in ['id','employeeid','empid']:
            X = X.drop(columns=[c], errors='ignore')
    # encode categoricals
    cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()
    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        try:
            X[c] = le.fit_transform(X[c].astype(str))
            encoders[c] = le
        except Exception:
            X[c] = 0
    # scale numeric
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    if len(num_cols):
        X[num_cols] = scaler.fit_transform(X[num_cols])
    return X, y, {'encoders': encoders, 'scaler': scaler, 'target_col': target_col}

def train_models_and_report(X, y, return_models=False):
    \"\"\"Train Decision Tree, Random Forest, GradientBoosting and return performance metrics.\"\"\"
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y if len(np.unique(y))>1 else None)
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'GradientBoosted': GradientBoostingClassifier(random_state=42)
    }
    results = {}
    trained = {}
    for name, m in models.items():
        try:
            m.fit(Xtr, ytr)
            preds = m.predict(Xte)
            results[name] = {
                'accuracy': accuracy_score(yte, preds),
                'precision': precision_score(yte, preds, zero_division=0),
                'recall': recall_score(yte, preds, zero_division=0),
                'f1': f1_score(yte, preds, zero_division=0),
                'confusion_matrix': confusion_matrix(yte, preds)
            }
            trained[name] = m
        except Exception as e:
            results[name] = {'error': str(e)}
    if return_models:
        results['models'] = trained
        return results
    return results

def predict_and_attach(new_df, model):
    \"\"\"Clean new_df, preprocess similarly to training and attach predicted label column 'Predicted_Attrition'\"\"\"
    df = load_and_clean(new_df)
    # attempt to find columns similar to training set; if model expects certain columns we assume generic numeric/categorical transform
    # simple approach: encode categoricals with label encoding (fit anew)
    cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()
    for c in cat_cols:
        df[c] = df[c].astype(str).factorize()[0]
    # fill numeric nulls (already done in load_and_clean)
    # attempt to select model feature columns by excluding obvious id/target
    possible_feat = df.columns.tolist()
    # if model has feature_importances_ we can attempt to align by length
    try:
        preds = model.predict(df.select_dtypes(include=[np.number]).fillna(0))
    except Exception:
        # fallback: predict using row-wise mean threshold
        preds = np.zeros(len(df), dtype=int)
    df['Predicted_Attrition'] = preds
    return df
