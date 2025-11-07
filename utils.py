import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def load_and_clean(path_or_df):
    """
    Load dataframe (path or df).
    Null handling:
        - Numeric columns -> replace with mean
        - Categorical columns -> replace with mode or 'Unknown'
    """
    if isinstance(path_or_df, str):
        df = pd.read_csv(path_or_df)
    else:
        df = path_or_df.copy()

    # Numeric nulls -> mean
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols):
        imputer = SimpleImputer(strategy='mean')
        df[num_cols] = imputer.fit_transform(df[num_cols])

    # Categorical nulls -> mode
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        try:
            mode_val = df[col].mode(dropna=True)[0]
            df[col] = df[col].fillna(mode_val)
        except Exception:
            df[col] = df[col].fillna("Unknown")

    return df


def preprocess_for_model(df, target_col=None):
    """
    Preprocess dataset for ML training:
        - Detect target column automatically if not supplied
        - Encode categoricals
        - Scale numeric features
        - Drop ID columns
    Returns X, y, preprocess_info
    """
    df = load_and_clean(df)

    # Detect target column
    if target_col is None:
        candidates = [c for c in df.columns if "attrit" in c.lower() or c.lower() in ["attrition", "label", "target"]]
        if not candidates:
            raise ValueError("No Attrition/label column found. Please rename your target.")
        target_col = candidates[0]

    y = df[target_col]

    # Convert Yes/No to 1/0 if needed
    if y.dtype == object:
        y = y.map(lambda x: 1 if str(x).strip().lower() in ["yes", "y", "1", "tr]()
