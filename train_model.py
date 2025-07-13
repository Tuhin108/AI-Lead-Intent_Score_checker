# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

def load_and_clean_data(filepath='leads.csv'):
    """Load the leads dataset and perform initial cleaning."""
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values per column:\n{df.isnull().sum()}")

    # Drop PII
    drop_cols = ['email', 'phone_number', 'last_contact_date']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    print(f"Dropped columns: {[c for c in drop_cols if c in df.columns]}")

    # Fill missing numericals with median
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    # Fill missing categoricals with mode
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'intent_binary' and df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)

    print("Data cleaning completed.")
    return df

def prepare_features(df):
    """Prepare features for model training."""
    print("Preparing features...")
    feature_cols = [
        'credit_score',
        'age_group',
        'family_background',
        'income',
        'education_level',
        'employment_type',
        'website_engagement_score'
    ]
    if any(c not in df.columns for c in feature_cols + ['intent_binary']):
        missing = [c for c in feature_cols + ['intent_binary'] if c not in df.columns]
        raise ValueError(f"Missing required columns: {missing}")

    # Target
    y = df['intent_binary'].copy()
    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)

    # Features
    X = df[feature_cols].copy()
    encoders = {}

    # Label-encode categoricals
    for col in ['age_group','family_background','education_level','employment_type']:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    # Scale numerics
    scaler = StandardScaler()
    X[['credit_score','income','website_engagement_score']] = scaler.fit_transform(
        X[['credit_score','income','website_engagement_score']]
    )
    encoders['scaler'] = scaler

    print(f"✅ Final features for training: {list(X.columns)}")
    return X, y, feature_cols, encoders

def train_model(X, y):
    """Train and evaluate the classifier."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    print("\nROC-AUC:", roc_auc_score(y_test, y_proba))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    return model

def save_model_and_metadata(model, encoders, feature_names):
    """Bundle and save the model plus preprocessing."""
    pkg = {'model': model, 'encoders': encoders, 'feature_names': feature_names}
    joblib.dump(pkg, 'model.pkl')
    print("✅ model.pkl saved with features:", feature_names)

def main():
    df = load_and_clean_data()
    X, y, feature_names, encoders = prepare_features(df)
    model = train_model(X, y)
    save_model_and_metadata(model, encoders, feature_names)

if __name__ == "__main__":
    main()
