import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Load the trained model package
model_package = joblib.load("model.pkl")

# Unpack components
model = model_package['model']
encoders = model_package['encoders']
feature_names = model_package['feature_names']

# Load your original dataset
df = pd.read_csv("leads.csv")

# Drop non-feature columns (as done during training)
df = df.drop(columns=["email", "phone_number", "last_contact_date", "intent_score"])

# Target column
y = df["intent_binary"]
X = df.drop(columns=["intent_binary"])

# Handle categorical encoding (same as during training)
categorical_features = ['age_group', 'family_background', 'education_level', 'employment_type']
for col in categorical_features:
    if col in encoders:
        valid_classes = set(encoders[col].classes_)
        def map_or_first(val):
            sval = str(val)
            return sval if sval in valid_classes else list(valid_classes)[0]
        X[col] = X[col].fillna(list(valid_classes)[0]).astype(str).apply(map_or_first)
        X[col] = encoders[col].transform(X[col])

# Scale numeric features
numerical_features = ['credit_score', 'income', 'website_engagement_score']
scaler = encoders['scaler']
X[numerical_features] = scaler.transform(X[numerical_features])

# Align feature order
X = X[feature_names]

# Predict
y_pred = model.predict(X)

# Metrics
print("Model Accuracy:", round(accuracy_score(y, y_pred), 4))
print("Precision:", round(precision_score(y, y_pred), 4))
print("Recall:", round(recall_score(y, y_pred), 4))
print("F1 Score:", round(f1_score(y, y_pred), 4))
print("\nClassification Report:")
print(classification_report(y, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y, y_pred))
