import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    ConfusionMatrixDisplay
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "APL_Logistics.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH, encoding="latin1")

target = "Late_delivery_risk"

drop_cols = [
    target, "Days for shipping (real)",
    "Days for shipment (scheduled)", "Delivery Status",
    "Order Status", "Customer Fname", "Customer Lname",
    "Customer Street", "Customer Zipcode", "Customer Id",
    "Order Customer Id", "Customer City", "Customer State",
    "Customer Country", "Order City", "Order State",
    "Order Country", "Latitude", "Longitude", "Type"
]

X = df.drop(columns=drop_cols, errors="ignore")
y = df[target].astype(int)

numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ]),
    "Random Forest": Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=200,
            max_depth=18,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        ))
    ])
}

results = []
fitted_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1 Score": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, prob)
    })

    fitted_models[name] = model

    print(f"\n{name} Classification Report")
    print(classification_report(y_test, pred, zero_division=0))

    ConfusionMatrixDisplay.from_predictions(y_test, pred)
    plt.title(f"{name} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            name.lower().replace(" ", "_") + "_confusion_matrix.png"
        )
    )
    plt.close()

results_df = pd.DataFrame(results).sort_values(
    "ROC-AUC", ascending=False
)

results_df.to_csv(
    os.path.join(OUTPUT_DIR, "model_comparison.csv"),
    index=False
)

print("\nModel Comparison")
print(results_df.round(4).to_string(index=False))

best_name = results_df.iloc[0]["Model"]
best_model = fitted_models[best_name]

joblib.dump(best_model, os.path.join(OUTPUT_DIR, "best_model.joblib"))
joblib.dump(X.columns.tolist(), os.path.join(OUTPUT_DIR, "feature_columns.joblib"))

rf_model = fitted_models["Random Forest"]
feature_names = rf_model.named_steps["preprocessor"].get_feature_names_out()
importances = rf_model.named_steps["model"].feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values("Importance", ascending=False)

importance_df.to_csv(
    os.path.join(OUTPUT_DIR, "feature_importance.csv"),
    index=False
)

top_features = importance_df.head(20).sort_values("Importance")

plt.figure(figsize=(11, 8))
plt.barh(top_features["Feature"], top_features["Importance"])
plt.xlabel("Feature Importance")
plt.title("Random Forest - Top 20 Feature Importance")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "feature_importance.png"),
    dpi=160
)
plt.close()

metadata = {
    "dataset_rows": int(df.shape[0]),
    "dataset_columns": int(df.shape[1]),
    "training_rows": int(len(X_train)),
    "testing_rows": int(len(X_test)),
    "best_model": best_name,
    "features": X.columns.tolist()
}

with open(
    os.path.join(OUTPUT_DIR, "metadata.json"),
    "w",
    encoding="utf-8"
) as f:
    json.dump(metadata, f, indent=2)

print(f"\nBest model by ROC-AUC: {best_name}")
print(f"Artifacts saved in: {OUTPUT_DIR}")
