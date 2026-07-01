import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "heart_data.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

data = pd.read_csv(DATA_PATH)
X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight="balanced"
    ),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(probability=True, class_weight="balanced", random_state=42))
    ])
}

results = []
best_model = None
best_model_name = None
best_f1 = -1
best_predictions = None

for model_name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="f1")

    result = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1_score": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "cv_f1_mean": cv_scores.mean(),
        "cv_f1_std": cv_scores.std()
    }
    results.append(result)

    if result["f1_score"] > best_f1:
        best_f1 = result["f1_score"]
        best_model = model
        best_model_name = model_name
        best_predictions = predictions

results_df = pd.DataFrame(results).sort_values("f1_score", ascending=False)
results_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

joblib.dump(best_model, MODEL_DIR / "heart_disease_model.pkl")

summary = {
    "best_model": best_model_name,
    "best_f1_score": best_f1,
    "confusion_matrix": confusion_matrix(y_test, best_predictions).tolist(),
    "features": X.columns.tolist()
}

with open(REPORT_DIR / "best_model_summary.json", "w") as file:
    json.dump(summary, file, indent=2)

print("Training completed.")
print(results_df)
print(f"Best model saved: {best_model_name}")