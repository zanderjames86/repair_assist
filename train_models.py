"""
Train the two classical ML models used by the app.

ML model 1 - Fault classifier (text classification)
    TF-IDF features from the technician's note -> fault category.
    Compares Naive Bayes, Logistic Regression and Linear SVM with
    5-fold cross-validation, then keeps Logistic Regression (it gives
    probabilities and explainable word weights).

ML model 2 - Repair outcome predictor (tabular)
    Board type, age, previous repairs, fault category ->
      a) Repairable vs BER (beyond economic repair)   [Random Forest classifier]
      b) Estimated repair hours                        [Random Forest regressor]

Outputs (in models/):
    fault_classifier.joblib, outcome_model.joblib, hours_model.joblib,
    metrics.json  (everything the Model Evaluation tab shows)

Runs in a few seconds on a laptop CPU:  python train_models.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, r2_score, roc_auc_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "repair_records.csv"
MODELS = ROOT / "models"
SEED = 42

TAB_FEATURES = ["board_type", "fault_category", "age_years", "previous_repairs"]
CAT_FEATURES = ["board_type", "fault_category"]
NUM_FEATURES = ["age_years", "previous_repairs"]


# Filler words that carry no fault information. "no" and "not" are kept
# on purpose: "no continuity" / "not responding" matter.
STOP_WORDS = ["a", "an", "the", "is", "are", "was", "on", "to", "of", "and",
              "in", "at", "near", "from", "with", "for", "it", "this", "will",
              "be", "now", "done", "board", "unit", "checked", "bench"]


def tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2,
                           stop_words=STOP_WORDS)


def train_fault_classifier(df):
    X, y = df["note"], df["fault_category"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    candidates = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=2000, C=5),
        "Linear SVM": LinearSVC(C=1),
    }
    comparison = []
    for name, clf in candidates.items():
        pipe = Pipeline([("tfidf", tfidf()), ("clf", clf)])
        cv = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="f1_macro")
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        comparison.append({
            "model": name,
            "cv_f1_mean": round(float(cv.mean()), 3),
            "cv_f1_std": round(float(cv.std()), 3),
            "test_accuracy": round(float(accuracy_score(y_te, pred)), 3),
            "test_f1_macro": round(float(f1_score(y_te, pred, average="macro")), 3),
        })

    # Keep Logistic Regression: probabilities + explainable coefficients
    final = Pipeline([("tfidf", tfidf()),
                      ("clf", LogisticRegression(max_iter=2000, C=5))])
    final.fit(X_tr, y_tr)
    pred = final.predict(X_te)
    labels = list(final.classes_)
    cm = confusion_matrix(y_te, pred, labels=labels)

    # Top keywords per fault class (what the model learned)
    vocab = np.array(final.named_steps["tfidf"].get_feature_names_out())
    coefs = final.named_steps["clf"].coef_
    keywords = {lab: vocab[np.argsort(coefs[i])[::-1][:8]].tolist()
                for i, lab in enumerate(labels)}

    # Refit on all data for the app
    final.fit(X, y)
    joblib.dump(final, MODELS / "fault_classifier.joblib")

    return {
        "comparison": comparison,
        "chosen": "Logistic Regression",
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "test_size": int(len(y_te)),
        "top_keywords": keywords,
    }


def tabular_preprocessor():
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ("num", "passthrough", NUM_FEATURES),
    ])


def grouped_importance(pipe):
    """Sum one-hot importances back to the original 4 features."""
    names = pipe.named_steps["prep"].get_feature_names_out()
    imps = pipe.named_steps["model"].feature_importances_
    out = {f: 0.0 for f in TAB_FEATURES}
    for n, v in zip(names, imps):
        for f in TAB_FEATURES:
            if f in n:
                out[f] += float(v)
    return {k: round(v, 3) for k, v in sorted(out.items(), key=lambda x: -x[1])}


def train_outcome_models(df):
    X = df[TAB_FEATURES]
    y_cls = (df["outcome"] == "BER").astype(int)
    y_reg = df["repair_hours"]
    X_tr, X_te, yc_tr, yc_te, yr_tr, yr_te = train_test_split(
        X, y_cls, y_reg, test_size=0.2, stratify=y_cls, random_state=SEED)

    # a) Repairable vs BER
    clf = Pipeline([("prep", tabular_preprocessor()),
                    ("model", RandomForestClassifier(
                        n_estimators=200, max_depth=6, min_samples_leaf=5,
                        class_weight="balanced", random_state=SEED))])
    cv = cross_val_score(clf, X_tr, yc_tr, cv=5, scoring="roc_auc")
    clf.fit(X_tr, yc_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    base = DummyClassifier(strategy="most_frequent").fit(X_tr, yc_tr)

    # b) Repair hours
    reg = Pipeline([("prep", tabular_preprocessor()),
                    ("model", RandomForestRegressor(
                        n_estimators=200, max_depth=8, min_samples_leaf=5,
                        random_state=SEED))])
    reg.fit(X_tr, yr_tr)
    hours_pred = reg.predict(X_te)
    base_reg = DummyRegressor().fit(X_tr, yr_tr)

    metrics = {
        "outcome": {
            "cv_roc_auc_mean": round(float(cv.mean()), 3),
            "cv_roc_auc_std": round(float(cv.std()), 3),
            "test_accuracy": round(float(accuracy_score(yc_te, pred)), 3),
            "test_f1_ber": round(float(f1_score(yc_te, pred)), 3),
            "test_roc_auc": round(float(roc_auc_score(yc_te, proba)), 3),
            "baseline_accuracy": round(float(base.score(X_te, yc_te)), 3),
            "confusion_matrix": confusion_matrix(yc_te, pred).tolist(),
            "labels": ["Repairable", "BER"],
            "feature_importance": grouped_importance(clf),
        },
        "hours": {
            "test_mae": round(float(mean_absolute_error(yr_te, hours_pred)), 2),
            "test_r2": round(float(r2_score(yr_te, hours_pred)), 3),
            "baseline_mae": round(float(mean_absolute_error(
                yr_te, base_reg.predict(X_te))), 2),
            "feature_importance": grouped_importance(reg),
        },
        "ber_rate": round(float(y_cls.mean()), 3),
    }

    clf.fit(X, y_cls)
    reg.fit(X, y_reg)
    joblib.dump(clf, MODELS / "outcome_model.joblib")
    joblib.dump(reg, MODELS / "hours_model.joblib")
    return metrics


def main():
    MODELS.mkdir(exist_ok=True)
    df = pd.read_csv(DATA)
    fault = train_fault_classifier(df)
    outcome = train_outcome_models(df)
    import sklearn
    metrics = {"n_records": int(len(df)), "sklearn_version": sklearn.__version__,
               "fault_classifier": fault, **outcome}
    (MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("Fault classifier comparison:")
    print(pd.DataFrame(fault["comparison"]).to_string(index=False))
    o, h = outcome["outcome"], outcome["hours"]
    print(f"\nOutcome RF: acc {o['test_accuracy']} (baseline {o['baseline_accuracy']}), "
          f"ROC AUC {o['test_roc_auc']}, F1(BER) {o['test_f1_ber']}")
    print(f"Hours RF:   MAE {h['test_mae']} h (baseline {h['baseline_mae']} h), R2 {h['test_r2']}")
    print("Importance (outcome):", o["feature_importance"])


if __name__ == "__main__":
    main()
