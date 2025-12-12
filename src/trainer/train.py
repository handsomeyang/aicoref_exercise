import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from scipy.stats import uniform, randint
import joblib
from utils import get_data_dir, get_artifacts_dir


def main() -> None:
    df = pd.read_csv(get_data_dir() / "dataset.csv", sep=";")

    numerical_features = [
        "age",
        "balance",
        "day",
        "duration",
        "campaign",
        "pdays",
        "previous",
    ]
    categorical_features = [
        "job",
        "marital",
        "education",
        "contact",
        "month",
        "poutcome",
    ]
    binary_features = ["default", "housing", "loan"]

    for bf in binary_features + ["y"]:
        df[bf] = df[bf].map({"yes": 1, "no": 0})

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numerical_features,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ],
        remainder="passthrough",
    )

    positive_count = df["y"].sum()
    negative_count = len(df) - positive_count
    scale_weight = negative_count / positive_count
    print(positive_count, negative_count, scale_weight)

    xgb_model = XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", random_state=42
    )

    cv_pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", xgb_model)]
    )

    X = df.drop(columns=["y"])
    y = df["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    param_distributions = {
        "classifier__n_estimators": randint(100, 1000),
        "classifier__learning_rate": uniform(loc=0.01, scale=0.29),  # Range 0.01 to 0.3
        "classifier__max_depth": randint(3, 10),
        "classifier__subsample": uniform(loc=0.6, scale=0.4),  # Range 0.6 to 1.0
        "classifier__colsample_bytree": uniform(loc=0.6, scale=0.4),  # Range 0.6 to 1.0
        "classifier__gamma": uniform(loc=0, scale=0.5),
        "classifier__scale_pos_weight": uniform(
            loc=scale_weight * 0.9, scale=scale_weight * 0.2
        ),
    }

    k_folds = 5
    cv_strategy = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    n_iter = 10

    random_search = RandomizedSearchCV(
        estimator=cv_pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv_strategy,
        verbose=1,
        random_state=42,
        n_jobs=-1,
    )

    random_search.fit(X_train, y_train)

    best_roc_auc_score = random_search.best_score_
    best_params = random_search.best_params_
    best_pipeline = random_search.best_estimator_

    print(best_roc_auc_score)
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]

    final_roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(final_roc_auc)

    joblib.dump(best_pipeline, get_artifacts_dir() / "best_ml_pipeline.joblib")


if __name__ == "__main__":
    main()
