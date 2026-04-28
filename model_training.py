import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import accuracy_score, r2_score, silhouette_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from data_preprocessing import preprocessing_pipeline

def train_model(X, y, task: str):
    results = {}

    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    numeric_cols = X.select_dtypes(include="number").columns.tolist()

    if task == "Classification":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            "RandomForest": Pipeline([
                ("preprocess", preprocessing_pipeline(categorical_cols, numeric_cols)),
                ("smote", SMOTE()),
                ("model", RandomForestClassifier(n_estimators=100, random_state=42))
            ]),
            "GradientBoosting": Pipeline([
                ("preprocess", preprocessing_pipeline(categorical_cols, numeric_cols)),
                ("smote", SMOTE()),
                ("model", GradientBoostingClassifier(n_estimators=100, random_state=42))
            ]),
        }
        best_name, best_model, best_score = None, None, -1
        for name, model in models.items():
            model.fit(X_train, y_train)
            score = accuracy_score(y_test, model.predict(X_test))
            if score > best_score:
                best_name, best_model, best_score = name, model, score

        results["best_model_name"] = best_name
        results["model"] = best_model
        results["X_test"] = X_test
        results["y_test"] = y_test

    elif task == "Regression":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            "LinearRegression": Pipeline([
                ("preprocess", preprocessing_pipeline(categorical_cols, numeric_cols)),
                ("model", LinearRegression())
            ]),
            "Ridge": Pipeline([
                ("preprocess", preprocessing_pipeline(categorical_cols, numeric_cols)),
                ("model", Ridge(alpha=1.0))
            ]),
        }
        best_name, best_model, best_score = None, None, -np.inf
        for name, model in models.items():
            model.fit(X_train, y_train)
            score = r2_score(y_test, model.predict(X_test))
            if score > best_score:
                best_name, best_model, best_score = name, model, score

        results["best_model_name"] = best_name
        results["model"] = best_model
        results["X_test"] = X_test
        results["y_test"] = y_test

    elif task == "Clustering":
        models = {
            "KMeans": Pipeline([
                ("preprocess", preprocessing_pipeline(categorical_cols, numeric_cols)),
                ("model", KMeans(n_clusters=3, random_state=42, n_init=10))
            ]),
            "Agglomerative": Pipeline([
                ("preprocess", preprocessing_pipeline(categorical_cols, numeric_cols)),
                ("model", AgglomerativeClustering(n_clusters=3))
            ]),
        }
        best_name, best_model, best_score = None, None, -1
        for name, model in models.items():
            labels = model.fit_predict(X)
            X_transformed = model.named_steps["preprocess"].transform(X)
            score = silhouette_score(X_transformed, labels)
            if score > best_score:
                best_name, best_model, best_score = name, model, score

        results["best_model_name"] = best_name
        results["model"] = best_model
        results["labels"] = best_model.fit_predict(X)

    print(f"Best model: {results['best_model_name']}")
    return results