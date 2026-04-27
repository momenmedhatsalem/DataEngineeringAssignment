import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import accuracy_score, r2_score, silhouette_score

MODEL_PATH = "final_model.joblib"

def train_model(X, y, task: str):
    results = {}

    if task == "Classification":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
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
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
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
            "KMeans": KMeans(n_clusters=3, random_state=42, n_init=10),
            "Agglomerative": AgglomerativeClustering(n_clusters=3),
        }
        best_name, best_model, best_score = None, None, -1
        for name, model in models.items():
            labels = model.fit_predict(X)
            score = silhouette_score(X, labels)
            if score > best_score:
                best_name, best_model, best_score = name, model, score

        results["best_model_name"] = best_name
        results["model"] = best_model
        results["labels"] = best_model.fit_predict(X)

    # Serialize best model + pipeline together
    pipeline = joblib.load("preprocessing_pipeline.joblib")
    joblib.dump({"model": results["model"], "pipeline": pipeline}, MODEL_PATH)
    print(f"Best model: {results['best_model_name']}")
    return results