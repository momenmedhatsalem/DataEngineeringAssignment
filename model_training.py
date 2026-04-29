import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import accuracy_score, r2_score, silhouette_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
        # Evaluation
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()
        results["metrics"] = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "confusion_matrix": cm,
            "best_model": best_name
        }

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
        # Evaluation
        y_pred = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results["metrics"] = {
            "mae": float(mae),
            "mse": float(mse),
            "r2": float(r2),
            "best_model": best_name
        }

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
        labels = best_model.fit_predict(X)
        # compute silhouette on transformed features
        X_transformed = best_model.named_steps["preprocess"].transform(X)
        sil = silhouette_score(X_transformed, labels)
        results["metrics"] = {
            "silhouette": float(sil),
            "best_model": best_name
        }
        results["labels"] = labels

    print(f"Best model: {results['best_model_name']}")
    return results