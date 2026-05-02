import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import accuracy_score, r2_score, silhouette_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from data_preprocessing import build_model_pipeline, data_analysis

def train_model(X, y, task: str):
    results = {"le": None}
    if task != "Clustering":
        # Drop rows where target is NaN before splitting
        mask = pd.Series(y).notna().values
        X, y = X[mask], y[mask]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        results["analysis"] = data_analysis(X_train, y_train, task)
    else:
        results["analysis"] = data_analysis(X, None, task)

    if task == "Classification":
        if (results["analysis"]["y_needs_encoding"]):
            results["le"] = LabelEncoder()
            y_train = results["le"].fit_transform(y_train)
            y_test = results["le"].transform(y_test)

        # cv must not exceed min_class_count — if a class has 1 sample it cannot be split at all
        # so skip GridSearchCV entirely and just fit directly
        safe_cv = min(5, results["analysis"]["min_class_count"])
        use_grid = safe_cv >= 2

        param_grids = {
            "RandomForest": {
                "model__max_depth": [None, 10, 20]
            },
            "GradientBoosting": {
                "model__learning_rate": [0.05, 0.1]
            }
        } if use_grid else {"RandomForest": None, "GradientBoosting": None}

        models = {
            "RandomForest": build_model_pipeline(
                RandomForestClassifier(n_estimators=200, random_state=42),
                results["analysis"],
                param_grids["RandomForest"],
                cv=safe_cv
            ),
            "GradientBoosting": build_model_pipeline(
                GradientBoostingClassifier(n_estimators=200, random_state=42),
                results["analysis"],
                param_grids["GradientBoosting"],
                cv=safe_cv
            ),
        }
        best_name, best_model, best_score = None, None, -1
        for name, model in models.items():
            model.fit(X_train, y_train)
            score = accuracy_score(y_test, model.predict(X_test))
            if score > best_score:
                best_name, best_model, best_score = name, model, score

        results["best_model_name"] = best_name
        results["model"] = best_model
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
        safe_cv = min(5, len(X_train) // 10)
        use_grid = safe_cv >= 2

        param_grids = {
            "Ridge": {"model__regressor__alpha": [0.1, 1.0, 10.0, 100.0]},
            "RandomForestRegressor": {"model__max_depth": [None, 10, 20]},
            "GradientBoostingRegressor": {"model__learning_rate": [0.05, 0.1]}
        } if use_grid else {
            "Ridge": None, "RandomForestRegressor": None, "GradientBoostingRegressor": None
        }

        ridge_scaled = TransformedTargetRegressor(
            regressor=Ridge(alpha=1.0),
            transformer=StandardScaler()
        )

        models = {
            "Ridge": build_model_pipeline(ridge_scaled, results["analysis"], param_grids["Ridge"], cv=safe_cv),
            "RandomForestRegressor": build_model_pipeline(RandomForestRegressor(n_estimators=200, random_state=42), results["analysis"], param_grids["RandomForestRegressor"], cv=safe_cv),
            "GradientBoostingRegressor": build_model_pipeline(GradientBoostingRegressor(n_estimators=200, random_state=42), results["analysis"], param_grids["GradientBoostingRegressor"], cv=safe_cv),
        }
        best_name, best_model, best_score = None, None, -np.inf
        for name, model in models.items():
            model.fit(X_train, y_train)
            score = r2_score(y_test, model.predict(X_test))
            if score > best_score:
                best_name, best_model, best_score = name, model, score

        results["best_model_name"] = best_name
        results["model"] = best_model
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
        analysis = results["analysis"]

        # Preprocess X once — all clustering happens in transformed space
        from sklearn.pipeline import Pipeline as SkPipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

        numeric_pipe = SkPipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
        low_cat_pipe = SkPipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
        high_cat_pipe = SkPipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))])
        preprocess = ColumnTransformer([
            ("num", numeric_pipe, analysis["numeric_cols"]),
            ("low_cat", low_cat_pipe, analysis["low_count"]),
            ("high_cat", high_cat_pipe, analysis["high_count"]),
        ])
        X_transformed = preprocess.fit_transform(X)

        best_name, best_labels, best_score = None, None, -1

        # KMeans + Agglomerative over k=2..10
        for k in range(2, 11):
            for name, algo in [
                (f"KMeans(k={k})", KMeans(n_clusters=k, random_state=42, n_init=10)),
                (f"Agglomerative(k={k})", AgglomerativeClustering(n_clusters=k)),
            ]:
                labels = algo.fit_predict(X_transformed)
                score = silhouette_score(X_transformed, labels)
                if score > best_score:
                    best_name, best_labels, best_score = name, labels, score

        # DBSCAN — auto-discovers k, robust to outliers
        # Try a small range of eps values; skip if all points are noise (label=-1 only)
        for eps in [0.3, 0.5, 0.8, 1.0, 1.5]:
            db = DBSCAN(eps=eps, min_samples=5)
            labels = db.fit_predict(X_transformed)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_ratio = (labels == -1).sum() / len(labels)
            if n_clusters >= 2 and noise_ratio < 0.5:
                score = silhouette_score(X_transformed, labels)
                if score > best_score:
                    best_name, best_labels, best_score = f"DBSCAN(eps={eps})", labels, score

        # Build a serialisable pipeline using the best k/algorithm for KMeans/Agglomerative,
        # or a plain KMeans fallback for DBSCAN (DBSCAN is not re-fittable on new data)
        if best_name and best_name.startswith("KMeans"):
            k = int(best_name.split("=")[1].rstrip(")"))
            best_model = build_model_pipeline(KMeans(n_clusters=k, random_state=42, n_init=10), analysis)
        elif best_name and best_name.startswith("Agglomerative"):
            k = int(best_name.split("=")[1].rstrip(")"))
            best_model = build_model_pipeline(AgglomerativeClustering(n_clusters=k), analysis)
        else:
            # DBSCAN won — store it directly; wrap in a simple object for joblib serialisation
            eps_val = float(best_name.split("=")[1].rstrip(")"))
            best_model = build_model_pipeline(DBSCAN(eps=eps_val, min_samples=5), analysis)

        results["best_model_name"] = best_name
        results["model"] = best_model
        results["metrics"] = {
            "silhouette": float(best_score),
            "best_model": best_name
        }
        results["labels"] = best_labels

    print(f"Best model: {results['best_model_name']}")
    return results
