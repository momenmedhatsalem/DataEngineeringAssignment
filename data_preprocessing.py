from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
import pandas as pd

class ModelWithEncoder:
    def __init__(self, pipeline, y_needs_encoding=False):
        self.pipeline = pipeline
        self.y_needs_encoding = y_needs_encoding
        self.label_encoder = None

    def encode_y(self, y):
        if not self.y_needs_encoding:
            return y

        self.label_encoder = LabelEncoder()
        return self.label_encoder.fit_transform(y)

    def decode_y(self, y):
        if not self.y_needs_encoding:
            return y

        return self.label_encoder.inverse_transform(y)

    def fit(self, X, y):
        y = self.encode_y(y)
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        preds = self.pipeline.predict(X)
        return self.decode_y(preds)

    def fit_predict(self, X):
        preds = self.pipeline.fit_predict(X)
        return preds

def data_analysis(X, y, task):
    analysis = {}

    missing_ratio = X.isnull().mean()
    drop_missing = missing_ratio[missing_ratio > 0.7].index.tolist()

    unique_ratio = X.nunique() / len(X)
    drop_unique = unique_ratio[unique_ratio > 0.95].index.tolist()

    initial_drop = list(set(drop_missing + drop_unique))
    X_reduced = X.drop(columns=initial_drop, errors="ignore")

    categorical_cols = X_reduced.select_dtypes(include="object").columns.tolist()
    numeric_cols = X_reduced.select_dtypes(include="number").columns.tolist()

    cat_unique_counts = {col: X_reduced[col].nunique() for col in categorical_cols}
    low_count = [c for c, n in cat_unique_counts.items() if n <= 15]
    high_count = [c for c, n in cat_unique_counts.items() if n > 15]


    drop_low_variance = []
    if numeric_cols:
        variances = X_reduced[numeric_cols].var()
        drop_low_variance = variances[variances < 0.01].index.tolist()

    all_drops = list(set(initial_drop + drop_low_variance))

    # Remove dropped columns from feature lists
    categorical_cols = [c for c in categorical_cols if c not in all_drops]
    numeric_cols = [c for c in numeric_cols if c not in all_drops]

    low_count = [c for c in low_count if c not in all_drops]
    high_count = [c for c in high_count if c not in all_drops]

    cols_with_missing = X_reduced.columns[X_reduced.isnull().any()].tolist()

    y_series = pd.Series(y)
    y_needs_encoding = y_series.dtype == "object"


    is_imbalanced = False
    dist = y_series.value_counts(normalize=True)
    if dist.min() < 0.3:
        is_imbalanced = True

    analysis.update({
        "drop_missing": drop_missing,
        "drop_unique": drop_unique,
        "drop_low_variance": drop_low_variance,
        "drop_cols": all_drops,

        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "low_count": low_count,
        "high_count": high_count,

        "cols_with_missing": cols_with_missing,
        "y_needs_encoding": y_needs_encoding,
        "is_imbalanced": is_imbalanced,
        "task": task
    })

    return analysis

def build_model_pipeline(model, analysis, param_grid=None):
    numeric_cols = analysis["numeric_cols"]
    low_cat = analysis["low_count"]
    high_cat = analysis["high_count"]

    task = analysis["task"]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    low_cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    high_cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    preprocess = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("low_cat", low_cat_pipe, low_cat),
        ("high_cat", high_cat_pipe, high_cat)
    ])

    steps = [("preprocess", preprocess)]

    if task == "Classification" and analysis["is_imbalanced"]:
        steps.append(("smote", SMOTE()))

    steps.append(("model", model))

    pipeline = Pipeline(steps)

    if param_grid:
        pipeline = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=3,
            scoring="accuracy" if task == "Classification" else "r2",
            n_jobs=-1
        )

    return ModelWithEncoder(
        pipeline=pipeline,
        y_needs_encoding=analysis["y_needs_encoding"]
    )