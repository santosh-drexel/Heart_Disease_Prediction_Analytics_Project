"""Leakage-aware preprocessing, model training and evaluation."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             ConfusionMatrixDisplay, RocCurveDisplay)
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, FunctionTransformer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

TARGET_COLUMN = 'output'
# Original notebook removed these three features based on exploratory correlation.
# Keep this an explicit, reviewable experiment choice, not a proven optimal subset.
OPTIONAL_DROPPED_FEATURES = ['fbs', 'chol', 'restecg']
NUMERIC_FEATURES = ['age', 'trtbps', 'thalachh', 'oldpeak']
CATEGORICAL_FEATURES = ['sex', 'cp', 'exng', 'slp', 'caa', 'thall']


class HeartFeatureCleaner(BaseEstimator, TransformerMixin):
    """Learn training-only clipping bounds; treat thall=0 as unknown."""
    def __init__(self, clip_quantile=0.99):
        self.clip_quantile = clip_quantile

    def fit(self, X, y=None):
        data = X.copy()
        # Upper-tail winsorization: estimate quantile caps from training folds only.
        self.upper_bounds_ = {
            column: float(data[column].quantile(self.clip_quantile))
            for column in ['trtbps', 'oldpeak']
        }
        return self

    def transform(self, X):
        data = X.copy()
        data['thall'] = data['thall'].replace(0, np.nan)
        for column, upper_bound in self.upper_bounds_.items():
            data[column] = data[column].clip(upper=upper_bound)
        # sqrt is defined at zero; this preserves the original notebook's
        # square-root transformation without introducing log(0).
        data['oldpeak'] = np.sqrt(data['oldpeak'].clip(lower=0))
        return data


def split_features_target(data, test_size=0.20, random_state=42):
    """Create a stratified, reproducible holdout set before fitting transforms."""
    features = data.drop(columns=[TARGET_COLUMN])
    labels = data[TARGET_COLUMN]
    return train_test_split(features, labels, test_size=test_size,
                            stratify=labels, random_state=random_state)


def build_preprocessor():
    """Encode categories and robust-scale numeric variables within each fit."""
    numeric_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', RobustScaler())
    ])
    categorical_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('encode', OneHotEncoder(handle_unknown='ignore'))
    ])
    return ColumnTransformer([
        ('numeric', numeric_pipeline, NUMERIC_FEATURES),
        ('categorical', categorical_pipeline, CATEGORICAL_FEATURES)
    ])


def build_model_pipeline(estimator):
    """Keep every learned preprocessing step inside CV and model fitting."""
    return Pipeline([
        ('clean', HeartFeatureCleaner()),
        ('preprocess', build_preprocessor()),
        ('model', estimator)
    ])


def get_candidate_models(random_state=42):
    """Return the four algorithms used in the original notebook."""
    return {
        'Logistic Regression': LogisticRegression(max_iter=2000, random_state=random_state),
        'Decision Tree': DecisionTreeClassifier(random_state=random_state),
        'Support Vector Machine': SVC(probability=True, random_state=random_state),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=random_state)
    }


def evaluate_model(fitted_pipeline, test_features, test_labels):
    """Compute holdout metrics from predictions made by the supplied model."""
    predictions = fitted_pipeline.predict(test_features)
    metrics = {
        'accuracy': accuracy_score(test_labels, predictions),
        'precision': precision_score(test_labels, predictions, zero_division=0),
        'recall': recall_score(test_labels, predictions, zero_division=0),
        'f1': f1_score(test_labels, predictions, zero_division=0)
    }
    if hasattr(fitted_pipeline, 'predict_proba'):
        scores = fitted_pipeline.predict_proba(test_features)[:, 1]
    elif hasattr(fitted_pipeline, 'decision_function'):
        scores = fitted_pipeline.decision_function(test_features)
    else:
        scores = None
    metrics['roc_auc'] = roc_auc_score(test_labels, scores) if scores is not None else np.nan
    return metrics


def compare_models(train_features, test_features, train_labels, test_labels,
                   cv_folds=5, random_state=42):
    """Cross-validate on training data, fit, then evaluate once on holdout."""
    fitted_models = {}
    records = []
    for name, estimator in get_candidate_models(random_state).items():
        pipeline = build_model_pipeline(estimator)
        validation_scores = cross_val_score(pipeline, train_features, train_labels,
                                            cv=cv_folds, scoring='f1')
        pipeline.fit(train_features, train_labels)
        holdout_metrics = evaluate_model(pipeline, test_features, test_labels)
        records.append({'model': name, 'cv_f1_mean': validation_scores.mean(),
                        'cv_f1_std': validation_scores.std(), **holdout_metrics})
        fitted_models[name] = pipeline
    return pd.DataFrame(records).sort_values('cv_f1_mean', ascending=False).reset_index(drop=True), fitted_models


def tune_logistic_regression(train_features, train_labels, cv_folds=5):
    """Tune only on training folds using compatible solver/penalty choices."""
    search = GridSearchCV(
        build_model_pipeline(LogisticRegression(max_iter=2000)),
        param_grid={'model__solver': ['liblinear'],
                    'model__l1_ratio': [0.0, 1.0],
                    'model__C': [0.01, 0.1, 1, 10]},
        cv=cv_folds, scoring='f1', n_jobs=-1
    )
    search.fit(train_features, train_labels)
    return search


def plot_model_diagnostics(fitted_pipeline, test_features, test_labels):
    """Plot confusion matrix and ROC curve on the holdout set."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ConfusionMatrixDisplay.from_estimator(fitted_pipeline, test_features, test_labels, ax=axes[0])
    RocCurveDisplay.from_estimator(fitted_pipeline, test_features, test_labels, ax=axes[1])
    fig.tight_layout()
    return fig


def show_training_winsorization(train_features, fitted_cleaner):
    """Return original and training-fitted winsorized features for EDA plots.

    Only call with training features; do not refit on holdout observations.
    """
    comparison = train_features.copy()
    for feature, upper_bound in fitted_cleaner.upper_bounds_.items():
        comparison[f'{feature}_winsorized'] = comparison[feature].clip(upper=upper_bound)
    comparison['oldpeak_winsorized_sqrt'] = np.sqrt(
        comparison['oldpeak_winsorized'].clip(lower=0))
    return comparison
