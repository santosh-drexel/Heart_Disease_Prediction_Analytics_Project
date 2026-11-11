"""Dataset inspection and exploratory analysis for the heart classification project."""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

TARGET_COLUMN = 'output'
NUMERIC_FEATURES = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']
CATEGORICAL_FEATURES = ['sex', 'cp', 'fbs', 'restecg', 'exng', 'slp', 'caa', 'thall']


def load_heart_data(path='heart.csv'):
    """Read the original CSV without changing its contents."""
    data = pd.read_csv(Path(path))
    required = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN])
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f'Missing expected columns: {sorted(missing)}')
    return data


def summarize_data(data):
    """Return descriptive statistics, missing counts, and unique counts."""
    return {'shape': data.shape, 'statistics': data.describe().T,
            'missing_values': data.isna().sum(), 'unique_values': data.nunique(),
            'target_distribution': data[TARGET_COLUMN].value_counts().sort_index()}


def identify_iqr_outliers(data, column, multiplier=1.5):
    """Find potential outlier observations; do not delete them."""
    first_quartile, third_quartile = data[column].quantile([0.25, 0.75])
    spread = third_quartile - first_quartile
    lower = first_quartile - multiplier * spread
    upper = third_quartile + multiplier * spread
    return data.loc[(data[column] < lower) | (data[column] > upper)]


def plot_target_distribution(data):
    """Plot observed class counts."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=data, x=TARGET_COLUMN, ax=ax)
    ax.set(title='Recorded target distribution', xlabel='Target label', ylabel='Observations')
    fig.tight_layout()
    return fig


def plot_numeric_distributions(data):
    """Show numeric feature histograms."""
    axes = data[NUMERIC_FEATURES].hist(figsize=(12, 8), bins=20, edgecolor='black')
    figure = axes.ravel()[0].figure
    figure.suptitle('Numeric feature distributions')
    figure.tight_layout()
    return figure


def plot_numeric_by_target(data):
    """Compare numerical features across target labels."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, column in zip(axes.flat, NUMERIC_FEATURES):
        sns.boxplot(data=data, x=TARGET_COLUMN, y=column, ax=ax)
        ax.set_title(column)
    for ax in list(axes.flat)[len(NUMERIC_FEATURES):]:
        ax.axis('off')
    fig.tight_layout()
    return fig


def plot_categorical_by_target(data):
    """Show counts for categorical variables grouped by target."""
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    for ax, column in zip(axes.flat, CATEGORICAL_FEATURES):
        sns.countplot(data=data, x=column, hue=TARGET_COLUMN, ax=ax)
        ax.set_title(column)
    for ax in list(axes.flat)[len(CATEGORICAL_FEATURES):]:
        ax.axis('off')
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(data):
    """Explore raw numeric-coded correlations (not causal relationships)."""
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(data.corr(numeric_only=True), cmap='Spectral', center=0, annot=True, fmt='.2f', linewidths=.5, ax=ax, vmin=-1, vmax=1)
    ax.set_title('Raw feature correlations (coded categories included)')
    fig.tight_layout()
    return fig


def plot_correlation_strength_labels(data, no_correlation=0.05, low_correlation=0.20):
    """Original notebook's NC/LC annotation heatmap, with numeric values otherwise."""
    correlation = data.corr(numeric_only=True)
    labels = correlation.map(lambda value: 'NC' if abs(value) <= no_correlation
                             else 'LC' if abs(value) < low_correlation
                             else f'{value:.2f}') if hasattr(correlation, 'map') else correlation.applymap(
                                 lambda value: 'NC' if abs(value) <= no_correlation
                                 else 'LC' if abs(value) < low_correlation
                                 else f'{value:.2f}')
    np.fill_diagonal(labels.values, '')
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(correlation, cmap='Spectral', annot=labels, fmt='',
                linewidths=.5, vmin=-1, vmax=1, ax=ax)
    ax.set_title('Correlation strength: NC ≤ 0.05, LC < 0.20')
    fig.tight_layout()
    return fig


def plot_variable_type_counts(data):
    """Count numeric and categorical variables (original variable-category chart)."""
    counts = pd.Series({'Numerical': len(NUMERIC_FEATURES),
                        'Categorical': len(CATEGORICAL_FEATURES)})
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=counts.index, y=counts.values, ax=ax, color='steelblue')
    for index, count in enumerate(counts):
        ax.text(index, count + .1, str(count), ha='center')
    ax.set(title='Variable categories', ylabel='Feature count', xlabel='Type')
    fig.tight_layout()
    return fig


def plot_category_pies(data):
    """Restore original categorical feature percentage pie charts."""
    fig, axes = plt.subplots(3, 3, figsize=(14, 13))
    for ax, feature in zip(axes.flat, CATEGORICAL_FEATURES):
        counts = data[feature].value_counts(dropna=False).sort_index()
        ax.pie(counts.values, labels=counts.index.astype(str), autopct='%1.1f%%', startangle=110)
        ax.set_title(feature)
    for ax in list(axes.flat)[len(CATEGORICAL_FEATURES):]: ax.axis('off')
    fig.tight_layout()
    return fig


def plot_numeric_density_by_target(data):
    """Restore target-separated KDE density curves from the original notebook."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, feature in zip(axes.flat, NUMERIC_FEATURES):
        sns.kdeplot(data=data, x=feature, hue=TARGET_COLUMN, common_norm=False, ax=ax)
        ax.set_title(f'{feature} by target')
    for ax in list(axes.flat)[len(NUMERIC_FEATURES):]: ax.axis('off')
    fig.tight_layout()
    return fig


def plot_outlier_boxplots(data, columns=('trtbps', 'thalachh', 'oldpeak')):
    """Show original and winsorized values side by side without altering data."""
    fig, axes = plt.subplots(len(columns), 2, figsize=(12, 3.2 * len(columns)))
    for index, feature in enumerate(columns):
        sns.boxplot(x=data[feature], ax=axes[index, 0], color='skyblue')
        axes[index, 0].set_title(f'{feature}: before')
        after_column = f'{feature}_winsorized'
        if after_column in data:
            sns.boxplot(x=data[after_column], ax=axes[index, 1], color='tan')
            axes[index, 1].set_title(f'{feature}: after winsorization')
        else:
            axes[index, 1].axis('off')
    fig.tight_layout()
    return fig


def plot_winsorization_histograms(data, features=('trtbps', 'oldpeak')):
    """Original versus winsorized distribution and optional sqrt transformation."""
    fig, axes = plt.subplots(len(features), 2, figsize=(12, 4 * len(features)))
    for row, feature in enumerate(features):
        sns.histplot(data[feature], kde=True, ax=axes[row, 0], color='steelblue')
        axes[row, 0].set_title(f'{feature}: original')
        sns.histplot(data[f'{feature}_winsorized'], kde=True, ax=axes[row, 1], color='peru')
        axes[row, 1].set_title(f'{feature}: winsorized')
    fig.tight_layout()
    return fig
