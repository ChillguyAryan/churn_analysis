# helper.py
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
from preprocessor import preprocess

# Load encoders if exists
try:
    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
except FileNotFoundError:
    encoders = None


def train_models(df):
    """
    Train multiple classification models and evaluate their performance.
    
    Parameters:
    df (DataFrame): Customer churn dataset
    
    Returns:
    tuple: (metrics_dict, best_model_name)
    """
    # Accept DataFrame directly
    if isinstance(df, str):
        df = pd.read_csv(df)

    # Preprocess data
    encoders, X_train_smote, y_train_smote, X_test, y_test = preprocess(df)

    # Define models
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "DecisionTree": DecisionTreeClassifier(random_state=42, max_depth=10),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, max_depth=15),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    }

    metrics_dict = {}

    # Train and evaluate each model
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_smote, y_train_smote)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics_dict[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "report": classification_report(y_test, y_pred, zero_division=0),
            "conf_matrix": metrics.confusion_matrix(y_test, y_pred)
        }

        # Save each model individually
        model_file = f"{name.replace(' ', '_').lower()}_model.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(model, f)
        print(f"✓ {name} saved as {model_file}")

    # Find the best model based on ROC-AUC
    best_model_name = max(metrics_dict, key=lambda x: metrics_dict[x]["roc_auc"])
    print(f"\n✅ Best Model: {best_model_name}")
    print(f"ROC-AUC Score: {metrics_dict[best_model_name]['roc_auc']:.4f}")
    print("\nClassification Report:")
    print(metrics_dict[best_model_name]["report"])

    return metrics_dict, best_model_name


def countplot(df):
    """
    Generate count plots for categorical features.
    
    Parameters:
    df (DataFrame): Dataset
    
    Returns:
    list: List of matplotlib figures
    """
    object_cols = df.select_dtypes(include="object").columns.to_list()
    if "SeniorCitizen" not in object_cols and "SeniorCitizen" in df.columns:
        object_cols = ['SeniorCitizen'] + object_cols

    figures = []
    for col in object_cols:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x=df[col], palette="coolwarm", ax=ax)
        ax.set_title(f"Count Plot of {col}", fontsize=14)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        figures.append(fig)
    
    return figures


def comparison(metrics_dict):
    """
    Create visual comparison of model performance.
    
    Parameters:
    metrics_dict (dict): Dictionary containing model metrics
    
    Returns:
    matplotlib.figure: Comparison plot
    """
    df_metrics = pd.DataFrame(metrics_dict).T
    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    fig, ax = plt.subplots(figsize=(12, 6))
    df_metrics[metric_cols].plot(kind="bar", ax=ax, colormap="viridis")
    ax.set_title("Model Performance Comparison", fontsize=16, fontweight='bold')
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xlabel("Models", fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
    ax.legend(loc="lower right", fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    return fig


def weightage(model, X_train):
    """
    Display feature importance for the given model.
    
    Parameters:
    model: Trained model
    X_train (DataFrame): Training features
    
    Returns:
    tuple: (feature_importance_df, figure) or None
    """
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0])
    else:
        print("⚠️ This model does not support feature importance.")
        return None

    feature_importance = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=feature_importance.head(15), 
                palette="mako", ax=ax)
    ax.set_title("Top 15 Important Features", fontsize=14, fontweight='bold')
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    plt.tight_layout()

    return feature_importance, fig


def churn(df):
    """
    Generate churn distribution and correlation heatmap.
    
    Parameters:
    df (DataFrame): Dataset
    
    Returns:
    tuple: (churn_fig, correlation_fig) or (None, None)
    """
    if "Churn" not in df.columns:
        print("⚠️ 'Churn' column not found in dataset.")
        return None, None

    # Churn distribution plot
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    churn_col = df["Churn"].copy()
    
    if churn_col.dtype == "object":
        sns.countplot(x=churn_col, palette="Set2", ax=ax1)
    else:
        sns.countplot(x=churn_col, palette="Set2", ax=ax1)
        ax1.set_xticklabels(["No", "Yes"])
    
    ax1.set_title("Churn Distribution", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Churn Status", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    
    # Add count labels on bars
    for container in ax1.containers:
        ax1.bar_label(container)
    
    plt.tight_layout()

    # Correlation heatmap
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) > 0:
        correlation_matrix = numeric_df.corr()
        sns.heatmap(correlation_matrix, cmap="coolwarm", annot=False, 
                    fmt=".2f", ax=ax2, center=0)
        ax2.set_title("Correlation Heatmap", fontsize=14, fontweight='bold')
        plt.tight_layout()
    else:
        print("⚠️ No numeric columns found for correlation matrix.")
        plt.close(fig2)
        fig2 = None

    return fig1, fig2