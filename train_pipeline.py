"""
Customer Churn Prediction Pipeline
- Data loading and preprocessing
- EDA with visualizations
- Model training (Logistic Regression, Random Forest, XGBoost)
- Model comparison and selection
- SHAP explainability
- Model persistence
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import xgboost as xgb
import shap
import joblib
import warnings
import os

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'data_path': 'data/churn_data.csv',
    'model_output_dir': 'models',
    'plots_output_dir': 'plots',
    'test_size': 0.2,
    'random_state': 42,
    'random_forest_params': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 20,
        'random_state': 42,
        'n_jobs': -1
    },
    'xgboost_params': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_directories() -> None:
    """Ensure required directories exist."""
    for directory in [CONFIG['model_output_dir'], CONFIG['plots_output_dir'], 'data']:
        os.makedirs(directory, exist_ok=True)


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Load and clean the churn dataset.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        Cleaned DataFrame
    """
    print("\n📂 Loading data...")
    df = pd.read_csv(filepath)
    print(f"   ✓ Loaded {len(df)} records with {len(df.columns)} features")
    
    # Handle missing values
    print("🔧 Handling missing values...")
    df['total_charges'].fillna(df['total_charges'].median(), inplace=True)
    print(f"   ✓ Missing values handled")
    
    # Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    print(f"   ✓ Removed {initial_rows - len(df)} duplicates")
    
    return df


def preprocess_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Encode categorical features and scale numerical features.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Processed DataFrame and encoder dictionary
    """
    print("\n🔤 Preprocessing features...")
    df_processed = df.copy()
    encoders = {}
    
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    categorical_cols.remove('customer_id')  # Keep ID for reference
    
    # Label encode categorical features
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        encoders[col] = le
        print(f"   ✓ Encoded {col}: {len(le.classes_)} classes")
    
    return df_processed, encoders


def perform_eda(df: pd.DataFrame, plots_dir: str = 'plots') -> None:
    """
    Perform Exploratory Data Analysis with 4+ visualizations.
    
    Args:
        df: Input DataFrame
        plots_dir: Directory to save plots
    """
    print("\n📊 Performing EDA...")
    
    # Set style
    sns.set_style("darkgrid")
    plt.rcParams['figure.figsize'] = (15, 12)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Churn Distribution
    ax1 = axes[0, 0]
    churn_counts = df['churn'].value_counts()
    colors = ['#2ecc71', '#e74c3c']
    ax1.bar(churn_counts.index, churn_counts.values, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Churn', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax1.set_title('1. Churn Distribution', fontsize=12, fontweight='bold')
    ax1.set_xticklabels(['No Churn', 'Churn'])
    for i, v in enumerate(churn_counts.values):
        ax1.text(i, v + 50, str(v), ha='center', fontweight='bold')
    
    # 2. Correlation Heatmap
    ax2 = axes[0, 1]
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numerical_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                ax=ax2, cbar_kws={'label': 'Correlation'}, linewidths=0.5)
    ax2.set_title('2. Correlation Heatmap (Numerical Features)', fontsize=12, fontweight='bold')
    
    # 3. Tenure vs Churn
    ax3 = axes[1, 0]
    tenure_churn = df.groupby('tenure')['churn'].agg(['sum', 'count'])
    tenure_churn['churn_rate'] = (tenure_churn['sum'] / tenure_churn['count'] * 100).round(2)
    ax3.plot(tenure_churn.index, tenure_churn['churn_rate'], marker='o', linestyle='-', 
             linewidth=2, markersize=4, color='#3498db')
    ax3.fill_between(tenure_churn.index, tenure_churn['churn_rate'], alpha=0.3, color='#3498db')
    ax3.set_xlabel('Tenure (months)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Churn Rate (%)', fontsize=11, fontweight='bold')
    ax3.set_title('3. Churn Rate by Tenure', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Monthly Charges vs Churn
    ax4 = axes[1, 1]
    churn_0 = df[df['churn'] == 0]['monthly_charges']
    churn_1 = df[df['churn'] == 1]['monthly_charges']
    ax4.hist([churn_0, churn_1], bins=30, label=['No Churn', 'Churn'], 
             color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Monthly Charges ($)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax4.set_title('4. Monthly Charges Distribution by Churn', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/eda_visualizations.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ EDA visualizations saved to {plots_dir}/eda_visualizations.png")
    plt.close()


def train_models(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                 y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """
    Train three classification models.
    
    Args:
        X_train, X_test: Feature sets
        y_train, y_test: Target sets
        
    Returns:
        Dictionary containing trained models and metrics
    """
    print("\n🤖 Training models...")
    
    models = {}
    metrics = {}
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ========== MODEL 1: Logistic Regression ==========
    print("\n  1️⃣  Logistic Regression")
    lr = LogisticRegression(max_iter=1000, random_state=CONFIG['random_state'], n_jobs=-1)
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    y_pred_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]
    
    metrics['Logistic Regression'] = calculate_metrics(y_test, y_pred_lr, y_pred_proba_lr)
    models['Logistic Regression'] = lr
    print(f"     ✓ Accuracy: {metrics['Logistic Regression']['Accuracy']:.4f}")
    print(f"     ✓ ROC-AUC: {metrics['Logistic Regression']['ROC-AUC']:.4f}")
    
    # ========== MODEL 2: Random Forest ==========
    print("\n  2️⃣  Random Forest")
    rf = RandomForestClassifier(**CONFIG['random_forest_params'])
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_pred_proba_rf = rf.predict_proba(X_test)[:, 1]
    
    metrics['Random Forest'] = calculate_metrics(y_test, y_pred_rf, y_pred_proba_rf)
    models['Random Forest'] = rf
    print(f"     ✓ Accuracy: {metrics['Random Forest']['Accuracy']:.4f}")
    print(f"     ✓ ROC-AUC: {metrics['Random Forest']['ROC-AUC']:.4f}")
    
    # ========== MODEL 3: XGBoost ==========
    print("\n  3️⃣  XGBoost")
    xgb_model = xgb.XGBClassifier(**CONFIG['xgboost_params'], eval_metric='logloss')
    xgb_model.fit(X_train, y_train, verbose=False)
    y_pred_xgb = xgb_model.predict(X_test)
    y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
    
    metrics['XGBoost'] = calculate_metrics(y_test, y_pred_xgb, y_pred_proba_xgb)
    models['XGBoost'] = xgb_model
    print(f"     ✓ Accuracy: {metrics['XGBoost']['Accuracy']:.4f}")
    print(f"     ✓ ROC-AUC: {metrics['XGBoost']['ROC-AUC']:.4f}")
    
    return {
        'models': models,
        'metrics': metrics,
        'scaler': scaler,
        'X_test': X_test,
        'X_test_scaled': X_test_scaled,
        'y_test': y_test
    }


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     y_pred_proba: np.ndarray) -> Dict[str, float]:
    """Calculate classification metrics."""
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred),
        'ROC-AUC': roc_auc_score(y_true, y_pred_proba)
    }


def compare_models(metrics: Dict[str, Dict[str, float]], 
                  plots_dir: str = 'plots') -> None:
    """
    Compare model performance and visualize results.
    
    Args:
        metrics: Dictionary of model metrics
        plots_dir: Directory to save plots
    """
    print("\n📈 Comparing models...")
    
    metrics_df = pd.DataFrame(metrics).T
    print("\n" + "="*70)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*70)
    print(metrics_df.to_string())
    print("="*70)
    
    # Determine best model
    best_model = metrics_df['ROC-AUC'].idxmax()
    print(f"\n🏆 Best Model: {best_model} (ROC-AUC: {metrics_df.loc[best_model, 'ROC-AUC']:.4f})")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics_df.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='lower right', fontsize=10)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Comparison plot saved")
    plt.close()
    
    return best_model


def explain_with_shap(model: Any, X_test: pd.DataFrame, 
                     model_name: str, plots_dir: str = 'plots') -> None:
    """
    Generate SHAP explainability plots.
    
    Args:
        model: Trained model
        X_test: Test features
        model_name: Name of the model
        plots_dir: Directory to save plots
    """
    print(f"\n🔍 Generating SHAP explanations for {model_name}...")
    
    # Create explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Handle binary classification
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Feature importance plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, plot_type='bar', show=False)
    plt.title(f'SHAP Feature Importance - {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/shap_feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ SHAP feature importance plot saved")
    plt.close()
    
    # Summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title(f'SHAP Summary Plot - {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/shap_summary.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ SHAP summary plot saved")
    plt.close()
    
    print(f"   ✓ SHAP explanations completed")


def save_model(model: Any, scaler: StandardScaler, 
              encoders: Dict[str, LabelEncoder], 
              model_name: str = 'best_model', 
              output_dir: str = 'models') -> str:
    """
    Save trained model, scaler, and encoders.
    
    Args:
        model: Trained model
        scaler: StandardScaler instance
        encoders: Dictionary of LabelEncoders
        model_name: Name for the model
        output_dir: Output directory
        
    Returns:
        Path to saved model
    """
    print(f"\n💾 Saving {model_name}...")
    
    model_path = f'{output_dir}/{model_name}.pkl'
    scaler_path = f'{output_dir}/scaler.pkl'
    encoders_path = f'{output_dir}/encoders.pkl'
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(encoders, encoders_path)
    
    print(f"   ✓ Model saved to {model_path}")
    print(f"   ✓ Scaler saved to {scaler_path}")
    print(f"   ✓ Encoders saved to {encoders_path}")
    
    return model_path


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Execute the complete ML pipeline."""
    print("\n" + "="*70)
    print("🎯 CUSTOMER CHURN PREDICTION - ML PIPELINE")
    print("="*70)
    
    ensure_directories()
    
    # Import and generate data if not exists
    from generate_data import generate_churn_dataset
    
    if not os.path.exists(CONFIG['data_path']):
        print("\n📊 Generating synthetic dataset...")
        df = generate_churn_dataset(n_samples=5000)
        df.to_csv(CONFIG['data_path'], index=False)
        print(f"✓ Dataset generated and saved to {CONFIG['data_path']}")
    
    # 1. Load and clean data
    df = load_and_clean_data(CONFIG['data_path'])
    print(f"\nDataset shape: {df.shape}")
    print(f"Churn rate: {df['churn'].mean():.2%}")
    
    # 2. Preprocess features
    df_processed, encoders = preprocess_features(df)
    
    # 3. Perform EDA
    perform_eda(df)
    
    # 4. Prepare data for modeling
    X = df_processed.drop(['customer_id', 'churn'], axis=1)
    y = df_processed['churn'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG['test_size'], random_state=CONFIG['random_state'],
        stratify=y
    )
    print(f"\n📊 Train set: {X_train.shape[0]} | Test set: {X_test.shape[0]}")
    
    # 5. Train models
    results = train_models(X_train, X_test, y_train, y_test)
    
    # 6. Compare models
    best_model_name = compare_models(results['metrics'])
    best_model = results['models'][best_model_name]
    
    # 7. Generate SHAP explanations
    explain_with_shap(best_model, X_test, best_model_name)
    
    # 8. Save best model
    save_model(best_model, results['scaler'], encoders, model_name='best_model')
    
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()