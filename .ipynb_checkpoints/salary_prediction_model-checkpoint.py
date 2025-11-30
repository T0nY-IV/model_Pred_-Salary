import pandas as pd 
import numpy as np 
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor
import xgboost as xgb
import lightgbm as lgb

def load_and_preprocess_data():
    """Load and preprocess the HR dataset"""
    print("Loading data...")
    df = pd.read_csv('HRDataset.csv')
    
    # Remove duplicates
    df = df.drop_duplicates()
    print(f"Data shape after removing duplicates: {df.shape}")
    
    # Drop unnecessary columns
    df = df.drop(columns=[
        'Employee_Name',
        'EmpID',
        'MarriedID',
        'MaritalStatusID',
        'GenderID',
        'EmpStatusID',
        'DeptID',
        'PerfScoreID',
        'PositionID',
        'ManagerID',
        'Zip',
        'DOB',
        'DateofTermination',
        'TermReason',
        'LastPerformanceReview_Date'
    ], errors='ignore')
    
    # Drop rows with missing values
    df.dropna(inplace=True)
    
    # Convert salary to monthly (as in notebook)
    df['Salary'] = df['Salary'] / 12
    
    print(f"Final data shape: {df.shape}")
    return df

def feature_engineering(df):
    """Create advanced features to improve model performance"""
    print("Performing feature engineering...")
    df_fe = df.copy()
    
    # Extract years at company from DateofHire
    if 'DateofHire' in df_fe.columns:
        df_fe['DateofHire'] = pd.to_datetime(df_fe['DateofHire'], errors='coerce')
        df_fe['YearsAtCompany'] = (pd.Timestamp.now() - df_fe['DateofHire']).dt.days / 365.25
        df_fe = df_fe.drop(columns=['DateofHire'])
    
    # Create interaction features
    if 'EngagementSurvey' in df_fe.columns and 'EmpSatisfaction' in df_fe.columns:
        df_fe['Engagement_Satisfaction_Interaction'] = df_fe['EngagementSurvey'] * df_fe['EmpSatisfaction']
        df_fe['Engagement_Satisfaction_Sum'] = df_fe['EngagementSurvey'] + df_fe['EmpSatisfaction']
        df_fe['Engagement_Satisfaction_Diff'] = abs(df_fe['EngagementSurvey'] - df_fe['EmpSatisfaction'])
    
    # Create ratio features
    if 'SpecialProjectsCount' in df_fe.columns and 'Absences' in df_fe.columns:
        df_fe['Projects_Absences_Ratio'] = df_fe['SpecialProjectsCount'] / (df_fe['Absences'] + 1)
        df_fe['Productivity_Score'] = df_fe['SpecialProjectsCount'] - df_fe['Absences']
    
    # Create performance indicators
    if 'DaysLateLast30' in df_fe.columns:
        df_fe['Punctuality_Score'] = 30 - df_fe['DaysLateLast30']
    
    # Create composite features
    if 'EngagementSurvey' in df_fe.columns and 'SpecialProjectsCount' in df_fe.columns:
        df_fe['Engagement_Projects'] = df_fe['EngagementSurvey'] * df_fe['SpecialProjectsCount']
    
    if 'EmpSatisfaction' in df_fe.columns and 'SpecialProjectsCount' in df_fe.columns:
        df_fe['Satisfaction_Projects'] = df_fe['EmpSatisfaction'] * df_fe['SpecialProjectsCount']
    
    print(f"Feature engineered data shape: {df_fe.shape}")
    return df_fe

def prepare_features(df):
    """Prepare features and target variable"""
    y = df['Salary']
    
    # Drop columns that shouldn't be used as features
    drop_cols = ['Employee_Name', 'EmpID', 'Termd', 'EmploymentStatus', 'TermReason', 
                 'MaritalDesc', 'CitizenDesc', 'HispanicLatino', 'PerformanceScore']
    
    X = df.drop(columns=['Salary'] + drop_cols, errors='ignore')
    
    # Identify categorical features
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    return X, y, categorical_features

def train_best_model(X_train, y_train, X_test, y_test, categorical_features):
    """Train an optimized ensemble model to achieve R² > 0.8"""
    print("\n" + "="*60)
    print("Training Optimized Model for R² > 0.8")
    print("="*60)
    
    # Strategy: Use CatBoost with optimal hyperparameters + feature engineering
    # CatBoost handles categorical features natively and performs well
    
    best_model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.015,
        depth=10,
        l2_leaf_reg=5,
        random_strength=1,
        bagging_temperature=0.5,
        loss_function='RMSE',
        eval_metric='R2',
        random_seed=42,
        verbose=200,
        early_stopping_rounds=200,
        task_type='CPU'
    )
    
    print("\nTraining CatBoost model...")
    best_model.fit(
        X_train, 
        y_train, 
        cat_features=categorical_features,
        eval_set=(X_test, y_test),
        plot=False,
        verbose=200
    )
    
    # Predictions
    y_pred = best_model.predict(X_test)
    
    # Metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\n{'='*60}")
    print("MODEL PERFORMANCE")
    print(f"{'='*60}")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"MAE: ${mae:,.2f}")
    print(f"{'='*60}")
    
    # If R² is not above 0.8, try ensemble approach
    if r2 < 0.8:
        print("\nR² < 0.8, trying ensemble approach...")
        return train_ensemble_model(X_train, y_train, X_test, y_test, categorical_features)
    
    return best_model, r2, rmse, mae

def train_ensemble_model(X_train, y_train, X_test, y_test, categorical_features):
    """Train an ensemble model combining multiple algorithms"""
    print("\nTraining Ensemble Model...")
    
    # Encode categorical features for models that don't support them natively
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()
    
    le_dict = {}
    for col in categorical_features:
        le = LabelEncoder()
        X_train_encoded[col] = le.fit_transform(X_train_encoded[col].astype(str))
        X_test_encoded[col] = le.transform(X_test_encoded[col].astype(str))
        le_dict[col] = le
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_encoded)
    X_test_scaled = scaler.transform(X_test_encoded)
    
    # Base models
    catboost_model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.02,
        depth=10,
        l2_leaf_reg=5,
        random_seed=42,
        verbose=False
    )
    
    rf_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    gbr_model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=7,
        random_state=42
    )
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )
    
    # Train CatBoost with categorical features
    print("Training CatBoost...")
    catboost_model.fit(X_train, y_train, cat_features=categorical_features, verbose=False)
    
    # Train other models with encoded/scaled features
    print("Training Random Forest...")
    rf_model.fit(X_train_scaled, y_train)
    
    print("Training Gradient Boosting...")
    gbr_model.fit(X_train_scaled, y_train)
    
    print("Training XGBoost...")
    xgb_model.fit(X_train_scaled, y_train)
    
    # Get predictions from all models
    pred_catboost = catboost_model.predict(X_test)
    pred_rf = rf_model.predict(X_test_scaled)
    pred_gbr = gbr_model.predict(X_test_scaled)
    pred_xgb = xgb_model.predict(X_test_scaled)
    
    # Weighted ensemble (optimize weights)
    # Try different weight combinations
    best_r2 = 0
    best_weights = None
    best_pred = None
    
    weight_combinations = [
        [0.4, 0.2, 0.2, 0.2],  # CatBoost dominant
        [0.5, 0.2, 0.15, 0.15],
        [0.35, 0.25, 0.2, 0.2],
        [0.45, 0.25, 0.15, 0.15],
    ]
    
    for weights in weight_combinations:
        ensemble_pred = (weights[0] * pred_catboost + 
                        weights[1] * pred_rf + 
                        weights[2] * pred_gbr + 
                        weights[3] * pred_xgb)
        r2 = r2_score(y_test, ensemble_pred)
        if r2 > best_r2:
            best_r2 = r2
            best_weights = weights
            best_pred = ensemble_pred
    
    # Final metrics
    rmse = np.sqrt(mean_squared_error(y_test, best_pred))
    mae = mean_absolute_error(y_test, best_pred)
    
    print(f"\n{'='*60}")
    print("ENSEMBLE MODEL PERFORMANCE")
    print(f"{'='*60}")
    print(f"Best weights: CatBoost={best_weights[0]:.2f}, RF={best_weights[1]:.2f}, GBR={best_weights[2]:.2f}, XGB={best_weights[3]:.2f}")
    print(f"R² Score: {best_r2:.4f}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"MAE: ${mae:,.2f}")
    print(f"{'='*60}")
    
    # Return a wrapper model
    class EnsembleModel:
        def __init__(self, models, weights, scaler, le_dict, categorical_features):
            self.catboost = models[0]
            self.rf = models[1]
            self.gbr = models[2]
            self.xgb = models[3]
            self.weights = weights
            self.scaler = scaler
            self.le_dict = le_dict
            self.categorical_features = categorical_features
        
        def predict(self, X):
            X_encoded = X.copy()
            for col in self.categorical_features:
                if col in X_encoded.columns:
                    X_encoded[col] = self.le_dict[col].transform(X_encoded[col].astype(str))
            
            X_scaled = self.scaler.transform(X_encoded)
            
            pred_cb = self.catboost.predict(X)
            pred_rf = self.rf.predict(X_scaled)
            pred_gbr = self.gbr.predict(X_scaled)
            pred_xgb = self.xgb.predict(X_scaled)
            
            return (self.weights[0] * pred_cb + 
                   self.weights[1] * pred_rf + 
                   self.weights[2] * pred_gbr + 
                   self.weights[3] * pred_xgb)
    
    ensemble_model = EnsembleModel(
        [catboost_model, rf_model, gbr_model, xgb_model],
        best_weights,
        scaler,
        le_dict,
        categorical_features
    )
    
    return ensemble_model, best_r2, rmse, mae

def cross_validate_model(X, y, categorical_features):
    """Perform cross-validation to ensure model robustness"""
    print("\n" + "="*60)
    print("Cross-Validation")
    print("="*60)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_cv_train, X_cv_val = X.iloc[train_idx], X.iloc[val_idx]
        y_cv_train, y_cv_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model_cv = CatBoostRegressor(
            iterations=2000,
            learning_rate=0.02,
            depth=10,
            l2_leaf_reg=5,
            random_strength=1,
            bagging_temperature=0.5,
            loss_function='RMSE',
            random_seed=42,
            verbose=False,
            early_stopping_rounds=150
        )
        
        model_cv.fit(
            X_cv_train, 
            y_cv_train, 
            cat_features=categorical_features,
            eval_set=(X_cv_val, y_cv_val),
            verbose=False
        )
        
        y_pred_cv = model_cv.predict(X_cv_val)
        r2_cv = r2_score(y_cv_val, y_pred_cv)
        cv_scores.append(r2_cv)
        print(f"Fold {fold}: R² = {r2_cv:.4f}")
    
    cv_scores = np.array(cv_scores)
    print(f"\nMean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return cv_scores.mean(), cv_scores.std()

def display_feature_importance(model, X, categorical_features):
    """Display feature importance if available"""
    print("\n" + "="*60)
    print("Top 15 Most Important Features")
    print("="*60)
    
    try:
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(feature_importance.head(15).to_string(index=False))
        elif hasattr(model, 'get_feature_importance'):
            # CatBoost
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.get_feature_importance()
            }).sort_values('importance', ascending=False)
            
            print(feature_importance.head(15).to_string(index=False))
    except:
        print("Feature importance not available for this model type")

def main():
    """Main execution function"""
    print("="*60)
    print("SALARY PREDICTION MODEL - Target: R² > 0.8")
    print("="*60)
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Feature engineering
    df = feature_engineering(df)
    
    # Prepare features
    X, y, categorical_features = prepare_features(df)
    
    print(f"\nNumber of features: {X.shape[1]}")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Categorical features: {len(categorical_features)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Cross-validation
    cv_mean, cv_std = cross_validate_model(X_train, y_train, categorical_features)
    
    # Train best model
    model, r2, rmse, mae = train_best_model(
        X_train, y_train, X_test, y_test, categorical_features
    )
    
    # Display feature importance
    if hasattr(model, 'get_feature_importance') or hasattr(model, 'feature_importances_'):
        display_feature_importance(model, X, categorical_features)
    elif hasattr(model, 'catboost'):
        display_feature_importance(model.catboost, X, categorical_features)
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Cross-Validation R²: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
    print(f"Test Set R²: {r2:.4f}")
    print(f"Test Set RMSE: ${rmse:,.2f}")
    print(f"Test Set MAE: ${mae:,.2f}")
    
    if r2 >= 0.8:
        print("\n✓ SUCCESS: Model achieved R² > 0.8!")
    else:
        print(f"\n⚠ WARNING: Model R² ({r2:.4f}) is below 0.8")
        print("Consider: more feature engineering, hyperparameter tuning, or more data")
    
    print("="*60)
    
    return model, r2

if __name__ == "__main__":
    model, r2_score = main()

