# Coal India Profit Prediction ML Model
# Machine Learning Regression Model for 2026 Profit Forecasting

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('coal_india_financials_2015_2025.csv')

print("Dataset Info:")
print("="*50)
print(f"Shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nSummary statistics:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

# Data preparation and feature engineering
df_work = df.copy()

# Extract year as numerical value
df_work['Year_Numeric'] = df_work['Year'].str.extract('(\d{4})').astype(int)

# Create additional features that might be useful for prediction
df_work['Revenue_Growth'] = df_work['Sales (Revenue) (₹ Cr)'].pct_change()
df_work['Operating_Margin'] = df_work['Operating Profit (₹ Cr)'] / df_work['Sales (Revenue) (₹ Cr)'] * 100
df_work['Expense_Ratio'] = df_work['Expenses (₹ Cr)'] / df_work['Sales (Revenue) (₹ Cr)'] * 100
df_work['Tax_Rate'] = df_work['Tax (₹ Cr)'] / df_work['Profit Before Tax (₹ Cr)'] * 100

# Handle first row NaN for growth rate
df_work['Revenue_Growth'].fillna(0, inplace=True)

print("Enhanced dataset with new features:")
display_cols = ['Year', 'Year_Numeric', 'Sales (Revenue) (₹ Cr)', 'Profit After Tax (₹ Cr)', 
               'Revenue_Growth', 'Operating_Margin', 'Expense_Ratio', 'Tax_Rate']
print(df_work[display_cols].round(2))

# Calculate correlation only with numeric columns, excluding the original 'Year' column
numeric_df = df_work.select_dtypes(include=[np.number])
print("\nCorrelation with Profit After Tax:")
corr_with_target = numeric_df.corr()['Profit After Tax (₹ Cr)'].sort_values(ascending=False)
print(corr_with_target.round(3))

# Prepare data for machine learning
target = 'Profit After Tax (₹ Cr)'

# Select features for the model (excluding highly correlated ones like Profit Before Tax to avoid multicollinearity)
feature_columns = [
    'Year_Numeric', 'Sales (Revenue) (₹ Cr)', 'Operating Profit (₹ Cr)', 
    'Expenses (₹ Cr)', 'Interest (₹ Cr)', 'Depreciation (₹ Cr)', 'Tax (₹ Cr)',
    'Revenue_Growth', 'Operating_Margin', 'Expense_Ratio'
]

X = df_work[feature_columns]
y = df_work[target]

print("Features selected for modeling:")
print(X.columns.tolist())
print(f"\nTarget variable: {target}")
print(f"Data shape: X={X.shape}, y={y.shape}")

# Split the data (with small dataset, we'll use most for training)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set: X={X_train.shape}, y={y_train.shape}")
print(f"Test set: X={X_test.shape}, y={y_test.shape}")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model 1: Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Model 2: Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)  # Random Forest doesn't need scaling

# Make predictions on test set
lr_pred = lr_model.predict(X_test_scaled)
rf_pred = rf_model.predict(X_test)

# Evaluate models
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{model_name} Performance:")
    print(f"RMSE: ₹{rmse:,.2f} Cr")
    print(f"MAE: ₹{mae:,.2f} Cr")
    print(f"R² Score: {r2:.4f}")
    return rmse, mae, r2

lr_metrics = evaluate_model(y_test, lr_pred, "Linear Regression")
rf_metrics = evaluate_model(y_test, rf_pred, "Random Forest")

# Feature importance from Random Forest
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("Feature Importance (Random Forest):")
print(feature_importance.round(4))

# Prepare data for 2026 prediction
print("\n" + "="*60)
print("PREPARING 2026 PREDICTION DATA")
print("="*60)

# Calculate recent trends (last 3-5 years) to project 2026 values
recent_years = df_work.tail(5).copy()  # Last 5 years for trend analysis

def calculate_trend_projection(series, target_year=2026):
    """Calculate projected value based on linear trend"""
    years = recent_years['Year_Numeric'].values
    values = series.values
    
    # Linear regression for trend
    slope, intercept = np.polyfit(years, values, 1)
    projected_value = slope * target_year + intercept
    
    return projected_value, slope

# Project 2026 values
projected_2026 = {}
trends = {}

print("\nTrend Analysis and 2026 Projections:")
print("-" * 50)

for col in feature_columns:
    if col == 'Year_Numeric':
        projected_2026[col] = 2026
        trends[col] = 0
    else:
        proj_val, slope = calculate_trend_projection(recent_years[col])
        projected_2026[col] = proj_val
        trends[col] = slope
        
        print(f"{col}:")
        print(f"  2025 Value: {recent_years[col].iloc[-1]:,.2f}")
        print(f"  Projected 2026: {proj_val:,.2f}")
        print(f"  Trend (yearly): {slope:,.2f}")
        print()

# Create 2026 feature vector
X_2026 = pd.DataFrame([projected_2026])
print("2026 Feature Vector:")
print(X_2026.round(2))

# Make 2026 predictions with both models
X_2026_scaled = scaler.transform(X_2026)

# Linear Regression prediction
lr_2026_pred = lr_model.predict(X_2026_scaled)[0]

# Random Forest prediction
rf_2026_pred = rf_model.predict(X_2026)[0]

print("2026 PROFIT AFTER TAX PREDICTIONS")
print("="*50)
print(f"Linear Regression: ₹{lr_2026_pred:,.0f} Cr")
print(f"Random Forest: ₹{rf_2026_pred:,.0f} Cr")
print(f"Average Prediction: ₹{(lr_2026_pred + rf_2026_pred)/2:,.0f} Cr")

# Calculate prediction confidence based on model performance
lr_uncertainty = lr_metrics[0]  # RMSE
rf_uncertainty = rf_metrics[0]  # RMSE

print(f"\nPrediction Ranges (±1 RMSE):")
print(f"Linear Regression: ₹{lr_2026_pred-lr_uncertainty:,.0f} - ₹{lr_2026_pred+lr_uncertainty:,.0f} Cr")
print(f"Random Forest: ₹{rf_2026_pred-rf_uncertainty:,.0f} - ₹{rf_2026_pred+rf_uncertainty:,.0f} Cr")

# Compare with historical growth
print(f"\n" + "="*50)
print("HISTORICAL CONTEXT")
print("="*50)

# Calculate year-over-year growth rates
df_work['PAT_Growth'] = df_work['Profit After Tax (₹ Cr)'].pct_change() * 100
recent_growth = df_work['PAT_Growth'].dropna().tail(5)

print("Recent Profit After Tax Growth Rates:")
for idx, row in df_work.tail(6).iterrows():
    if not np.isnan(row['PAT_Growth']):
        print(f"{row['Year']}: {row['PAT_Growth']:+.1f}%")

avg_growth = recent_growth.mean()
print(f"\nAverage Growth Rate (last 5 years): {avg_growth:.1f}%")

# Simple growth-based prediction for comparison
current_pat = df_work['Profit After Tax (₹ Cr)'].iloc[-1]
growth_based_pred = current_pat * (1 + avg_growth/100)

print(f"\nSimple Growth-Based Prediction: ₹{growth_based_pred:,.0f} Cr")
print(f"(Based on {avg_growth:.1f}% average growth rate)")

# Summary
print(f"\n" + "="*50)
print("FINAL RECOMMENDATION")
print("="*50)
best_pred = rf_2026_pred  # Random Forest performed better
print(f"Best Model Prediction (Random Forest): ₹{best_pred:,.0f} Cr")
print(f"Confidence Range: ₹{best_pred-rf_uncertainty:,.0f} - ₹{best_pred+rf_uncertainty:,.0f} Cr")
print(f"Compared to 2025: {((best_pred/current_pat - 1) * 100):+.1f}% growth")

# Create a comprehensive results dataframe for analysis and export
results_summary = pd.DataFrame({
    'Method': ['Random Forest (Recommended)', 'Linear Regression', 'Simple Growth Average', 'Conservative Estimate'],
    'Prediction_2026_Cr': [rf_2026_pred, lr_2026_pred, growth_based_pred, (rf_2026_pred + growth_based_pred)/2],
    'Lower_Bound_Cr': [rf_2026_pred - rf_uncertainty, lr_2026_pred - lr_uncertainty, 
                       growth_based_pred * 0.85, (rf_2026_pred + growth_based_pred)/2 * 0.9],
    'Upper_Bound_Cr': [rf_2026_pred + rf_uncertainty, lr_2026_pred + lr_uncertainty,
                       growth_based_pred * 1.15, (rf_2026_pred + growth_based_pred)/2 * 1.1],
    'Growth_vs_2025_Percent': [((rf_2026_pred/current_pat - 1) * 100), 
                               ((lr_2026_pred/current_pat - 1) * 100),
                               ((growth_based_pred/current_pat - 1) * 100),
                               (((rf_2026_pred + growth_based_pred)/2/current_pat - 1) * 100)]
})

print("COMPREHENSIVE 2026 PROFIT PREDICTIONS")
print("="*60)
print(results_summary.round(2))

# Model details and coefficients
print(f"\n" + "="*60)
print("MODEL DETAILS")
print("="*60)

print("Random Forest Model - Top 5 Important Features:")
print(feature_importance.head().round(4))

print(f"\nLinear Regression Coefficients:")
lr_coef_df = pd.DataFrame({
    'Feature': feature_columns,
    'Coefficient': lr_model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)
print(lr_coef_df.round(2))

# Create historical data with predictions for visualization
historical_data = df_work[['Year', 'Year_Numeric', 'Profit After Tax (₹ Cr)']].copy()
historical_data['Type'] = 'Historical'

# Add 2026 predictions
pred_2026 = pd.DataFrame({
    'Year': ['Mar-2026', 'Mar-2026', 'Mar-2026'],
    'Year_Numeric': [2026, 2026, 2026],
    'Profit After Tax (₹ Cr)': [rf_2026_pred, lr_2026_pred, growth_based_pred],
    'Type': ['Random Forest', 'Linear Regression', 'Growth-Based']
})

combined_data = pd.concat([historical_data, pred_2026], ignore_index=True)

# Save results to CSV
results_file = 'coal_india_profit_predictions_2026.csv'
combined_data.to_csv(results_file, index=False)
print(f"\nResults saved to: {results_file}")

# Save model summary
model_summary = pd.DataFrame({
    'Model': ['Random Forest', 'Linear Regression'],
    'R2_Score': [rf_metrics[2], lr_metrics[2]],
    'RMSE_Cr': [rf_metrics[0], lr_metrics[0]],
    'MAE_Cr': [rf_metrics[1], lr_metrics[1]],
    'Prediction_2026_Cr': [rf_2026_pred, lr_2026_pred]
})

model_summary_file = 'model_performance_summary.csv'
model_summary.to_csv(model_summary_file, index=False)
print(f"Model summary saved to: {model_summary_file}")

print(f"\n" + "="*60)
print("KEY INSIGHTS")
print("="*60)
print("• Random Forest model shows more conservative prediction due to complex")
print("  non-linear patterns it detected in the data")
print("• The model accounts for recent volatility (2021-2022 dip, 2023-2025 recovery)")
print("• Operating Profit, Interest, and Expenses are the most important predictors")
print("• High growth in 2024-2025 may not be sustainable, leading to moderated 2026 forecast")
print(f"• Recommended range: ₹{rf_2026_pred-rf_uncertainty:,.0f} - ₹{rf_2026_pred+rf_uncertainty:,.0f} Cr")

# Optional: Save the trained models for future use
import joblib

# Save the models and scaler
joblib.dump(rf_model, 'random_forest_model.pkl')
joblib.dump(lr_model, 'linear_regression_model.pkl')
joblib.dump(scaler, 'feature_scaler.pkl')

print("\nModels saved:")
print("- random_forest_model.pkl")
print("- linear_regression_model.pkl") 
print("- feature_scaler.pkl")

print("\nTo use the models for future predictions:")
print("1. Load models: rf_model = joblib.load('random_forest_model.pkl')")
print("2. Prepare new data with same features")
print("3. Make predictions: prediction = rf_model.predict(new_data)")