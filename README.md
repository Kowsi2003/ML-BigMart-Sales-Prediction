# Big Mart Sales Prediction

This project uses the [Big Mart Sales dataset from Kaggle](https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data) to predict retail product sales across different Big Mart outlets. Using Python, pandas, and XGBoost, a complete pipeline was built covering each essential ML workflow stage:

## Dataset:

* 8523 rows and 12 columns, including product details, outlet details, and the target variable `Item_Outlet_Sales`.

## Detailed Workflow:

* **Data Collection:** Loaded CSV data into pandas DataFrame, verified structure using `.head()`, `.shape`, and `.info()` for initial inspection.
* **Data Cleaning:**

  * Checked for missing values using `.isnull().sum()`.
  * Filled missing `Item_Weight` with the column mean.
  * Filled missing `Outlet_Size` using mode based on `Outlet_Type` via pivot tables for consistent domain-aware imputation.
* **EDA:**

  * Visualized distributions of numerical features (`Item_Weight`, `Item_MRP`, `Item_Visibility`, `Item_Outlet_Sales`) using seaborn distplots.
  * Analyzed categorical features (`Item_Fat_Content`, `Item_Type`, `Outlet_Size`) using countplots.
  * Identified correlations and distribution patterns for better feature understanding.
* **Preprocessing:**

  * Standardized category labels in `Item_Fat_Content` (e.g., 'LF', 'low fat' → 'Low Fat').
  * Label encoded categorical columns (`Item_Identifier`, `Outlet_Identifier`, `Outlet_Type`, etc.) for ML model compatibility.
* **Model Training:**

  * Split data into training and testing sets (80/20 split) using `train_test_split`.
  * Trained an XGBoost Regressor model on the training data to learn patterns in sales based on features.
* **Evaluation:**

  * Predicted on training and testing data.
  * Calculated R² scores for training (0.636) and testing (0.587), demonstrating the model’s capability to generalize while capturing key trends in sales data.

## Summary Results:

✅ **Training R² Score:** 0.636 (model explains \~63.6% of variance in training data).

✅ **Testing R² Score:** 0.587 (model explains \~58.7% of variance on unseen data).

## Inference:

* `Item_MRP` and `Outlet_Type` strongly influence sales predictions.
* The model provides a reliable baseline for sales forecasting to aid inventory and demand planning for Big Mart.
* Future enhancements can include hyperparameter tuning, feature engineering, and comparison with other regression models for performance improvement.

This project demonstrates a complete, practical ML pipeline for retail sales forecasting using Python, pandas, and XGBoost, structured for clear learning and portfolio-ready presentation.
