# California Housing Project

This repository contains a structured machine learning workflow for predicting housing prices in California.  
The project follows industry-standard practices for organization, reproducibility, and clarity.

---

## Project Structure

ca_housing_project
├── data
│ ├── raw # Original dataset (never modified)
│ ├── train # Stratified train splits (raw and processed)
│ └── test # Stratified test split
├── images # Visualizations and plots generated during analysis
├── models # Trained models and modeling notebooks
├── analysis # Initial and exploratory data analysis notebooks
└── README.md # Project documentation


---

## Contents

### /data
- **raw/**: Contains the original `housing.csv`. This file is never modified.  
- **train/**: Includes raw and processed training splits.  
  - `housing_train.csv` → raw split with 10 features  
  - `housing_train_processed.csv` → processed dataset with 24 engineered features  
- **test/**: Contains the stratified test split (`housing_test.csv`).  

### /analysis
- **ida.ipynb**: Initial Data Analysis (IDA)  
  - Data loading, type analysis, stratified splitting, saving raw splits.  
- **eda.ipynb**: Exploratory Data Analysis (EDA)  
  - Geographic visualizations, correlation analysis, feature engineering, saving processed dataset.  
- **preprocessing_pipeline.py**: Script that builds a preprocessing pipeline (imputation, scaling, encoding) and saves the final processed training set.  

### /models
- **LinearRegression.ipynb**: Fits and evaluates a linear regression model.  
- **DecisionTree.ipynb**: Fits and evaluates a decision tree regressor, with cross-validation and hyperparameter tuning.  
- **RandomForest.ipynb**: Fits and evaluates a random forest regressor, with cross-validation and hyperparameter tuning.  
- **SVR.ipynb**: Fits and evaluates a support vector regression model (RBF kernel).  

Trained models are saved in this directory as `.pkl` files (e.g., `linear_regression_model.pkl`).  

---

## Workflow

1. **Initial Data Analysis (IDA)**  
   - Load raw dataset  
   - Analyze datatypes, missing values  
   - Stratified train/test split  
   - Save splits to `/data/train` and `/data/test`  

2. **Exploratory Data Analysis (EDA)**  
   - Geographic visualization of housing data  
   - Correlation heatmaps  
   - Feature engineering and one-hot encoding  
   - Save processed training dataset with 24 features  

3. **Preprocessing Pipeline**  
   - Encodes categorical features, imputes missing values, scales numerics  
   - Saves pipeline object for reuse  

4. **Modeling**  
   - Linear Regression, Decision Tree, Random Forest, and SVR  
   - Each model includes:  
     - Data loading  
     - Model fitting  
     - Cross-validation  
     - Hyperparameter tuning (where applicable)  
     - Model saving  

---

## How to Run

1. Clone this repository.  
2. Place the raw dataset in `/data/raw/housing.csv`.  
3. Run `analysis/ida.ipynb` → generates train/test splits.  
4. Run `analysis/eda.ipynb` → generates processed training dataset.  
5. Run any model notebook in `/models` to train and save models.  

---

## Notes
- File paths are relative to the project root.  
- All notebooks are designed to run end-to-end without errors.  
- Processed datasets contain 24 features including engineered ratios and one-hot encoded categorical variables.  

