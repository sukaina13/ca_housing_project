# ca_housing_project/analysis/preprocessing_pipeline.py

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

# -------------------------------------------------------------------
# 1. Paths
PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_PATH = os.path.join(PROJECT_DIR, "data", "raw", "housing.csv")

TRAIN_RAW_PATH = os.path.join(PROJECT_DIR, "data", "train", "housing_train.csv")
TRAIN_PROCESSED_PATH = os.path.join(PROJECT_DIR, "data", "train", "housing_train_processed.csv")
TEST_RAW_PATH = os.path.join(PROJECT_DIR, "data", "test", "housing_test.csv")

PIPELINE_PATH = os.path.join(PROJECT_DIR, "models", "preprocessing_pipeline.pkl")

# -------------------------------------------------------------------
# 2. Load dataset
housing = pd.read_csv(RAW_PATH)
print("Dataset shape:", housing.shape)
print("Columns:", housing.columns.tolist())

# -------------------------------------------------------------------
# 3. Train/test split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# -------------------------------------------------------------------
# 4. Add engineered features (for raw splits too)
for df in (train_set, test_set):
    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
    df["population_per_household"] = df["population"] / df["households"]

# ✅ Save raw splits with engineered features (13 columns each)
train_set.to_csv(TRAIN_RAW_PATH, index=False)
test_set.to_csv(TEST_RAW_PATH, index=False)
print(f"Saved raw train: {TRAIN_RAW_PATH} (shape={train_set.shape})")
print(f"Saved raw test: {TEST_RAW_PATH} (shape={test_set.shape})")

# -------------------------------------------------------------------
# 5. Extra engineered features for processed train
train_processed = train_set.copy()
train_processed["log_median_income"] = np.log(train_processed["median_income"] + 1)
train_processed["log_population"] = np.log(train_processed["population"] + 1)
train_processed["rooms_per_person"] = train_processed["total_rooms"] / (train_processed["population"] + 1)
train_processed["bedrooms_per_person"] = train_processed["total_bedrooms"] / (train_processed["population"] + 1)
train_processed["income_per_room"] = train_processed["median_income"] / (train_processed["total_rooms"] + 1)
train_processed["population_per_bedroom"] = train_processed["population"] / (train_processed["total_bedrooms"] + 1)
train_processed["age_income_ratio"] = train_processed["housing_median_age"] / (train_processed["median_income"] + 1)

# -------------------------------------------------------------------
# 6. Separate predictors and target
housing_train = train_processed.drop("median_house_value", axis=1)   # predictors
housing_labels = train_processed["median_house_value"].copy()        # target

numeric_features = housing_train.drop("ocean_proximity", axis=1).columns
categorical_features = ["ocean_proximity"]

# -------------------------------------------------------------------
# 7. Define preprocessing pipeline
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

full_pipeline = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# -------------------------------------------------------------------
# 8. Fit pipeline and transform training data
housing_prepared = full_pipeline.fit_transform(housing_train)
housing_prepared = np.array(housing_prepared)

# Build DataFrame with column names
cat_onehot = list(full_pipeline.named_transformers_["cat"]["onehot"].get_feature_names_out(categorical_features))
all_features = list(numeric_features) + cat_onehot
housing_prepared_df = pd.DataFrame(housing_prepared, columns=all_features)

# ✅ Add target back
housing_prepared_df["median_house_value"] = housing_labels.reset_index(drop=True)

# -------------------------------------------------------------------
# 9. Save processed training set
housing_prepared_df.to_csv(TRAIN_PROCESSED_PATH, index=False)
print(f"Saved processed train: {TRAIN_PROCESSED_PATH} (shape={housing_prepared_df.shape})")

# -------------------------------------------------------------------
# 10. Save pipeline
joblib.dump(full_pipeline, PIPELINE_PATH)
print(f"Pipeline saved at {PIPELINE_PATH}")
