# train_model.py
import re
import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score

# ---------- 0. Config ----------
DATA_PATH = "dataset.csv"   # change if needed
SAVE_MODEL_PATH = "vehicle_price_model.pkl"
CURRENT_YEAR = 2025         # use fixed year (your dataset year reference)

# ---------- 1. Load & quick EDA ----------
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print(df.head())

# Drop unnecessary columns if present
cols_to_remove = [col for col in ['name', 'description'] if col in df.columns]
if cols_to_remove:
    df.drop(columns=cols_to_remove, inplace=True)
    print(f"Dropped columns: {cols_to_remove}")

print(df.dtypes)
print("Missing values:\n", df.isnull().sum())

# ---------- 2. Basic cleaning ----------
# Ensure numeric columns are numeric
for col in ['mileage', 'cylinders', 'doors', 'year', 'price']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows where target is missing
df = df[~df['price'].isna()].copy()

# Fill numeric missing values
num_fill_cols = ['mileage', 'cylinders', 'doors']
for c in num_fill_cols:
    if c in df.columns:
        df[c] = df[c].fillna(df[c].median())

# Fill categorical missing values
cat_cols = ['make', 'model', 'engine', 'fuel', 'transmission', 'trim', 'body',
            'exterior_color', 'interior_color', 'drivetrain']
for c in cat_cols:
    if c in df.columns:
        df[c] = df[c].fillna('Unknown')

# ---------- 3. Feature engineering ----------
# Age
if 'year' in df.columns:
    df['age'] = CURRENT_YEAR - df['year']
    df['age'] = df['age'].clip(lower=0)

# Engine displacement parser
def parse_displacement(s):
    if not isinstance(s, str):
        return np.nan
    m = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*[lL]\b', s)
    if m:
        return float(m.group(1))
    m2 = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*cc\b', s)
    if m2:
        return float(m2.group(1)) / 1000.0
    return np.nan

if 'engine' in df.columns:
    df['engine_disp_l'] = df['engine'].apply(parse_displacement)
    df['engine_disp_l'] = df['engine_disp_l'].fillna(0)
    df['has_turbo'] = df['engine'].str.lower().str.contains('turbo', na=False).astype(int)

# Normalize mileage
if 'mileage' in df.columns:
    df['mileage'] = pd.to_numeric(df['mileage'].astype(str).str.replace(',', ''), errors='coerce')
    df['mileage'] = df['mileage'].fillna(df['mileage'].median())

# ---------- 4. Prepare features and target ----------
TARGET = 'price'
X = df.drop(columns=[TARGET])
y = df[TARGET].astype(float)

# Log-transform target
use_log_target = True
y_trans = np.log1p(y) if use_log_target else y

numeric_features = [c for c in ['age', 'mileage', 'engine_disp_l', 'cylinders', 'doors'] if c in X.columns]
categorical_features = [c for c in X.columns if c not in numeric_features]

print("Numeric features:", numeric_features)
print("Categorical features (count):", len(categorical_features))

# ---------- 5. Build preprocessing + model pipeline ----------
if int(sklearn.__version__.split(".")[1]) >= 2:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
else:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('onehot', ohe)
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
])

# ---------- 6. Train/test split ----------
X_train, X_test, y_train, y_test = train_test_split(X, y_trans, test_size=0.2, random_state=42)

# ---------- 7. Fit baseline model ----------
model.fit(X_train, y_train)

# ---------- 8. Evaluate ----------
def rmse(a, b): return np.sqrt(mean_squared_error(a, b))

y_pred_test = model.predict(X_test)
if use_log_target:
    y_test_orig = np.expm1(y_test)
    y_pred_test_orig = np.expm1(y_pred_test)
else:
    y_test_orig, y_pred_test_orig = y_test, y_pred_test

print("TEST RMSE:", rmse(y_test_orig, y_pred_test_orig))
print("TEST R2:", r2_score(y_test_orig, y_pred_test_orig))

# ---------- 9. Cross-validation ----------
cv_scores = cross_val_score(model, X, y_trans, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
cv_rmse = np.sqrt(-cv_scores)
print("CV RMSE mean:", cv_rmse.mean())

# ---------- 10. Hyperparameter tuning ----------
param_dist = {
    'regressor__n_estimators': [100, 200, 300, 400],
    'regressor__max_depth': [None, 8, 12, 20],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4],
    'regressor__max_features': ['sqrt', 'log2', None]  # fixed: removed 'auto'
}

search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=20,
                            scoring='neg_mean_squared_error', cv=3, verbose=1,
                            n_jobs=-1, random_state=42)
search.fit(X_train, y_train)

best_model = search.best_estimator_

y_pred_tuned = best_model.predict(X_test)
y_pred_tuned_orig = np.expm1(y_pred_tuned) if use_log_target else y_pred_tuned

print("TUNED TEST RMSE:", rmse(y_test_orig, y_pred_tuned_orig))
print("TUNED TEST R2:", r2_score(y_test_orig, y_pred_tuned_orig))

# ---------- 11. Save model ----------
joblib.dump(best_model, SAVE_MODEL_PATH)
print(f"Model saved to {SAVE_MODEL_PATH}")

# ---------- 12. Prediction function ----------
def predict_single(sample_dict):
    model_loaded = joblib.load(SAVE_MODEL_PATH)
    sample_df = pd.DataFrame([sample_dict])

    # Drop extra columns if present
    cols_to_remove = [col for col in ['name', 'description'] if col in sample_df.columns]
    if cols_to_remove:
        sample_df.drop(columns=cols_to_remove, inplace=True)

    # Derived features
    if 'year' in sample_df.columns:
        sample_df['age'] = CURRENT_YEAR - sample_df['year']
    if 'engine' in sample_df.columns:
        sample_df['engine_disp_l'] = sample_df['engine'].apply(parse_displacement).fillna(0)
        sample_df['has_turbo'] = sample_df['engine'].str.lower().str.contains('turbo', na=False).astype(int)

    pred_log = model_loaded.predict(sample_df)
    return float(np.expm1(pred_log[0])) if use_log_target else float(pred_log[0])

# ---------- 13. Example prediction ----------
example = {
    'make': 'Toyota',
    'model': 'Camry',
    'year': 2018,
    'engine': '2.5L',
    'cylinders': 4,
    'fuel': 'Gasoline',
    'mileage': 40000,
    'transmission': 'Automatic',
    'trim': 'LE',
    'body': 'Sedan',
    'doors': 4,
    'exterior_color': 'White',
    'interior_color': 'Black',
    'drivetrain': 'FWD'
}
print("Example predicted price USD:", predict_single(example))
