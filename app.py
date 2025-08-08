from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import re

# ---------- Config ----------
MODEL_PATH = "vehicle_price_model.pkl"
CURRENT_YEAR = 2025
USE_LOG_TARGET = True

# ---------- Load model ----------
model = joblib.load(MODEL_PATH)

# ---------- Helper function ----------
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

def predict_price(data_dict):
    # Convert dict to DataFrame
    df = pd.DataFrame([data_dict])

    # Drop unused columns if they appear
    for col in ['name', 'description']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Feature engineering
    if 'year' in df.columns:
        df['age'] = CURRENT_YEAR - df['year']
    if 'engine' in df.columns:
        df['engine_disp_l'] = df['engine'].apply(parse_displacement).fillna(0)
        df['has_turbo'] = df['engine'].str.lower().str.contains('turbo', na=False).astype(int)

    # Predict
    pred_log = model.predict(df)
    if USE_LOG_TARGET:
        return float(np.expm1(pred_log[0]))
    return float(pred_log[0])

# ---------- Flask App ----------
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = {
            'make': request.form['make'],
            'model': request.form['model'],
            'year': int(request.form['year']),
            'engine': request.form['engine'],
            'cylinders': float(request.form['cylinders']),
            'fuel': request.form['fuel'],
            'mileage': float(request.form['mileage']),
            'transmission': request.form['transmission'],
            'trim': request.form['trim'],
            'body': request.form['body'],
            'doors': float(request.form['doors']),
            'exterior_color': request.form['exterior_color'],
            'interior_color': request.form['interior_color'],
            'drivetrain': request.form['drivetrain']
        }

        predicted_price = predict_price(data)
        return render_template("index.html", prediction_text=f"Predicted Price: ${predicted_price:,.2f}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

# ---------- Run ----------
if __name__ == "__main__":
    app.run(debug=True)
