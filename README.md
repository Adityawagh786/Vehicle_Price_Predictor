# Vehicle_Price_Predictor
The Vehicle Price Predictor uses machine learning to estimate a car’s market value based on details like make, model, year, engine, mileage, and more. With a simple web form and instant results, it helps buyers, sellers, and dealers make informed pricing decisions quickly and accurately.          Ask ChatGPT
# 🚗 Vehicle Price Predictor

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-2.0+-black?logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-orange?logo=scikit-learn)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

> A **Machine Learning–powered web app** that predicts the market price of a vehicle based on its features.  
> Built with **Python, Flask, HTML, CSS, and scikit-learn**.

---

## 📋 Table of Contents
- [✨ Features](#-features)
- [📊 Dataset](#-dataset)
- [🛠 Tech Stack](#-tech-stack)
- [📂 Project Structure](#-project-structure)
- [⚙ Installation](#-installation)
- [▶ Usage](#-usage)
- [📌 Example Predictions](#-example-predictions)
- [🎯 Customization](#-customization)
- [🖼 Screenshots](#-screenshots)
- [📜 License](#-license)

---

## ✨ Features
✅ Predicts prices using **make, model, year, engine, fuel, mileage, trim, body type, colors, drivetrain**, etc.  
✅ **Interactive form** for easy input.  
✅ **Instant predictions** without page reload.  
✅ Preprocessing pipeline with **feature engineering**.  
✅ Works with **custom datasets**.  
✅ Includes **pre-trained RandomForest model**.

---

## 📊 Dataset
The training dataset contains:
- **Make & Model** — brand and specific model.
- **Year** — manufacturing year.
- **Engine Specs** — capacity, turbo, electric, etc.
- **Mileage** — driven distance.
- **Fuel & Transmission** — type and gear system.
- **Trim & Body** — variant and style.
- **Colors** — exterior and interior.
- **Drivetrain** — FWD, AWD, 4WD.
- **Price** — actual selling price (target variable).

---

## 🛠 Tech Stack
| Layer       | Technology |
|-------------|------------|
| **Backend** | Python, Flask |
| **ML Model**| scikit-learn, pandas, numpy |
| **Frontend**| HTML5, CSS3 |
| **Storage** | joblib `.pkl` file |
| **Model**   | RandomForestRegressor |



---

## 📂 Project Structure
vehicle_price_predictor/<br>
│── app.py # Flask backend<br>
│── train_model.py # ML training script<br>
│── vehicle_price_model.pkl # Saved trained model<br>
│── templates/<br>
│ └── index.html # Frontend form<br>
│── static/<br>
│ └── car.svg # Logo/icon<br>
│── dataset.csv # Dataset<br>
│── requirements.txt # Dependencies<br>
│── README.md # Documentation<br>


---

## ⚙ Installation

1️⃣ Clone the repository:
git clone  https://github.com/Adityawagh786/Vehicle_Price_Predictor.git
cd Vehicle_Price_Predictor

2️⃣ Clone the repository:
pip install -r requirements.txt

3️⃣ (Optional) Retrain the model:
python train_model.py

4️⃣ Run the Flask app:
python app.py

5️⃣ Visit:
http://127.0.0.1:5000/

▶ Usage
1>Open the web form.

2>Fill in vehicle details.

3>Click Predict Price.

4>Get an instant price estimate.
<br>
📌 Example Predictions
Make	Model	Year	Engine	Mileage	Fuel	Predicted Price
Toyota	Camry	2018	2.5L	40000	Gasoline	$16,500
Tesla	Model 3	2022	Electric	5000	Electric	$44,200
Ford	F-150	2020	3.5L	25000	Gasoline	$38,900
<br>

🎯 Customization
*Replace dataset.csv with your own dataset.
*Modify train_model.py for a different ML algorithm.
*Change currency format in app.py.
<br>


🙌 Credits:<br>
Developed by Aditya Wagh using Python, Flask, and scikit-learn.


