# 🚗 Car Price Prediction — CarDekho Dataset

> **Internship Project** | Data Analytics with AI  
> **Author:** Vaishnavi  
> **Task:** Regression — Predict the resale/selling price of used cars

---

## 📌 Project Overview

This project builds a complete end-to-end machine learning pipeline to predict the selling price of used cars listed on CarDekho. It covers data cleaning, exploratory data analysis (EDA), feature engineering, model training and evaluation, and deployment via a Flask REST API and Streamlit interactive frontend.

---

## 📂 Dataset Information

| Attribute | Details |
|-----------|---------|
| **File** | `CAR DETAILS FROM CAR DEKHO.csv` |
| **Source** | [CarDekho Used Car Dataset — Kaggle](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho) |
| **Rows** | 4,340 (after removing 763 exact duplicates) |
| **Columns** | 8 |
| **Target Column** | `selling_price` (continuous — regression) |

### Column Descriptions

| Column | Type | Description |
|--------|------|-------------|
| `name` | String | Full car model name |
| `year` | Integer | Manufacturing year |
| `selling_price` | Integer | Resale price in INR (**target**) |
| `km_driven` | Integer | Total kilometres driven |
| `fuel` | Categorical | Fuel type (Petrol/Diesel/CNG/LPG/Electric) |
| `seller_type` | Categorical | Individual / Dealer / Trustmark Dealer |
| `transmission` | Categorical | Manual / Automatic |
| `owner` | Categorical | First / Second / Third / Fourth & Above / Test Drive |

---

## 🛠️ Technologies Used

| Layer | Technology |
|-------|-----------|
| Data & EDA | `pandas`, `numpy`, `matplotlib`, `seaborn` |
| Machine Learning | `scikit-learn`, `xgboost` |
| Model Persistence | `joblib` |
| Backend API | `Flask`, `Flask-CORS` |
| Frontend | `Streamlit` |
| HTTP Client | `requests` |

---

## 📁 Project Structure

```
Data_Analytics_with_AI_Internship/
│
├── CAR DETAILS FROM CAR DEKHO.csv      # Source dataset
│
├── Vaishnavi_CarPricePrediction.ipynb  # Main notebook (EDA + ML)
├── requirements.txt                    # All Python dependencies
├── Vaishnavi_ProjectReport.docx        # Complete project report
├── README.md                           # This file
│
├── app.py                              # Flask backend API
├── streamlit_app.py                    # Streamlit frontend
│
└── models/
    └── best_model.pkl                  # Saved best model pipeline
```

---

## ⚙️ Installation

### 1. Clone / open the project folder

```bash
cd "Data_Analytics_with_AI_Internship"
```

### 2. (Recommended) Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install all dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Step 1 — Run the Jupyter Notebook (generates the trained model)

```bash
jupyter notebook Vaishnavi_CarPricePrediction.ipynb
```

- Run all cells top-to-bottom.
- This will train all models, print real MAE / RMSE / R² metrics, and save `models/best_model.pkl`.

### Step 2 — Start the Flask Backend API

```bash
python app.py
```

The API will start at: `http://localhost:5000`

### Step 3 — Start the Streamlit Frontend

Open a **new terminal** (keep Flask running) and run:

```bash
streamlit run streamlit_app.py
```

Streamlit will open automatically at: `http://localhost:8501`

---

## 🔌 Flask API Usage

### Endpoint

```
POST http://localhost:5000/predict
Content-Type: application/json
```

### Request Body

```json
{
    "brand": "Maruti",
    "km_driven": 50000,
    "fuel": "Petrol",
    "seller_type": "Individual",
    "transmission": "Manual",
    "owner": "First Owner",
    "car_age": 7
}
```

### Response

```json
{
    "predicted_price": 385000.0,
    "currency": "INR",
    "model_used": "GradientBoostingRegressor"
}
```

### Health Check

```
GET http://localhost:5000/health
```

---

## 🤖 Models Trained & Compared

| Model | Description |
|-------|-------------|
| Linear Regression | Baseline linear model |
| Random Forest Regressor | Ensemble of decision trees |
| Gradient Boosting Regressor | Sequential boosting ensemble |

Evaluation metrics used: **MAE**, **RMSE**, **R²**  
Best model selected by highest **R² score** on the 20% test set.

---

## 📊 Features Used for Prediction

| Feature | Engineering |
|---------|-------------|
| `brand` | Extracted from first word of `name` column |
| `car_age` | `2024 - year` |
| `km_driven` | Used directly (scaled) |
| `fuel` | One-Hot Encoded |
| `seller_type` | One-Hot Encoded |
| `transmission` | One-Hot Encoded |
| `owner` | One-Hot Encoded |

**Target:** `log1p(selling_price)` during training; inverse-transformed (`expm1`) for output.

---

## 📄 Submission Files

1. `Vaishnavi_CarPricePrediction.ipynb` — Complete analysis notebook
2. `requirements.txt` — Python dependencies
3. `Vaishnavi_ProjectReport.docx` — Full project report
4. `README.md` — This file

---

## 📝 Notes

- The dataset file (`CAR DETAILS FROM CAR DEKHO.csv`) must be in the same directory as the scripts.
- Run the notebook **before** starting the Flask API so `models/best_model.pkl` is created.
- The Streamlit app requires the Flask API to be running for predictions to work.

---

<p align="center">Made with ❤️ | Data Analytics with AI Internship | Vaishnavi</p>
