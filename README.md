# 💳 Credit Risk Prediction System

A machine learning-powered web application built with **Streamlit** that predicts credit default probability, generates a credit score, and classifies applicants into risk segments.

---

## 🚀 Features

- 🔍 **Default Probability** — Predicts the likelihood of a credit default
- 📊 **Credit Score** — Maps probability to a 300–850 scorecard range
- 🚦 **Risk Segmentation** — Classifies into Low / Medium / High risk
- ⚡ **Real-time Inference** — Instant predictions via a pre-trained Gradient Boosting model

---

## 🧠 Model Details

| Component | Details |
|-----------|---------|
| Algorithm | Gradient Boosting Classifier |
| Scaler | Standard Scaler (pre-fitted) |
| Input Features | 23 (demographic + payment history + billing amounts) |
| Engineered Feature | `TOTAL_BILL_AMT` (sum of 6 billing months) |

### Credit Score Formula
```
Credit Score = 850 - (Default Probability × 550)
Clamped between 300 and 850
```

### Risk Thresholds

| Score Range | Risk Level |
|-------------|------------|
| ≥ 700 | 🟢 Low Risk |
| 600 – 699 | 🟡 Medium Risk |
| < 600 | 🔴 High Risk |

---

## 🗂️ Project Structure
```
credit-risk-predictor/
│
├── app.py                 # Main Streamlit application
├── gb_model (2).pkl       # Pre-trained Gradient Boosting model
├── scaler (1).pkl         # Pre-fitted Standard Scaler
├── requirements.txt       # Python dependencies
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/credit-risk-predictor.git
cd credit-risk-predictor
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Add Model Files

Place the following files in the root directory:
- `gb_model (2).pkl`
- `scaler (1).pkl`

### 5. Run the App
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📋 Requirements
```
streamlit
pandas
numpy
scikit-learn
```

> Generate with: `pip freeze > requirements.txt`

---

## 🖥️ Input Features

| Feature | Description |
|---------|-------------|
| `LIMIT_BAL` | Credit limit amount |
| `AGE` | Applicant age (18–100) |
| `SEX` | 1 = Male, 2 = Female |
| `EDUCATION` | 1 = Graduate School, 2 = University, 3 = High School, 4+ = Others |
| `MARRIAGE` | 1 = Married, 2 = Single, 3 = Others |
| `PAY_0` – `PAY_6` | Repayment status (past 6 months) |
| `BILL_AMT1` – `BILL_AMT6` | Bill statement amounts (past 6 months) |
| `PAY_AMT1` – `PAY_AMT6` | Previous payment amounts (past 6 months) |

---

## 📊 Sample Output
```
Default Probability   Credit Score   Risk Segment
      0.3214              673          🟡 Medium Risk
```

---

## 📄 Dataset Reference

Based on the [UCI Credit Card Default Dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients).

---


---

## 📃 License

This project is licensed under the [MIT License](LICENSE).
