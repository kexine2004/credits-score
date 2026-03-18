import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import pickle

# Load the pre-trained model and scaler
try:
    with open('gb_model (2).pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scaler (1).pkl', 'rb') as file:
        scaler = pickle.load(file)
except FileNotFoundError:
    messagebox.showerror("Error", "Model files 'gb_model.pkl' and 'scaler.pkl' not found. Please ensure they are in the same directory as the script.")
    exit()

def create_scorecard(prob_of_default):
    """
    Converts a probability of default (0-1) into a simplified credit score (300-850).
    """
    score = 850 - (prob_of_default * 550)
    score = np.clip(score, 300, 850)
    return int(score)

def sanitize_float_input(input_string):
    """
    Sanitizes the input string for float conversion by removing
    non-numerical characters (except for a single period).
    """
    if input_string is None:
        return ""
    sanitized_str = ''.join(c for c in input_string if c.isdigit() or c == '.')
    # Handle multiple periods by keeping only the first one
    if sanitized_str.count('.') > 1:
        parts = sanitized_str.split('.', 1)
        sanitized_str = parts[0] + '.' + parts[1].replace('.', '')
    return sanitized_str.strip()

def sanitize_int_input(input_string):
    """
    Sanitizes the input string for integer conversion by keeping only digits.
    """
    if input_string is None:
        return ""
    sanitized_str = ''.join(c for c in input_string if c.isdigit())
    return sanitized_str.strip()

def predict_risk():
    """
    Handles the prediction logic with improved input sanitization.
    """
    try:
        # Sanitize and get input values from the UI
        limit_bal_str = sanitize_float_input(entry_limit.get())
        age_str = sanitize_int_input(entry_age.get())
        
        # Check for empty sanitized strings
        if not limit_bal_str or not age_str:
            messagebox.showerror("Invalid Input", "Please fill in all the required fields (Limit Balance and Age).")
            return
            
        limit_bal = float(limit_bal_str)
        sex = int(combo_sex.get().split('=')[0].strip())
        education = int(combo_education.get().split('=')[0].strip())
        marriage = int(combo_marriage.get().split('=')[0].strip())
        age = int(age_str)
        
        # Create a DataFrame with the input data.
        # It is CRITICAL that the column order and names match the training data.
        input_data = pd.DataFrame([[
            limit_bal, sex, education, marriage, age,
            0, 0, 0, 0, 0, 0, # PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6
            0, 0, 0, 0, 0, 0, # BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6
            0, 0, 0, 0, 0, 0, # PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6
        ]], columns=[
            'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
            'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
            'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
            'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
        ])
        
        # This is the crucial fix: calculate TOTAL_BILL_AMT and add it to the DataFrame
        # to match the feature set of the trained model.
        input_data['TOTAL_BILL_AMT'] = input_data[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']].sum(axis=1)

        # Standardize the input data
        scaled_input = scaler.transform(input_data)
        
        # Get the default probability from the model
        prob_of_default = model.predict_proba(scaled_input)[:, 1][0]
        
        # Calculate the credit score
        credit_score = create_scorecard(prob_of_default)
        
        # Determine the risk segment
        if credit_score >= 700:
            risk_segment = "Low Risk"
        elif credit_score >= 600:
            risk_segment = "Medium Risk"
        else:
            risk_segment = "High Risk"
            
        # Update the UI labels
        label_prob_val.config(text=f"{prob_of_default:.4f}")
        label_score_val.config(text=f"{credit_score}")
        label_risk_val.config(text=risk_segment)
        
    except ValueError as e:
        # Provide a more specific error message for common issues
        messagebox.showerror("Invalid Input", f"An error occurred during data processing. This may be due to a mismatch in the number or order of data columns.\n\nError details: {e}")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")

# --- Set up the main application window ---
root = tk.Tk()
root.title("Credit Risk Predictor")
root.geometry("400x400")
root.resizable(False, False)

# Style for the widgets
style = ttk.Style(root)
style.configure("TLabel", font=("Helvetica", 12))
style.configure("TButton", font=("Helvetica", 12))
style.configure("TCombobox", font=("Helvetica", 12))

# --- Create and place the input widgets ---
frame_input = ttk.Frame(root, padding="20")
frame_input.pack(fill="both", expand=True)

# Limit Balance
label_limit = ttk.Label(frame_input, text="Limit Balance:")
label_limit.grid(row=0, column=0, sticky=tk.W, pady=5)
entry_limit = ttk.Entry(frame_input, width=30)
entry_limit.grid(row=0, column=1, pady=5)

# Sex
label_sex = ttk.Label(frame_input, text="Sex (1=M, 2=F):")
label_sex.grid(row=1, column=0, sticky=tk.W, pady=5)
combo_sex = ttk.Combobox(frame_input, values=["1 = M", "2 = F"], state="readonly", width=28)
combo_sex.grid(row=1, column=1, pady=5)
combo_sex.current(0)

# Education
label_education = ttk.Label(frame_input, text="Education:")
label_education.grid(row=2, column=0, sticky=tk.W, pady=5)
combo_education = ttk.Combobox(frame_input, values=[
    "1 = graduate school", "2 = university", "3 = high school",
    "4 = others", "5 = unknown", "6 = unknown"
], state="readonly", width=28)
combo_education.grid(row=2, column=1, pady=5)
combo_education.current(0)

# Marriage
label_marriage = ttk.Label(frame_input, text="Marriage:")
label_marriage.grid(row=3, column=0, sticky=tk.W, pady=5)
combo_marriage = ttk.Combobox(frame_input, values=[
    "1 = married", "2 = single", "3 = others"
], state="readonly", width=28)
combo_marriage.grid(row=3, column=1, pady=5)
combo_marriage.current(0)

# Age
label_age = ttk.Label(frame_input, text="Age:")
label_age.grid(row=4, column=0, sticky=tk.W, pady=5)
entry_age = ttk.Entry(frame_input, width=30)
entry_age.grid(row=4, column=1, pady=5)

# Predict button
button_predict = ttk.Button(root, text="Predict Risk", command=predict_risk)
button_predict.pack(pady=10)

# --- Create and place the output widgets ---
frame_output = ttk.Frame(root, padding="20")
frame_output.pack(fill="both", expand=True)

# Default Probability
label_prob = ttk.Label(frame_output, text="Default Probability:", font=("Helvetica", 12, "bold"))
label_prob.grid(row=0, column=0, sticky=tk.W, pady=5)
label_prob_val = ttk.Label(frame_output, text="", font=("Helvetica", 12))
label_prob_val.grid(row=0, column=1, sticky=tk.W, pady=5)

# Credit Score
label_score = ttk.Label(frame_output, text="Credit Score:", font=("Helvetica", 12, "bold"))
label_score.grid(row=1, column=0, sticky=tk.W, pady=5)
label_score_val = ttk.Label(frame_output, text="", font=("Helvetica", 12))
label_score_val.grid(row=1, column=1, sticky=tk.W, pady=5)

# Risk Segment
label_risk = ttk.Label(frame_output, text="Risk Segment:", font=("Helvetica", 12, "bold"))
label_risk.grid(row=2, column=0, sticky=tk.W, pady=5)
label_risk_val = ttk.Label(frame_output, text="", font=("Helvetica", 12))
label_risk_val.grid(row=2, column=1, sticky=tk.W, pady=5)

root.mainloop()
