from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import traceback
import os
import sys

app = Flask(__name__)

# Load models
try:
    lr_model = joblib.load("logistic_model.joblib")
    rf_model = joblib.load("Random_classifier.joblib")
    scaler = joblib.load("scaler.joblib")
    label_encoders = joblib.load("label_encoders.joblib")
    print("Models loaded successfully!")
    print("Available label encoders:", list(label_encoders.keys()))
    
    # Debug: Check model feature expectations
    if hasattr(lr_model, 'feature_names_in_'):
        print("Logistic Regression expects these features:", lr_model.feature_names_in_)
        print("Number of features expected:", len(lr_model.feature_names_in_))
    else:
        print("Logistic Regression model doesn't have feature_names_in_ attribute")
        print("Number of features expected by model:", lr_model.coef_.shape[1])
    
    if hasattr(rf_model, 'feature_names_in_'):
        print("Random Forest expects these features:", rf_model.feature_names_in_)
        print("Number of features expected:", len(rf_model.feature_names_in_))
    else:
        print("Random Forest model doesn't have feature_names_in_ attribute")
        print("Number of features expected by model:", rf_model.n_features_in_)
        
except Exception as e:
    print("Error loading models:", str(e))
    print("Current directory:", os.getcwd())
    print("Files in directory:", os.listdir())
    sys.exit(1)

@app.route("/")
@app.route("/home")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data as strings
        Gender = request.form.get("Gender")
        Married = request.form.get("Married")
        Dependents = request.form.get("Dependents")
        Education = request.form.get("Education")
        Self_Employed = request.form.get("Self_Employed")
        ApplicantIncome = float(request.form.get("ApplicantIncome", 0))
        CoapplicantIncome = float(request.form.get("CoapplicantIncome", 0))
        LoanAmount = float(request.form.get("LoanAmount", 1))
        Loan_Amount_Term = float(request.form.get("Loan_Amount_Term", 360))
        Credit_History = float(request.form.get("Credit_History", 1))
        Property_Area = request.form.get("Property_Area")

        print("Input data received:", Gender, Married, Dependents, Education, Self_Employed, Property_Area)

        # Validate required fields
        if not all([Gender, Married, Dependents, Education, Self_Employed, Property_Area]):
            return render_template("index.html", error_message="Please fill all required fields")

        # Create a dictionary with all features
        data_dict = {
            'Gender': Gender,
            'Married': Married,
            'Dependents': Dependents,
            'Education': Education,
            'Self_Employed': Self_Employed,
            'ApplicantIncome': ApplicantIncome,
            'CoapplicantIncome': CoapplicantIncome,
            'LoanAmount': LoanAmount,
            'Loan_Amount_Term': Loan_Amount_Term,
            'Credit_History': Credit_History,
            'Property_Area': Property_Area
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([data_dict])
        
        # Apply label encoding to categorical variables
        # Gender encoding
        if Gender in label_encoders['Gender'].classes_:
            input_df['Gender'] = label_encoders['Gender'].transform([Gender])[0]
        else:
            input_df['Gender'] = label_encoders['Gender'].transform([label_encoders['Gender'].classes_[0]])[0]
        
        # Married encoding
        if Married in label_encoders['Married'].classes_:
            input_df['Married'] = label_encoders['Married'].transform([Married])[0]
        else:
            input_df['Married'] = label_encoders['Married'].transform([label_encoders['Married'].classes_[0]])[0]
        
        # Education encoding
        if Education in label_encoders['Education'].classes_:
            input_df['Education'] = label_encoders['Education'].transform([Education])[0]
        else:
            input_df['Education'] = label_encoders['Education'].transform([label_encoders['Education'].classes_[0]])[0]
        
        # Self_Employed encoding
        if Self_Employed in label_encoders['Self_Employed'].classes_:
            input_df['Self_Employed'] = label_encoders['Self_Employed'].transform([Self_Employed])[0]
        else:
            input_df['Self_Employed'] = label_encoders['Self_Employed'].transform([label_encoders['Self_Employed'].classes_[0]])[0]
        
        # Handle Dependents (replace '3+' with 3)
        if Dependents == '3+':
            input_df['Dependents'] = 3
        else:
            input_df['Dependents'] = int(Dependents)
        
        # Create new features
        input_df["Total_Income"] = input_df["ApplicantIncome"] + input_df["CoapplicantIncome"]
        
        # Avoid division by zero
        safe_income = input_df["Total_Income"].values[0] + 0.01
        safe_term = input_df["Loan_Amount_Term"].values[0] + 0.01
        
        input_df['Loan_to_income_ratio'] = input_df['LoanAmount'] / safe_income
        input_df['EMI_to_income_ratio'] = (input_df['LoanAmount'] / safe_term) / safe_income
        
        # Create a copy for scaling
        scaled_df = input_df.copy()
        
        # Scale numerical features
        num_cols = [
            'Total_Income',
            'LoanAmount',
            'Loan_Amount_Term',
            'Loan_to_income_ratio',
            'EMI_to_income_ratio',
            'ApplicantIncome',
            'CoapplicantIncome'
        ]
        
        # Apply scaling
        scaled_values = scaler.transform(scaled_df[num_cols])
        for i, col in enumerate(num_cols):
            input_df[col] = scaled_values[0][i]
        
        # Create one-hot encoding for Property_Area
        input_df['Property_Area_Rural'] = 1 if Property_Area == 'Rural' else 0
        input_df['Property_Area_Semiurban'] = 1 if Property_Area == 'Semiurban' else 0
        
        # Based on your notebook, after dropping columns, you had these features
        # Let's create a DataFrame with the exact same column order as in training
        final_features_order = [
            'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
            'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
            'Total_Income', 'Loan_to_income_ratio', 'EMI_to_income_ratio',
            'Property_Area_Rural', 'Property_Area_Semiurban'
        ]
        
        # Create final input DataFrame with correct column names
        final_input = pd.DataFrame(index=[0])
        for col in final_features_order:
            final_input[col] = input_df[col].values[0]
        
        print("=" * 50)
        print("Final input columns:", final_input.columns.tolist())
        print("Final input values:", final_input.values.tolist())
        print("Final input shape:", final_input.shape)
        print("=" * 50)
        
        # If the model has feature_names_in_, check if our columns match
        if hasattr(lr_model, 'feature_names_in_'):
            expected_features = list(lr_model.feature_names_in_)
            print("Model expects:", expected_features)
            print("Our features:", final_input.columns.tolist())
            
            # Check if we have all expected features
            missing_features = set(expected_features) - set(final_input.columns)
            extra_features = set(final_input.columns) - set(expected_features)
            
            if missing_features:
                print(f"Missing features: {missing_features}")
            if extra_features:
                print(f"Extra features: {extra_features}")
            
            # Reorder columns to match model expectation
            final_input = final_input[expected_features]
            print("Reordered columns to match model expectation")
        
        # Make predictions
        lr_pred = lr_model.predict(final_input)[0]
        lr_prob = lr_model.predict_proba(final_input)[0][1]
        
        rf_pred = rf_model.predict(final_input)[0]
        rf_prob = rf_model.predict_proba(final_input)[0][1]
        
        return render_template(
            "index.html",
            lr_status="Approved" if lr_pred == 1 else "Rejected",
            lr_prob=round(lr_prob * 100, 2),
            rf_status="Approved" if rf_pred == 1 else "Rejected",
            rf_prob=round(rf_prob * 100, 2),
            # Pass back the input values to preserve form data
            gender=Gender,
            married=Married,
            dependents=Dependents,
            education=Education,
            self_employed=Self_Employed,
            applicant_income=ApplicantIncome,
            coapplicant_income=CoapplicantIncome,
            loan_amount=LoanAmount,
            loan_term=Loan_Amount_Term,
            credit_history=Credit_History,
            property_area=Property_Area
        )
        
    except Exception as e:
        error_msg = traceback.format_exc()
        print("ERROR in prediction:", error_msg)
        return render_template(
            "index.html",
            error_message=f"Prediction failed: {str(e)}"
        )

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)