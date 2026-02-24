# ML Web App: Logistic & Random Classifier 

This project is a local web application that serves pre-trained Machine Learning models. It provides a user-friendly interface to get predictions from a **Logistic Regression** model and a **Random classifier** without needing to interact with the code directly.

## Features
- **Instant Predictions:** Input data via a web form and get results immediately.
- **Pre-processing Included:** Uses saved `scaler` and `label_encoders` to ensure data consistency.
- **Model:** Model is loaded using the `Joblib` to store the model parameters so no need of training each time for prediction. 
- **Lightweight Frontend:** Simple, responsive design using HTML5 and CSS.

## Project Structure
```
web_loan_pred/
├── app.py                  # Flask/Web server logic
├── logistic_model.joblib    # Trained Logistic Regression model
├── Random_classifier.joblib # Trained Random Forest/Linear model
├── scaler.joblib            # Fitted StandardScaler
├── label_encoders.joblib    # Saved LabelEncoders
├── templates/
│   └── index.html           # Webpage structure
└── static/
    └── style.css            # Styling & Layout
```
---
Follow these steps to get the project running on your local machine:

## 1. Clone the Repository
```bash
git clone https://github.com//vaibhavreddy0226/ACM-tasks-
cd Loan_pred/web_loan_pred
```

## 2. Install Requirements
Make sure you have Python installed. Run the following to install the necessary libraries:
```bash
pip install flask joblib scikit-learn pandas numpy
```

## 3. Run the Application
Start the local server by executing:
```bash
python app.py
```

## 4. View in Browser
Once the terminal shows the server is active, open your browser and go to:
http://127.0.0.1:5000


## Usage  
Enter the required values into the input fields on the webpage.  
Click Predict to see the results calculated by the `Logistic` and `Random Classifier` models.

---
Note: Ensure all .joblib files remain in the root directory for the app to load the models successfully.
