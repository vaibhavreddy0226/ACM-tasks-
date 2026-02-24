## Loan Approval Prediction System
A machine learning model that predicts loan approval status based on applicant information using Logistic Regression and Random Forest classifiers.

---
## Dataset Features  

- Categorical: Gender, Married, Dependents, Education, Self_Employed, Property_Area
- Numerical: ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History
- Target: Loan_Status (Approved/Rejected)

---
### Data Preparation & Preprocessing
- **Input length**: 615 rows
- **Target variable**: Loan_status, Loan_approval_probability
- **Features used** (13 total after selection):Gender,	Married,	Dependents,	Education,	Self_Employed,	LoanAmount,	Loan_Amount_Term,	Credit_History,	Loan_Status,	Total_Income,	Loan_to_income_ratio,	Property_Area_Rural,	Property_Area_Semiurban

**Preprocessing steps**:
## Handling Missing Values:
- Categorical: Filled with mode (e.g., Gender → 'Male', Self_Employed → 'No').
- Numerical: LoanAmount → median, Loan_Amount_Term → mode (360), Credit_History → mode (1.0).

## Encoding:
- Binary categorical: LabelEncoder (e.g., Gender: Male=1/Female=0, Married: Yes=1/No=0).
- Ordinal: Mapped (e.g., Dependents: '0'=0, '1'=1, '2'=2, '3+'=3; Education: Graduate=0/Not Graduate=1).
- One-hot: Property_Area (Rural, Semiurban, Urban → dummy columns).

## Feature Engineering:
- Total_Income = ApplicantIncome + CoapplicantIncome (logged for skewness).
- Loan_to_income_ratio = LoanAmount / Total_Income (logged).

- Scaling: Numerical features (LoanAmount, Loan_Amount_Term, Total_Income, Loan_to_income_ratio) standardized using StandardScaler.
- Drop Columns: Loan_ID (irrelevant).
- Final Features: 13 columns after preprocessing.
- Train/Test Split: 80% train / 20% test (random_state=42 for reproducibility).

--- 
## Training Setup
- Framework: scikit-learn (sklearn).
- Loss/Metric: Not explicit (sklearn uses log-loss for logistic; Gini for RF).
- Optimizer: Default solvers (liblinear for logistic; bootstrap for RF).
- Train Size: 80% of data (random_state=42).
- No Hyperparameter Tuning: Models trained with defaults; no grid search or cross-validation.
- Evaluation Metrics:
- Accuracy, Precision, Recall, F1 Score, ROC-AUC (using sklearn.metrics).

Model Saving: Both models saved as .joblib files (logistic_model.joblib, Random_classifier.joblib) using joblib.dump.

### Results Snapshot (from training logs)
| Model       | Accuracy | Precision | Recall | F1 Score | ROC - AUC |
|-------------|----------|-----------|--------|----------|-----------|
| Logistic Regression | ~0.808 | ~0.809 | ~0.948 | ~0.873 | ~0.766    |
| Random Forest     | ~0.772   | ~0.810 | ~0.879 | ~0.843 | ~0.754    |

---
**Note**: To view the web part navigate to [web_loan_pred folder](https://github.com/vaibhavreddy0226/ACM-tasks-/tree/main/Loan_pred/web_loan_pred) and Read the Readme file to view the web part
