# Loan Fraud Detection

This project provides an end-to-end pipeline for loan fraud detection using Ensemble model, with a FastAPI interface and SHAP explanations for fraud predictions.

## Features
- Data processing with feature engineering and robust one-hot encoding
- Fraud detection model (Ensemble of LightGBM + XGBoost + Random Forest)
- Rule-based credit worthiness logic
- FastAPI endpoint for predictions, with file uploads for document verification
- SHAP explanations for fraud predictions
- LLM (OpenAI) natural language explanations for fraud cases
- Comprehensive logging with request traceability
- Functionality to To check for data and prediction drift between your training data and recent production data

## Setup
```bash
pip install -r requirements.txt
```

## Training
1. Prepare your training CSV with the required columns, including `is_fraud` label.
2. Train the fraud detection model:
```bash
python -m src.train.train_fraud_model --train_csv data/training_data.csv --model_path models/fraud_model_lgbm.pkl
```


## Running the API
```bash
uvicorn src.api:app --reload
```

## API Usage

**POST** `/predict`

- **Request body:** JSON with application data (see fields below) and four required PDF uploads (`aadhar_card`, `pan_card`, `bank_statement`, `salary_slip`).
- **Response:** Fraud prediction, credit worthiness (if not fraud), SHAP explanation (if fraud), and a natural language LLM based explanation (if fraud).

**Request JSON fields:**
```json
{
  "age": 35,
  "gender": "male",
  "marital_status": "single",
  "residential_status": "owned",
  "employment_stats": "employed",
  "monthly_income": 50000,
  "other_income_source": "none",
  "total_monthly_expenses": 20000,
  "number_of_loans": 1,
  "cibil_score": 750,
  "loan_amount_requested": 300000,
  "loan_purpose": "home",
  "loan_term_in_months": 60
}
```
**File uploads:**  
- `aadhar_card` (PDF), `pan_card` (PDF), `bank_statement` (PDF), `salary_slip` (PDF)

**Response fields:**
- `is_fraud` (bool)
- `credit_worthiness` (float)
- `shap_explanation` (dict, if fraud)
- `llm_explanation` (str, if fraud)
- `error` (str, if any error occurs)

## Model Monitoring

### Drift Detection
Need training data and production logs 
To check for data and prediction drift between your training data and recent production data:

```bash
python monitoring/drift_check.py
```

### Periodic Model Performance Monitoring
To track model performance metrics over time.
Tracking the following metrics:
-Precision
-Recall
-F1 Score
-ROC AUC
```bash
python monitoring/periodic_model_monitoring.py
```

## Project Structure
- `src/` - Main application package
  - `api.py` - FastAPI app
  - `predict/` - Prediction logic
    - `predict.py` - Main prediction logic
    - `explain.py` - SHAP + LLM explanation
  - `data/` - Data processing and document verification
    - `data_processing.py` - Data cleaning & feature engineering
    - `document_verification.py` - Document verification logic
  - `train/` - Model training scripts
    - `train_fraud_model.py`
    - `train_credit_model.py`
  - `utils/` - Utility modules (e.g., Azure Form Recognizer, logger)
    - `logger.py`
    - `form_recognizer.py`
  - `__init__.py` (and in all subfolders)
- `models/` - Trained model binaries and encoders
- `data/` - Data files
- `monitoring/` - Monitoring scripts and reports
  - `drift_check.py` - Data drift detection
  - `periodic_model_monitoring.py` - Performance tracking
  - `predictions_log.csv` - API request logs
- `loan_fraud_detection.log` - Log file

## **Assumptions & Important Notes**

- **Input Schema:** The input JSON and file uploads must match the fields and types in `src/api.py`'s `PredictionRequest` model.
- **Document Verification:** The API expects **all four documents** (Aadhaar, PAN, bank statement, salary slip) as PDF uploads for each prediction.
- **Categorical Encoding:** OneHotEncoder is fit on training data and reused for inference.
- **Model Loading:** Models are loaded **once at API startup** and reused for all requests for efficiency.
- **Rule-based Credit Worthiness:** If the application is not fraud, a rule-based function determines the maximum eligible loan amount.
- **SHAP & LLM:** SHAP explanations and LLM (OpenAI) explanations are **only provided for fraud predictions**.
- **Environment Variables:** You must set `OPENAI_API_KEY` for LLM explanations and Azure Form Recognizer Endpoint for document parsing.
- **Logging:** Every API request is logged with a **unique request ID**, including all input values and file names, for traceability.
- **Error Handling:** All exceptions are logged and returned in the API response with the request ID for debugging.
- **Model Paths:** The default model and encoder paths are under the `models/` directory.
- **Thread Safety:** The loaded models are assumed to be thread-safe for inference.
- **Monitoring:** Drift detection and performance monitoring scripts require reference data and monitoring logs to function properly.

---

**If you have custom requirements or want to change any assumptions, update the relevant code and documentation accordingly.** 