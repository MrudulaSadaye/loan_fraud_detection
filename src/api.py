import os
import csv
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
from src.predict.predict import Predictor
from src.utils.logger import get_logger
from src.data.data_processing import process_data
import pandas as pd
import traceback
from uuid import uuid4

logger = get_logger()

app = FastAPI()
predictor = Predictor()

MONITORING_LOG = 'monitoring/predictions_log.csv'

def log_monitoring(request_id, input_data, prediction, error=None):  
    """
    Logs the prediction results to the monitoring log csv file.
    Stores the request_id, timestamp, input_data, prediction, and error (if any) in the monitoring log file.
    """

    os.makedirs(os.path.dirname(MONITORING_LOG), exist_ok=True)
    with open(MONITORING_LOG, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            request_id,
            pd.Timestamp.now(),
            input_data,
            prediction,
            error if error else ''
        ])

class PredictionRequest(BaseModel):
    age: int
    gender: str
    marital_status: str
    residential_status: str
    employment_stats: str
    monthly_income: float
    other_income_source: str
    total_monthly_expenses: float
    number_of_loans: int
    cibil_score: float
    loan_amount_requested: float
    loan_purpose: str
    loan_term_in_months: int

class PredictionResponse(BaseModel):
    is_fraud: Optional[bool]
    credit_worthiness: Optional[bool]
    shap_explanation: Optional[Dict[str, Any]]
    llm_explanation: Optional[str]
    error: Optional[str]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    aadhar_card: UploadFile = File(...),
    pan_card: UploadFile = File(...),
    bank_statement: UploadFile = File(...),
    salary_slip: UploadFile = File(...)
):
    """
    Predicts loan application fraud and creditworthiness.

    This endpoint accepts applicant data and four required document uploads (Aadhaar card, PAN card, bank statement, salary slip).
    It processes the data, runs the fraud and creditworthiness models, and returns the prediction results.

    Input:
        request (PredictionRequest): Applicant data as JSON body.
        aadhar_card (UploadFile): Aadhaar card PDF upload.
        pan_card (UploadFile): PAN card PDF upload.
        bank_statement (UploadFile): Bank statement PDF upload.
        salary_slip (UploadFile): Salary slip PDF upload.

    Output:
        PredictionResponse: Contains is_fraud, credit_worthiness, SHAP explanation, LLM explanation, or error message.
    """
    request_id = str(uuid4())
    try:
        data = request.dict()
        file_info = {
            'aadhar_card': aadhar_card.filename,
            'pan_card': pan_card.filename,
            'bank_statement': bank_statement.filename,
            'salary_slip': salary_slip.filename
        }
        logger.info(f"[RequestID: {request_id}] Received prediction request: input={data}, files={file_info}")

        # Read the files
        aadhar_card_data = await aadhar_card.read()
        pan_card_data = await pan_card.read()
        bank_statement_data = await bank_statement.read()
        salary_slip_data = await salary_slip.read()
        
        # Process the data
        df = pd.DataFrame([data])
        processed_data = process_data(df, aadhar_card_data, pan_card_data, bank_statement_data, salary_slip_data, is_train=False)
        logger.info(f"[RequestID: {request_id}] Processed data columns: {list(processed_data.columns)}")
        result = predictor.predict(processed_data)
        logger.info(f"[RequestID: {request_id}] Prediction result: {result}")
        log_monitoring(request_id, data, result)
        return PredictionResponse(**result)
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"[RequestID: {request_id}] Error in /predict endpoint: {e}\n{tb}")
        log_monitoring(request_id, data if 'data' in locals() else None, None, error=str(e))
        return PredictionResponse(error=f"Prediction failed: {str(e)} | RequestID: {request_id}") 