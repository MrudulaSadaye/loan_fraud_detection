import pytest
from fastapi.testclient import TestClient
from src.api import app
import io

client = TestClient(app)

def test_api_predict():
    data = {
        "age": 30,
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
    files = {
        "aadhar_card": ("aadhar.pdf", io.BytesIO(b"dummy"), "application/pdf"),
        "pan_card": ("pan.pdf", io.BytesIO(b"dummy"), "application/pdf"),
        "bank_statement": ("bank.pdf", io.BytesIO(b"dummy"), "application/pdf"),
        "salary_slip": ("salary.pdf", io.BytesIO(b"dummy"), "application/pdf"),
    }
    response = client.post("/predict", data={"request": data}, files=files)
    assert response.status_code == 200
    resp_json = response.json()
    assert "is_fraud" in resp_json
    assert "credit_worthiness" in resp_json