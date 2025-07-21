import pytest
from unittest.mock import patch, MagicMock
from src.predict.predict import Predictor
import pandas as pd

def test_predictor_predict_keys():
    # Patch joblib.load to return a mock model
    with patch('src.predict.predict.joblib.load') as mock_load:
        mock_model = MagicMock()
        mock_model.predict.return_value = [0]
        mock_load.return_value = mock_model
        predictor = Predictor()
        data = pd.DataFrame([{
            'age': 30,
            'gender': 'male',
            'marital_status': 'single',
            'residential_status': 'owned',
            'employment_stats': 'employed',
            'monthly_income': 50000,
            'other_income_source': 'none',
            'total_monthly_expenses': 20000,
            'number_of_loans': 1,
            'cibil_score': 750,
            'loan_amount_requested': 300000,
            'loan_purpose': 'home',
            'loan_term_in_months': 60
        }])
        result = predictor.predict(data)
        assert 'is_fraud' in result
        assert 'credit_worthiness' in result or 'is_creditworthy' in result 