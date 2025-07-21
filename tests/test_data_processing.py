import pytest
import pandas as pd
from src.data.data_processing import process_data

def test_process_data_basic():
    data = pd.DataFrame([
        {
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
        }
    ])
    processed = process_data(data, is_train=True)
    # Check derived column
    assert 'income_to_expense_ratio' in processed.columns
    # Check that one-hot columns exist
    assert any('gender_' in col for col in processed.columns)
    assert any('marital_status_' in col for col in processed.columns)
    # Check missing values filled
    assert not processed.isnull().any().any() 