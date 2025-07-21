import joblib
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.utils.logger import get_logger
from src.data.document_verification import Document_Verification

logger = get_logger()

CATEGORICAL_COLS = [
    'gender', 'marital_status', 'residential_status', 'employment_status', 'loan_purpose'
]

ENCODER_PATH = 'models/onehot_encoder.pkl'

def process_data(df: pd.DataFrame, aadhar_card: bytes = None, pan_card: bytes = None, bank_statement: bytes = None, salary_slip: bytes = None, is_train: bool = False):
    """
    Process the training data for fraud detection.
    Converting the categorical columns to numerical columns using OneHotEncoder
    Calculating the income_to_expense_ratio
    Handling the missing values
    Input:
        df: pandas DataFrame containing the training data
    Output:
        df: pandas DataFrame containing the processed training data
    """
    try:
        logger.info('Starting data processing...')
        df = df.copy()

        # Derived: income_to_expense_ratio
        try:
            df['income_to_expense_ratio'] = (df['monthly_income'] + df.get('other_income', 0)) / (df['total_monthly_expenses'] + 1e-5)
        except Exception as e:
            logger.error(f'Error calculating income_to_expense_ratio: {e}')
            df['income_to_expense_ratio'] = -1

        try:
            df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].astype(str)
        except Exception as e:
            logger.error(f'Error converting categorical columns to string: {e}')
            return df

        if is_train:
            try:
                #TODO: Fit the OneHotEncoder on the training data and save the encoder to the models folder
            except Exception as e:
                logger.error(f'Error fitting/saving OneHotEncoder: {e}')
                return df
        else:
            try:
                #TODO: Load the OneHotEncoder from the models folder
            except Exception as e:
                logger.error(f'Error loading OneHotEncoder: {e}')
                return df

        # Transform categorical columns
        try:
            #TODO: Apply the OneHotEncoder to the categorical columns
            
        except Exception as e:
            logger.error(f'Error applying OneHotEncoder transformation: {e}')
            return df

        # Fill missing values
        try:
            logger.info('Filling missing values')
            df = df.fillna(-1)
        except Exception as e:
            logger.error(f'Error filling missing values: {e}')

        # If is_train is False, add document verification score
        if is_train == False:
            document_verification = Document_Verification(aadhar_card, pan_card, bank_statement, salary_slip)
            df['document_verification_score'] = document_verification.compute_document_mismatch()

        logger.info('Data processing complete.')
        return df
        
    except Exception as e:
        logger.critical(f'Critical error in process_data: {e}')
        return df 