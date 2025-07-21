import joblib
import pandas as pd
from src.data.data_processing import process_data    
from src.utils.logger import get_logger
from src.predict.explain import explain_with_shap, llm_explanation
import os

logger = get_logger()

FRAUD_MODEL_PATH = 'models/fraud_model_ensemble.pkl'

class Predictor:
    """
    Predictor class for fraud detection and creditworthiness prediction.
    Input:
        fraud_model_path: Path to the fraud detection model
        credit_model_path: Path to the creditworthiness prediction model
    """

    def __init__(self, fraud_model_path=FRAUD_MODEL_PATH, credit_model_path=CREDIT_MODEL_PATH):
        """
        Load the Fraud and Credits models from the models folder
        """
        logger.info('Loading models...')
        self.fraud_model = joblib.load(fraud_model_path)


    def rule_based_credit_worthiness(self, applicant_data, interest_rate=0.105):
        """
        Calculate the maximum eligible loan amount for an applicant using a rule-based approach.

        This method uses the applicant's monthly income, existing EMI, loan tenure, and requested loan amount
        to determine the maximum loan amount the applicant is eligible for, based on a fixed EMI-to-income ratio
        and a given interest rate. If the eligible EMI is less than or equal to zero, the applicant is not creditworthy.

        Input:
            applicant (pd.DataFrame): Applicant data containing at least 'monthly_income', 'existing_emi',
                                           'loan_term_in_months', and 'loan_amount_requested'.
            interest_rate (float): Annual interest rate for the loan (default: 0.105).

        Output:
            credit_worthiness: The maximum eligible loan amount, capped at the requested loan amount. Returns 0 if not eligible.
        """
        emi_ratio = 0.4
        monthly_income = applicant_data['monthly_income']
        existing_emi = applicant_data['existing_emi']
        tenure_years = applicant_data['loan_term_in_months']
        loan_requested = applicant_data['loan_amount_requested']

        eligible_emi = (monthly_income * emi_ratio) - existing_emi
        if eligible_emi <= 0:
            return 0

        r = interest_rate / 12
        n = tenure_years * 12
        max_loan = (eligible_emi * ((1 + r)**n - 1)) / (r * (1 + r)**n)
        credit_worthiness = min(max_loan, loan_requested)
        return credit_worthiness

    def predict(self, applicant_data):
        """
        Predict the fraud and creditworthiness of the input data.
        Input:
            input_dict: Dictionary containing the input data
        Output:
            result: Dictionary containing the prediction results
            is_fraud: Boolean indicating if the application is fraud
            is_creditworthy: Boolean indicating if the application is creditworthy
            shap_explanation: SHAP explanation of the prediction
            llm_explanation: Natural language explanation of the prediction
        """

        is_fraud = int(self.fraud_model.predict(applicant_data)[0])
        result = {'is_fraud': is_fraud}

        if is_fraud:
            logger.info('Application flagged as fraud. Generating SHAP explanation...')
            shap_explanation = explain_with_shap(self.fraud_model, applicant_data, feature_names=applicant_data.columns)
            result['shap_explanation'] = shap_explanation

            llm_explanation = llm_explanation(applicant_data, shap_explanation)
            result['llm_explanation'] = llm_explanation

        else:
            logger.info('Application not fraud. Predicting credit worthiness...')
            credit_worthiness = self.rule_based_credit_worthiness(applicant_data)
            result['credit_worthiness'] = credit_worthiness

        return result 