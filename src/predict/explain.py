import shap
import numpy as np
from src.utils.logger import get_logger
import os

logger = get_logger()

def explain_with_shap(model, X, feature_names=None)->dict:
    """
    Generate SHAP explanation for the model.
    Input:
        model: The model to be explained
        X: The input dataframe
        feature_names: The names of the features
    Output:
        explanation: Dictionary containing the SHAP explanation
    """

    logger.info('Generating SHAP explanation...')
    
    #TODO: Define Shaps Explainer for the fraud model
    #TODO: Call the explainer to generate the SHAP values for the input data
    #TODO: Return the SHAP values as a dictionary


    return explanation

# --- Natural Language Explanation ---
def llm_explanation(user_input: dict, shap_explanation: dict)->str:
    """
    Calls an LLM to generate a natural language explanation for why the application is considered fraud.
    """
    import openai
    
    prompt = (
        "Given the following loan application data and SHAP feature importances, "
        "Explain  why this application is considered as fraud and provide the details of which features are contributing to the fraud."
        f"Application data: {user_input}\n"
        f"SHAP feature importances: {shap_explanation}\n"
        "Explanation:"
    )
    logger.info('Calling LLM for Fraud explanation...')
    
    #TODO: Call OpenAI Completion API with gpt-4o-mini model to generate an explanation for the fraud
    
    return explanation 