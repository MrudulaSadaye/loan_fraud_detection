import pandas as pd
from evidently.report import Report
from evidently.metrics import DataDriftPreset, TargetDriftPreset
import os



REFERENCE_PATH = 'data/reference_data.csv'  # Your training data sample
CURRENT_PATH = 'monitoring/predictions_log.csv'  # Recent production data
REPORT_PATH = 'monitoring/data_drift_report.html'



reference = pd.read_csv(REFERENCE_PATH)
current = pd.read_csv(CURRENT_PATH)

features = [
    'age', 'gender', 'marital_status', 'residential_status', 'employment_status', 'monthly_income',
    'other_income', 'total_monthly_expenses', 'income_to_expense_ratio', 'number_of_loans',
    'cibil_score', 'loan_amount_requested', 'loan_purpose', 'loan_term_in_months', 'document_verification_score'
]
reference = reference[features]
current = current[features]



# Create and run the drift report
report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
report.run(reference_data=reference, current_data=current)
report.save_html(REPORT_PATH)
print(f"Drift report saved to {REPORT_PATH}") 