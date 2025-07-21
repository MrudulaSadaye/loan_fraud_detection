import logging
from rapidfuzz import fuzz
from typing import Dict, Any, List
from src.utils.form_recognizer import AzureFormRecognizerClient
# Set up logger
logger = logging.getLogger("document_verification")
logger.setLevel(logging.INFO)


#Assigning weights to the fields based on their importance
FIELD_WEIGHTS = {
    'name': 0.3,
    'date_of_birth': 0.2,
    'employer': 0.2,
    'income': 0.3
}

class Document_Verification:

    def __init__(self, aadhar_card:bytes, pan_card:bytes, bank_statement:bytes, salary_slip:bytes):
        self.aadhar_card = aadhar_card
        self.pan_card = pan_card
        self.bank_statement = bank_statement
        self.salary_slip = salary_slip


    # --- Fuzzy Matching Logic ---
    def match_score(self, val1, val2, field):
        if val1 is None or val2 is None:
            return 0
        if field == 'income':
            try:
                v1, v2 = float(val1), float(val2)
                if abs(v1 - v2) < 1e-2:
                    return 1
                elif abs(v1 - v2) / max(abs(v1), abs(v2), 1) < 0.1:
                    return 0.9
                elif abs(v1 - v2) / max(abs(v1), abs(v2), 1) < 0.2:
                    return 0.6
                else:
                    return 0
            except Exception:
                return 0
        else:
            score = fuzz.ratio(str(val1).lower(), str(val2).lower())
            if score == 100:
                return 1
            elif score >= 90:
                return 0.9
            elif score >= 70:
                return 0.6
            else:
                return 0

    # --- Weighted Mismatch Score ---
    def compute_document_mismatch(self) -> float:
        """
        Compares fields across all documents, computes weighted mismatch score.
        parsed_docs: List of dicts, one per document, with extracted fields.
        Returns: document_mismatch score (0 to 1, where 1 is perfect match)
        """
        try:
            #Create AzureFormRecognizerClient object
            form_recognizer = AzureFormRecognizerClient()

            #Call the get_document_fields function to get the fields from the documents
            aadhar_fields = form_recognizer.get_document_fields(self.aadhar_card, ['name', 'address', 'date_of_birth'])
            pan_fields = form_recognizer.get_document_fields(self.pan_card, ['name', 'date_of_birth'])
            salary_slip_fields = form_recognizer.get_document_fields(self.salary_slip, ['name', 'employer', 'income'])
            bank_statement_fields = form_recognizer.get_document_fields(self.bank_statement, ['name', 'income'])


            #TODO: Compare each field across all documents using "match_score" function

            #TODO: Calculate average match score for each field across all documents
                #average match score = sum of the match scores for the field / number of documents

            #TODO: Calculate the weighted sum of the match scores
                #weighted score = average match score * weight of the field

            #TODO: Calculate the total weight of the fields
                #total weight = sum of the weights of all fields

            #TODO: Calculate the document mismatch score  
                #document mismatch score = weighted sum of the match scores / total weight

            return document_mismatch_score
        except Exception as e:
            return 0

