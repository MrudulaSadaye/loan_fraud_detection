import os
import logging
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import DefaultAzureCredential
logger = logging.getLogger("form_recognizer")
logger.setLevel(logging.INFO)

class AzureFormRecognizerClient:
    def __init__(self):
        self.endpoint = os.getenv("AZURE_FORM_RECOGNIZER_ENDPOINT")
    
        #Initialize the DocumentAnalysisClient
        self.client = DocumentAnalysisClient(
            endpoint=self.endpoint,
            credential=DefaultAzureCredential()     #use Managed Identity for authentication
        )
        logger.info("Azure Form Recognizer client initialized.")


    def get_document_fields(self, document: bytes, fields: List[str]) -> dict:     #document is the file to be analyzed
        """
        Analyzes a document using Azure Form Recognizer and returns extracted fields as a dict.
        Input:
            document: File to be analyzed
            fields: List of fields to be extracted
        Output: 
            Dict of extracted fields
        """

        #Read the document using form_recognizer's begin_analyze_document and model_id = "prebuilt-idDocument"
        #Extract the fields that are provided in the fields parameter
        #Return the extracted fields in a dictionary

        

        