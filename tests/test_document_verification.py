import pytest
from src.data.document_verification import compute_document_mismatch

def test_document_mismatch_perfect():
    # All fields match
    docs = [
        {'name': 'John Doe', 'date_of_birth': '1990-01-01', 'employer': 'ABC Corp', 'income': '50000'},
        {'name': 'John Doe', 'date_of_birth': '1990-01-01', 'employer': 'ABC Corp', 'income': '50000'},
        {'name': 'John Doe', 'date_of_birth': '1990-01-01', 'employer': 'ABC Corp', 'income': '50000'},
        {'name': 'John Doe', 'date_of_birth': '1990-01-01', 'employer': 'ABC Corp', 'income': '50000'},
    ]
    score = compute_document_mismatch.__wrapped__(*docs) if hasattr(compute_document_mismatch, '__wrapped__') else compute_document_mismatch(*docs)
    assert score == pytest.approx(1.0)

def test_document_mismatch_mismatch():
    # All fields mismatch
    docs = [
        {'name': 'John Doe', 'date_of_birth': '1990-01-01', 'employer': 'ABC Corp', 'income': '50000'},
        {'name': 'Jane Smith', 'date_of_birth': '1985-05-05', 'employer': 'XYZ Inc', 'income': '10000'},
        {'name': 'Foo Bar', 'date_of_birth': '1970-12-12', 'employer': 'QRS Ltd', 'income': '20000'},
        {'name': 'Baz Qux', 'date_of_birth': '2000-07-07', 'employer': 'LMN LLC', 'income': '30000'},
    ]
    score = compute_document_mismatch.__wrapped__(*docs) if hasattr(compute_document_mismatch, '__wrapped__') else compute_document_mismatch(*docs)
    assert score < 0.5 