"""
unit tests for app.py using pytest and FastAPI TestClient.
These tests cover the health endpoint, text submission, and large text rejection.

to run:
pip install pytest
pytest -v

"""

from fastapi.testclient import TestClient
from app import app  # import your FastAPI app

client = TestClient(app)

def test_health_endpoint():
    """Test that the health endpoint returns 'OK'."""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.text.strip('"') == "OK"

def test_submit_text_endpoint():
    """Test text submission endpoint returns an AnalysisResult with mandatory fields."""
    payload = {"text": "This is a test sentence", "isCodeMode": False}
    response = client.post("/api/submit-text", json=payload)
    assert response.status_code == 200
    json_data = response.json()
    # Check required keys in the AnalysisResult
    assert "id" in json_data
    assert "verdict" in json_data
    assert "modelVersion" in json_data
    assert "createdAt" in json_data

def test_reject_too_long_text():
    """Test that submitting text over MAX_TOKENS returns HTTP 400."""
    # Generate a string with way more than 5000 tokens
    long_text = "word " * 6000
    payload = {"text": long_text, "isCodeMode": False}
    response = client.post("/api/submit-text", json=payload)
    assert response.status_code == 400
    assert "Maximum words" in response.text
