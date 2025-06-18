from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

# Load model at startup
try:
    model = joblib.load("fraud_detection_pipeline.pkl")
except Exception as e:
    raise RuntimeError(f"Error loading model: {str(e)}") from e

class TransactionRequest(BaseModel):
    transaction_type: str
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float

class PredictionResponse(BaseModel):
    prediction: int
    fraud_status: str
    message: str

@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: TransactionRequest):
    # Validate transaction type
    valid_types = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT"]
    if transaction.transaction_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid transaction type. Must be one of {valid_types}"
        )
    
    # Prepare input data
    input_data = pd.DataFrame([{
        "type": transaction.transaction_type,
        "amount": transaction.amount,
        "oldbalanceOrg": transaction.oldbalanceOrg,
        "newbalanceOrig": transaction.newbalanceOrig,
        "oldbalanceDest": transaction.oldbalanceDest,
        "newbalanceDest": transaction.newbalanceDest
    }])
    
    # Make prediction
    try:
        prediction = model.predict(input_data)[0]
        prediction = int(prediction)  # Convert to int for cleaner response
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )
    
    # Prepare response
    if prediction == 1:
        return {
            "prediction": prediction,
            "fraud_status": "fraud",
            "message": "This transaction is predicted to be fraudulent"
        }
    else:
        return {
            "prediction": prediction,
            "fraud_status": "legitimate",
            "message": "This transaction is predicted to be legitimate"
        }
