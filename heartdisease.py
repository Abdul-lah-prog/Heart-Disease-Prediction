from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Load model and scaler
model = joblib.load(r"c:\Users\user\OneDrive\Desktop\heart disease detection\gb_model(heart).pkl")
scaler = joblib.load(r"c:\Users\user\OneDrive\Desktop\heart disease detection\scaler(heart).pkl")  # Optional, use only if you used a scaler

# Define feature input structure
class EquipmentData(BaseModel):
    feature1: int
    feature2: int
    feature3: int
    feature4: int
    feature5: int
    feature6: int
    feature7: int
    feature8: int
    feature9: int
    feature10: float
    feature11: int
    feature12: int
    feature13:int
    

# Initialize FastAPI app
app = FastAPI(title="heart disease prediction API")

@app.post("/predict")
def predict(data: EquipmentData):
    # Convert input to NumPy array
    input_data = np.array([[data.feature1, data.feature2, data.feature3, data.feature4, data.feature5, data.feature6,data.feature7, data.feature8, data.feature9, data.feature10, data.feature11, data.feature12,data.feature13]])
    
    # Scale the input (if scaler was used during training)
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    prediction_label = "presence" if prediction == 1 else "absence"

    return {
        "prediction": int(prediction),
        "label": prediction_label
    }
