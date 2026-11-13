import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import time

# 1. Initialize FastAPI
app = FastAPI(
    title="Order Cancellation Risk API",
    description="Predicts whether an e-commerce line item is at risk of cancellation.",
    version="1.0.0"
)

# 2. Load your saved sklearn Pipeline at startup
MODEL_PATH = "models/cancellation_pipe.joblib"
model = joblib.load(MODEL_PATH)

# 3. Define Prometheus Monitoring Metrics (for Grafana!)
PREDICTION_COUNT = Counter(
    "model_predictions_total",
    "Total predictions served",
    ["predicted_class"]
)
PREDICTION_LATENCY = Histogram(
    "model_prediction_latency_seconds",
    "Time spent running model inference"
)

# 4. Input Schema Validation (Matches your exact X columns!)
class OrderItemInput(BaseModel):
    AbsQuantity: int = Field(..., gt=0, example=12, description="Absolute item quantity (>0)")
    Price: float = Field(..., gt=0, example=6.95, description="Unit price (>0)")
    Hour: int = Field(..., ge=0, le=23, example=14, description="Order hour (0-23)")
    DayOfWeek: int = Field(..., ge=0, le=6, example=2, description="Day of week (0=Mon, 6=Sun)")
    Month: int = Field(..., ge=1, le=12, example=11, description="Month (1-12)")
    Country: str = Field(..., example="United Kingdom", description="Customer country")

# 5. Health Check Endpoint (Used by Docker & AWS to verify app is alive)
@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

# 6. Prediction Endpoint
@app.post("/predict")
def predict_cancellation(item: OrderItemInput):
    start_time = time.time()
    try:
        # Convert single JSON input into a 1-row DataFrame matching X
        input_df = pd.DataFrame([item.model_dump()])
        
        # Run inference through your saved sklearn Pipeline
        pred_class = int(model.predict(input_df)[0])
        pred_prob = float(model.predict_proba(input_df)[0][1])

        # Record metrics for Prometheus / Grafana
        PREDICTION_COUNT.labels(predicted_class=str(pred_class)).inc()
        PREDICTION_LATENCY.observe(time.time() - start_time)

        return {
            "is_cancelled_pred": pred_class,
            "cancellation_probability": round(pred_prob, 4),
            "risk_level": "HIGH" if pred_prob >= 0.5 else "LOW"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 7. Prometheus Metrics Endpoint
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/")
def home():
    return {"message": "Order Cancellation API is running. Visit /docs to test."}