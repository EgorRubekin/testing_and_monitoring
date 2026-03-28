import asyncio
import pandas as pd
from typing import Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from prometheus_client import make_asgi_app

from ml_service import config
from ml_service.features import to_dataframe
from ml_service.mlflow_utils import configure_mlflow
from ml_service.model import Model
from ml_service.schemas import (
    PredictRequest, PredictResponse,
    UpdateModelRequest, UpdateModelResponse,
)
from ml_service.metrics import (
    REQUEST_COUNT, REQUEST_LATENCY, MODEL_INFERENCE_TIME, 
    MODEL_PROBABILITY, MODEL_PREDICTIONS, PREPROCESSING_TIME, MODEL_INFO
)

from evidently.report import Report
from evidently.presets import DataDriftPreset
from evidently.ui.workspace import RemoteWorkspace

MODEL = Model()
current_data_buffer = []
EVIDENTLY_URL = 'http://158.160.2.37:8000/'
PROJECT_ID = '019d3623-92ea-73db-8932-a48c52d702b2'

async def evidently_monitoring_task():
    """Фоновая задача для отправки отчетов в Evidently"""
    while True:
        await asyncio.sleep(600) 
        if len(current_data_buffer) >= 10:
            try:
                current_df = pd.DataFrame(current_data_buffer)
                reference_df = current_df.copy() 
                
                drift_report = Report(metrics=[DataDriftPreset()])
                drift_report.run(reference_data=reference_df, current_data=current_df)
                
                workspace = RemoteWorkspace(EVIDENTLY_URL)
                workspace.add_run(PROJECT_ID, drift_report)
                current_data_buffer.clear()
            except Exception as e:
                print(f"Evidently error: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_mlflow()
    try:
        run_id = config.default_run_id()
        MODEL.set(run_id=run_id)
        MODEL_INFO.labels(run_id=run_id, model_type="sklearn", features_count=len(MODEL.features)).set(1)
    except Exception as e:
        print(f"Startup error: {e}")
    
    monitor_task = asyncio.create_task(evidently_monitoring_task())
    yield
    monitor_task.cancel()

def create_app() -> FastAPI:
    app = FastAPI(title='MLflow FastAPI service', version='1.0.0', lifespan=lifespan)
    
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    @app.get('/health')
    def health() -> dict[str, Any]:
        return {'status': 'ok', 'run_id': MODEL.get().run_id}

    @app.post('/predict', response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        with REQUEST_LATENCY.labels(endpoint="/predict").time():
            model_data = MODEL.get()
            if model_data.model is None:
                REQUEST_COUNT.labels(method="POST", endpoint="/predict", http_status=503).inc()
                raise HTTPException(status_code=503, detail='Model not loaded')

            try:
                with PREPROCESSING_TIME.time():
                    df = to_dataframe(request, needed_columns=MODEL.features)
                
                with MODEL_INFERENCE_TIME.time():
                    probability = float(model_data.model.predict_proba(df)[0][1])
                    prediction = int(probability >= 0.5)

                current_data_buffer.append(request.dict(by_alias=True))

                MODEL_PROBABILITY.observe(probability)
                MODEL_PREDICTIONS.labels(prediction_class=str(prediction)).inc()
                REQUEST_COUNT.labels(method="POST", endpoint="/predict", http_status=200).inc()

                return PredictResponse(prediction=prediction, probability=probability)

            except ValueError as e:
                REQUEST_COUNT.labels(method="POST", endpoint="/predict", http_status=400).inc()
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                REQUEST_COUNT.labels(method="POST", endpoint="/predict", http_status=500).inc()
                raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    @app.post('/updateModel', response_model=UpdateModelResponse)
    def update_model(req: UpdateModelRequest) -> UpdateModelResponse:
        try:
            MODEL.set(run_id=req.run_id)
            MODEL_INFO.clear()
            MODEL_INFO.labels(run_id=req.run_id, model_type="sklearn", features_count=len(MODEL.features)).set(1)
            return UpdateModelResponse(run_id=req.run_id)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Invalid run_id: {e}")

    return app

app = create_app()