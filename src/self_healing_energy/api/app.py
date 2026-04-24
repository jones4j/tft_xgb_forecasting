from fastapi import FastAPI

from self_healing_energy.orchestration.batch_pipeline import BatchForecastPipeline
from self_healing_energy.serving.contracts import BatchForecastRequest, BatchForecastResponse


app = FastAPI(title="Self-Healing Energy Forecasting")
pipeline = BatchForecastPipeline.build_default()


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/forecast", response_model=BatchForecastResponse)
def forecast(request: BatchForecastRequest) -> BatchForecastResponse:
    return pipeline.run(request)

