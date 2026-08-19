# ml-serve 🚀

A FastAPI + Docker template for shipping HuggingFace models to production. Under the hood it wraps a `distilbert` sentiment pipeline and exposes it over HTTP — ready to containerize and deploy.

## What's here

```
app/
  main.py       # FastAPI app — /predict, /health, /metrics endpoints
  predictor.py  # SentimentPredictor: loads the HF pipeline on startup
  schemas.py    # Pydantic request/response models
Dockerfile      # multi-stage build (builder → runtime, non-root user)
requirements.txt
```

The `/predict` endpoint accepts a JSON body `{"text": "..."}` and returns a sentiment label + confidence score. `/health` surfaces the loaded model version. `/metrics` exposes Prometheus counters and a request latency histogram.

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Visit `http://localhost:8000/docs` for the auto-generated Swagger UI.

## What's coming

- pytest suite covering the predictor, API contract, and container smoke test
- GitHub Actions CI (test + docker build on every push)
- Deployed URL with example curl in this README
