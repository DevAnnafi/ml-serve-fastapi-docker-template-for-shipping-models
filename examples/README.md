# Examples

## Python client

```bash
# Start the server first
uvicorn app.main:app --reload

# In another terminal
python examples/predict.py --text "This product is amazing"
# {
#   "label": "POSITIVE",
#   "score": 0.9998,
#   "model": "distilbert-base-uncased-finetuned-sst-2-english"
# }
```

## curl

```bash
# Single prediction
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I really enjoyed this"}' | python -m json.tool

# Health check
curl -s http://localhost:8000/health | python -m json.tool

# Prometheus metrics
curl -s http://localhost:8000/metrics | head -20
```

## Docker

```bash
# Build and run
docker build -t ml-serve .
docker run -p 8000:8000 ml-serve

# With a custom model
docker run -p 8000:8000 -e MODEL_ID=siebert/sentiment-roberta-large-english ml-serve
```
