# Fishease Model API
## *\*\***NOT COMPLETED**\*\*\*

#### /health (GET)
checks api health

#### /classes (GET)
List all Disease Classes

#### /predict (POST)
Disease prediction handler

## Current Issues
- Inaccurate predictions
- 100% indicated accuracy on all predictions
- Predictions not as expected as initial prediction on model creation notebook (fishease.ipynb)

## How to use
1. Install dependencies
```
pip install --no-cache-dir -r requirements.txt
```
2. Run the application
```
uvicorn main:app --host 0.0.0.0 --port 3500
```