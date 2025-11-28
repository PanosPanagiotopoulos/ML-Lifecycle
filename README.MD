# ML Lifecycle - Local Setup

## Step 1: Create Virtual Environment

```powershell
python -m venv .venv
```

## Step 2: Activate Virtual Environment

```powershell
.\.venv\Scripts\Activate
```

## Step 3: Install Dependencies

```powershell
pip install -r requirements.txt
```

## Step 4: Train Model

```powershell
python train.py configs/train.yaml
```

**Expected time:** 5-10 minutes on CPU

## Step 5: Start MLflow (Terminal 1)

```powershell
.\.venv\Scripts\Activate
mlflow ui --host 0.0.0.0 --port 5000
```

**Access:** http://localhost:5000

## Step 6: Start API (Terminal 2)

```powershell
.\.venv\Scripts\Activate
uvicorn src.controller.predict:app --host 0.0.0.0 --port 8000
```

## Requirements

- **Python:** 3.11+
- **RAM:** 8-16 GB
- **Disk:** 5-10 GB free
- **CPU:** 4+ cores (or GPU for faster training)

---