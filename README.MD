# ML Lifecycle - PDF LLM Fine-tuning

Fine-tune large language models on PDF documents for question answering.

## Quick Start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Add PDFs
Place PDF files in `data/raw/`

### 3. Train
```bash
python train.py configs/train.yaml
```

### 4. Serve API
```bash
uvicorn src.controller.predict:app --reload
```

### 5. Test
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main topics?", "max_length": 200}'
```

Or visit: http://localhost:8000/docs

## Configuration

Edit `configs/train.yaml` to change model, training parameters, etc.

## Troubleshooting

**Out of memory:** Reduce `per_device_train_batch_size` in `configs/train.yaml`

**Slow training:** Use smaller model: `model_name: "gpt2"`
