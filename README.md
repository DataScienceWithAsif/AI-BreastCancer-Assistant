# AI Breast Health Companion — Setup

## Files added
- `src/app.py` — main Streamlit app (run this)
- `src/assistant.py` — chatbot wrapper class (uses your existing `model_loader.py`)

## Files you already have (must be present)
- `src/predictive_model.py`
- `src/model_loader.py`
- `BC_predictor_model.keras` and `BC_scaler.pkl` at the project root (one level above `src/`)

## Install dependencies

```bash
pip install -r requirements.txt
```

(If you already have `tensorflow`/`torch`/`transformers` installed in your `devsphere_capstone_env`, you can skip re-installing those.)

## Run the app

From the project root:

```bash
cd src
streamlit run app.py
```

This opens the app in your browser (usually `http://localhost:8501`).

## Notes

- The prediction model and the LLM assistant are both loaded lazily via `st.cache_resource`,
  so they only load once per session, not on every interaction. First load of the assistant
  may take a little while since it's loading the full LLM weights.
- Sliders on the Prediction page use the min/max ranges you provided, with the midpoint as
  the default value.
- The "Explain this result with AI Assistant" button on the Prediction page feeds the model's
  prediction into the chatbot for a plain-language explanation — this is the connector between
  your two capstone components.
- If the assistant is slow to respond, that's expected for CPU inference on a 3B model; consider
  reducing `max_new_tokens` in `assistant.py` if you need faster responses for a live demo.
