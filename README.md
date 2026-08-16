# 🎗️ AI Breast Cancer Assistant

A modern Streamlit application that combines:
- **Breast cancer prediction** from diagnostic feature inputs
- **AI-powered educational chat** for breast health questions

> ⚠️ This project is for educational support only and is **not** a medical diagnostic system.

---

## ✨ UI Overview

The app is designed with a clean, modern interface:
- Dark gradient theme with glass-style cards
- Two focused workflows in the sidebar:
  - **🔬 Cancer Prediction**
  - **💬 AI Assistant**
- Color-coded prediction result cards (Benign/Malignant)
- Confidence progress bar and safety disclaimer
- Styled conversational chat with persistent session history

---

## 🧠 Core Features

### 1) Breast Tissue Prediction
- Accepts **15 diagnostic measurements** via sliders
- Uses a trained Keras model (`BC_predictor_model.keras`)
- Applies preprocessing using saved scaler (`BC_scaler.pkl`)
- Returns:
  - Predicted class (`Benign` / `Malignant`)
  - Confidence score
  - Raw probabilities

### 2) AI Breast Health Assistant
- Built with Transformers + PyTorch
- Uses a supportive system prompt focused on plain-language education
- Helps explain:
  - Screening concepts
  - General treatment topics
  - Prediction outcomes (via one-click explanation from the prediction page)

---

## 🗂️ Project Structure

```text
AI-BreastCancer-Assistant/
├── BC_predictor_model.keras
├── BC_scaler.pkl
├── requirements.txt
├── README.md
└── src/
    ├── app.py               # Streamlit UI
    ├── predictive_model.py  # Prediction model wrapper
    ├── model_loader.py      # LLM loading utilities
    └── assistant.py         # AI assistant logic
```

---

## ⚙️ Requirements

- Python environment with required packages
- Model artifacts present at repository root:
  - `BC_predictor_model.keras`
  - `BC_scaler.pkl`

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

From the project root:

```bash
streamlit run src/app.py
```

Then open the local URL shown in terminal (usually `http://localhost:8501`).

---

## 🧪 How to Use the App

### Cancer Prediction Flow
1. Open **🔬 Cancer Prediction**
2. Adjust the 15 feature sliders using report values
3. Click **Run Prediction**
4. Review class + confidence
5. Optionally click **Explain this result with AI Assistant**

### AI Assistant Flow
1. Open **💬 AI Assistant**
2. Ask a breast health education question
3. Read the model response in chat format
4. Use **Clear conversation** to reset context

---

## 🔒 Safety Note

This assistant does **not** diagnose disease or provide personalized medical advice.
Always consult a licensed healthcare professional for medical decisions.

---

## 📄 License

This project is released under the terms of the repository's [LICENSE](LICENSE).

## Streamlit Chatbot
<img width="958" height="440" alt="dev" src="https://github.com/user-attachments/assets/f04581b2-ac20-4170-b513-367816e892fd" />

<img width="933" height="445" alt="dev1" src="https://github.com/user-attachments/assets/9c534f99-2223-47c9-a137-58bd5c4dad63" />

<img width="941" height="391" alt="dev2" src="https://github.com/user-attachments/assets/a1fe7ed0-2665-4509-8510-3f1016a1a44d" />

<img width="902" height="433" alt="dev4" src="https://github.com/user-attachments/assets/de6c061f-c20d-4f70-b3ce-334b40d939ae" />

