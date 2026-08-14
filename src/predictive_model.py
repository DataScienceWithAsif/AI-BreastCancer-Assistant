import os
import numpy as np
import joblib
from tensorflow import keras

# Feature order MUST match the order used in featured_df during training
FEATURE_NAMES = [
    "worst concave points", "worst area", "mean perimeter", "mean texture",
    "area error", "texture error", "perimeter error", "worst concavity",
    "worst texture", "worst perimeter", "worst compactness", "radius error",
    "mean fractal dimension", "worst smoothness", "mean concave points",
]


class PredictiveModel:
    def __init__(self):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        scaler_path = os.path.join(base_dir, "BC_scaler.pkl")
        model_path = os.path.join(base_dir, "BC_predictor_model.keras")

        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found at: {scaler_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")

        self.scaler = joblib.load(scaler_path)
        self.model = keras.models.load_model(model_path)

    def predict_label(self, input_data) -> dict:
        """
        input_data: tuple/list of 15 values in FEATURE_NAMES order,
                    OR a dict mapping feature name -> value (safer, order-independent).
        """
        if isinstance(input_data, dict):
            missing = [f for f in FEATURE_NAMES if f not in input_data]
            if missing:
                raise ValueError(f"Missing required features: {missing}")
            ordered_values = [input_data[f] for f in FEATURE_NAMES]
        else:
            if len(input_data) != len(FEATURE_NAMES):
                raise ValueError(
                    f"Expected {len(FEATURE_NAMES)} features, got {len(input_data)}"
                )
            ordered_values = list(input_data)

        input_array = np.asarray(ordered_values, dtype=float).reshape(1, -1)
        std_input = self.scaler.transform(input_array)

        prediction = self.model.predict(std_input, verbose=0)
        label = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        result = "Malignant" if label == 0 else "Benign"

        return {
            "label": label,
            "result": result,
            "confidence": round(confidence, 4),
            "raw_probs": prediction[0].tolist(),
        }


if __name__ == "__main__":
    predictor = PredictiveModel()

    data = (0.2654, 2019.0, 122.80, 10.38, 153.40, 0.9053, 8.589, 0.7119,
            17.33, 184.60, 0.66560, 1.0950, 0.07871, 0.16220, 0.14710)

    result = predictor.predict_label(data)
    print(result)