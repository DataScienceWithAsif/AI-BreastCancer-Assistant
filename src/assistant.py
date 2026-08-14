import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

import torch
from model_loader import ModelLoader

SYSTEM_PROMPT = (
    "You are a warm, empathetic breast health assistant. You explain breast cancer "
    "topics in plain, supportive language. You never diagnose or give personalized "
    "medical advice, and you always recommend the person consult their doctor for "
    "decisions specific to their situation."
)


class BreastCancerAssistant:
    def __init__(self):
        loader = ModelLoader()
        self.tokenizer, self.model = loader.load_model()
        self.model.eval()

    def ask(self, user_message: str, history=None, max_new_tokens: int = 250, temperature: float = 0.6) -> str:
        """
        user_message: the latest user message
        history: optional list of {"role": "user"/"assistant", "content": ...} prior turns
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs["input_ids"], outputs)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()


def explain_prediction(predictor_result: dict, assistant: "BreastCancerAssistant") -> str:
    """Feed a PredictiveModel result into the assistant for a plain-language explanation."""
    prompt = (
        f"A breast tissue screening model analyzed some measurements and predicted: "
        f"{predictor_result['result']}, with {predictor_result['confidence'] * 100:.1f}% confidence. "
        f"In plain, supportive language, explain what a result like this generally means and what "
        f"someone should consider doing next. Keep it concise, warm, and non-alarming."
    )
    return assistant.ask(prompt)


if __name__ == "__main__":
    assistant = BreastCancerAssistant()
    response = assistant.ask("Can you explain the difference between a benign breast cyst and a malignant tumor?")
    print("--- ASSISTANT RESPONSE ---")
    print(response)