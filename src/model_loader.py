import os
import shutil

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

load_dotenv()


class ModelLoader:
    def __init__(self):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.model_id = "A-Asif/llama3.2-breastcancer-assistant"
        self.local_dir = os.path.join(base_dir, "local_models", "llama3.2-breastcancer-assistant")
        self.cache_dir = os.path.join(base_dir, ".hf_cache")

    def _local_model_ready(self):
        required_files = ["config.json", "tokenizer_config.json"]
        if not os.path.isdir(self.local_dir):
            return False
        return all(os.path.exists(os.path.join(self.local_dir, file_name)) for file_name in required_files)

    def _remove_stale_local_dir(self):
        if os.path.isdir(self.local_dir):
            print(f"🧹 Removing stale/incomplete local model directory: {self.local_dir}")
            shutil.rmtree(self.local_dir)

    def _load_from(self, path_to_load, local_only):
        """
        Tries loading with `dtype=`, falling back to the older `torch_dtype=`
        kwarg if the installed transformers version doesn't support `dtype=`.
        """
        common_kwargs = dict(
            cache_dir=self.cache_dir,
            local_files_only=local_only,
        )

        tokenizer = AutoTokenizer.from_pretrained(path_to_load, **common_kwargs)

        device_kwargs = dict(
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        try:
            model = AutoModelForCausalLM.from_pretrained(
                path_to_load,
                dtype=torch.bfloat16,
                **common_kwargs,
                **device_kwargs,
            )
        except TypeError:
            # Older transformers versions use `torch_dtype` instead of `dtype`
            model = AutoModelForCausalLM.from_pretrained(
                path_to_load,
                torch_dtype=torch.bfloat16,
                **common_kwargs,
                **device_kwargs,
            )

        return tokenizer, model

    def load_model(self):
        if self._local_model_ready():
            print(f"🔄 Loading model locally from disk: {self.local_dir}")
            path_to_load = self.local_dir
            local_only = True
        else:
            self._remove_stale_local_dir()
            print(f"🌐 Local model not found or incomplete. Downloading from Hugging Face Hub: {self.model_id}")
            path_to_load = self.model_id
            local_only = False

        try:
            tokenizer, model = self._load_from(path_to_load, local_only)
        except Exception as exc:
            if path_to_load != self.model_id:
                raise

            print(f"⚠️ Download failed with a Hub transfer issue: {exc}")
            print("🔁 Retrying with the Xet transfer pipeline disabled and the stale cache cleared.")
            os.environ["HF_HUB_DISABLE_XET"] = "1"
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

            if os.path.isdir(self.cache_dir):
                for entry in os.listdir(self.cache_dir):
                    full_path = os.path.join(self.cache_dir, entry)
                    if os.path.isdir(full_path):
                        shutil.rmtree(full_path)

            self._remove_stale_local_dir()
            tokenizer, model = self._load_from(path_to_load, local_only=False)

        if path_to_load == self.model_id:
            print(f"💾 Saving model and tokenizer locally to: {self.local_dir}")
            os.makedirs(self.local_dir, exist_ok=True)
            tokenizer.save_pretrained(self.local_dir)
            model.save_pretrained(self.local_dir)

        return tokenizer, model


if __name__ == "__main__":
    print(f"🖥️  CUDA available: {torch.cuda.is_available()}")
    loader = ModelLoader()
    tokenizer, model = loader.load_model()
    print(model.config)