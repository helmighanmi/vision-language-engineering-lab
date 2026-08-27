# Path: examples/qwen_model_selection.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Show the supported Qwen3-VL model-selection modes without loading model weights."""

from vlm_engineering import QwenVLModel


def main() -> None:
    default_model = QwenVLModel()
    balanced_model = QwenVLModel(model_size="4b")
    larger_model = QwenVLModel(model_size="8b")
    custom_model = QwenVLModel.from_hub("Qwen/Qwen3-VL-4B-Instruct")

    print("Default:", default_model.model_source)
    print("4B preset:", balanced_model.model_source)
    print("8B preset:", larger_model.model_source)
    print("Custom Hub ID:", custom_model.model_source)


if __name__ == "__main__":
    main()
