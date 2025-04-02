import os

AVAILABLE_MODELS = {
    "llava_llama": "LlavaLlamaForCausalLM, LlavaConfig",
    "llava_s2s_llama": "LlavaS2SLlamaForCausalLM, LlavaS2SLlamaConfig",
    "llava_qwen": "LlavaQwenForCausalLM, LlavaQwenConfig",
    "llava_s2s_qwen": "LlavaS2SQwenForCausalLM, LlavaS2SQwenConfig",
    "llava_mistral": "LlavaMistralForCausalLM, LlavaMistralConfig",
    # Add other models as needed
}

for model_name, model_classes in AVAILABLE_MODELS.items():
    try:
        exec(f"from .language_model.{model_name} import {model_classes}")
    except Exception as e:
        raise e
        print(f"Failed to import {model_name} from longva.language_model.{model_name}")
