# Adopted from https://github.com/ddlBoJack/SLAM-LLM/blob/main/src/slam_llm/models/encoder.py

import types
import torch
import torch.nn as nn
import torch.nn.functional as F

import whisper
from whisper.model import Whisper, ModelDimensions

from transformers import WhisperModel

class WhisperWrappedEncoder:
    
    @classmethod
    def load(cls, model_config):

        def replace_layer_norm(module):
            from whisper.model import LayerNorm
            for name, child in module.named_children():
                if isinstance(child, LayerNorm):
                    old_params = child.state_dict()
                    new_layer_norm = nn.LayerNorm(child.normalized_shape, eps=child.eps, elementwise_affine=child.elementwise_affine)
                    new_layer_norm.load_state_dict(old_params)
                    setattr(module, name, new_layer_norm)
                else:
                    replace_layer_norm(child)

        
        # load by hand
        checkpoint_file = model_config.speech_encoder
        checkpoint = torch.load(checkpoint_file, map_location="cpu")
        dims = ModelDimensions(**checkpoint["dims"])
        whisper_model = Whisper(dims)
        whisper_model.load_state_dict(checkpoint["model_state_dict"])
        encoder = whisper_model.encoder
        
        # def load(module: nn.Module, prefix=""):
        #     # because zero3 puts placeholders in model params, this context
        #     # manager gathers (unpartitions) the params of the current layer, then loads from
        #     # the state dict and then re-partitions them again
        #     with deepspeed.zero.GatheredParameters(list(module.parameters(recurse=False)), modifier_rank=0):
        #         if deepspeed.comm.get_rank() == 0:
        #             # module._load_from_state_dict(checkpoint["model_state_dict"], prefix)
        #             module.load_state_dict(checkpoint["model_state_dict"])

        #     for name, child in module._modules.items():
        #         if child is not None:
        #             load(child, prefix + name + ".")
        # whisper_model = load(whisper_model)
        # encoder = whisper_model.encoder
        
        # encoder = whisper.load_model(name=model_config.speech_encoder, device='cpu').encoder
        # encoder = WhisperModel.from_pretrained(model_config.speech_encoder, torch_dtype=torch.bfloat16).encoder
        # encoder.requires_grad_(False)
        
        # if ".pt" in model_config.speech_encoder:
        #     # # zero3 not available
        #     encoder = whisper.load_model(name=model_config.speech_encoder, device='cpu').encoder
        # else:
        #     # zero2 or zero3
        #     encoder = WhisperModel.from_pretrained(model_config.speech_encoder, torch_dtype=torch.bfloat16).encoder
        
        
        
        replace_layer_norm(encoder)
        return encoder