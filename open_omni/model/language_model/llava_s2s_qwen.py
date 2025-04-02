import time
from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

# from ...constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from open_omni.constants import IMAGE_TOKEN_INDEX, SPEECH_TOKEN_INDEX
from open_omni.model.language_model.llava_qwen import LlavaQwenForCausalLM
from open_omni.model.speech_generator.builder import build_speech_generator
from open_omni.model.speech_generator.generation import GenerationWithCTC
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM

# from .qwen.modeling_qwen import QWenLMHeadModel, QWenModel
# from .qwen.configuration_qwen import QWenConfig

from transformers.cache_utils import (
    DynamicCache,
    SinkCache,
    StaticCache,
    SlidingWindowCache,
    QuantoQuantizedCache,
    QuantizedCacheConfig,
)

from torch.nn.attention.flex_attention import create_mask, create_block_mask
import torch


def transition_attention_mask_pt(b, h, q_idx, kv_idx, prefix_length, channel, device):
    q_idx = torch.tensor(q_idx, dtype=torch.long, device=device)
    kv_idx = torch.tensor(kv_idx, dtype=torch.long, device=device)
    
    # Validate indices
    if q_idx.max() >= len(channel) or kv_idx.max() >= len(channel):
        raise ValueError("Index out of bounds. Ensure q_idx and kv_idx are within the valid range.")
    
    # Allow attention to prefix positions
    prefix_mask = kv_idx.unsqueeze(0) < prefix_length
    # Allow attention within the same channel
    channel_tensor = torch.tensor(channel, dtype=torch.long, device=device)
    block_mask = channel_tensor[q_idx].unsqueeze(-1) == channel_tensor[kv_idx]
    # Causal mask to prevent attending to future positions
    causal_mask = q_idx.unsqueeze(-1) >= kv_idx
    # Combine masks: prefix or block should be true, and must satisfy causal constraints
    combined_mask = (prefix_mask | block_mask) & causal_mask
    return combined_mask


class LlavaS2SQwenConfig(Qwen2Config):
    model_type = "llava_s2s_qwen"


class LlavaS2SQwenForCausalLM(LlavaQwenForCausalLM, GenerationWithCTC):
    config_class = LlavaS2SQwenConfig

    def __init__(self, config):
        super().__init__(config)
        
        self.post_init()
        # if hasattr(config, "speech_generator_type"):
        #     self.speech_generator = build_speech_generator(config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        speeches: Optional[torch.FloatTensor] = None,
        speech_lengths: Optional[List[List[int]]] = None,
        tgt_units: Optional[torch.LongTensor] = None,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            if speeches is None:
                (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes)
            else:
                (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal_av(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes, speeches, speech_lengths)

        
        if self.training:
            
            if self.model.tune_speech_generator_only:
                with torch.no_grad():
                    output = super(LlavaQwenForCausalLM, self).forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        inputs_embeds=inputs_embeds,
                        labels=labels,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                        output_hidden_states=True,
                        return_dict=return_dict,
                    )
                
                loss = self.model.speech_generator(output["hidden_states"][-1][..., :-1, :].contiguous(), labels[..., 1:].contiguous(), tgt_units)
                # loss = self.model.speech_generator(output["hidden_states"][-1], labels, tgt_units)
                
            else:
                output = super(LlavaQwenForCausalLM, self).forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=True,
                    return_dict=return_dict,
                )
                lm_loss = output.loss
                if tgt_units is not None:
                    ctc_loss = self.model.speech_generator(output["hidden_states"][-1], labels, tgt_units)
                    loss = lm_loss + ctc_loss * self.config.ctc_loss_weight
                else:
                    loss = lm_loss
                
            return CausalLMOutputWithPast(
                loss=loss,
                logits=output.logits,
                past_key_values=output.past_key_values,
                hidden_states=output.hidden_states,
                attentions=output.attentions
            )
        else:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        speeches: Optional[torch.FloatTensor] = None,
        speech_lengths: Optional[List[List[int]]] = None,
        streaming_unit_gen=False,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None or speeches is not None:
            if speeches is None:
                (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
            else:
                (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal_av(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes, speeches=speeches, speech_lengths=speech_lengths)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        outputs = GenerationWithCTC.generate(
            self,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            return_dict_in_generate=True,
            streaming_unit_gen=streaming_unit_gen,
            **kwargs
        )
        hidden_states = outputs["hidden_states"]
        hidden_states = torch.cat([hidden_states[0][-1][:, -1:, :]] + [hidden_states[i][-1] for i in range(1, len(hidden_states))], dim=1)
        ctc_pred = self.model.speech_generator.predict(hidden_states.squeeze(0))
        
        
        
        return outputs.sequences, ctc_pred
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        speech = kwargs.pop("speech", None)
        speech_lengths = kwargs.pop("speech_lengths", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        if speech is not None:
            inputs["speech"] = speech
        if speech_lengths is not None:
            inputs["speech_lengths"] = speech_lengths
        return inputs


AutoConfig.register("llava_s2s_qwen", LlavaS2SQwenConfig)
AutoModelForCausalLM.register(LlavaS2SQwenConfig, LlavaS2SQwenForCausalLM)


