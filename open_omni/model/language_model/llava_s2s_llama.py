#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig

from torch.nn import CrossEntropyLoss


# , LlamaModel, LlamaForCausalLM, GenerationConfig
# from .modeling_llama import LlamaModel, LlamaForCausalLM
from transformers import LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

# from open_omni.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from open_omni.model.language_model.llava_llama import LlavaLlamaForCausalLM
from open_omni.model.speech_generator.generation import GenerationWithCTC


class LlavaS2SLlamaConfig(LlamaConfig):
    model_type = "llava_s2s_llama"


class LlavaS2SLlamaForCausalLM(LlavaLlamaForCausalLM, GenerationWithCTC):
    config_class = LlavaS2SLlamaConfig

    def __init__(self, config):
        super().__init__(config)
        self.post_init()

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
        dpo_forward: Optional[bool] = None,
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
                    output = super(LlavaLlamaForCausalLM, self).forward(
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
                
                # loss = self.model.speech_generator(output["hidden_states"][-1][..., :-1, :].contiguous(), labels[..., 1:].contiguous(), tgt_units)
                loss = self.model.speech_generator(output["hidden_states"][-1], labels, tgt_units)
                
            else:
                output = super(LlavaLlamaForCausalLM, self).forward(
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


AutoConfig.register("llava_s2s_llama", LlavaS2SLlamaConfig)
AutoModelForCausalLM.register(LlavaS2SLlamaConfig, LlavaS2SLlamaForCausalLM)
