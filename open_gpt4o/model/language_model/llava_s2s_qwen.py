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
from open_gpt4o.constants import IMAGE_TOKEN_INDEX, SPEECH_TOKEN_INDEX
from open_gpt4o.model.language_model.llava_qwen import LlavaQwenForCausalLM
from open_gpt4o.model.speech_generator.builder import build_speech_generator
from open_gpt4o.model.speech_generator.generation import GenerationWithCTC
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
                
                loss = self.model.speech_generator(output["hidden_states"][-1], labels, tgt_units)
                
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
                ctc_loss = self.model.speech_generator(output["hidden_states"][-1], labels, tgt_units)
                loss = lm_loss + ctc_loss * self.config.ctc_loss_weight
                
            return CausalLMOutputWithPast(
                loss=loss,
                logits=output.logits,
                past_key_values=output.past_key_values,
                hidden_states=output.hidden_states,
                attentions=output.attentions
            )
        else:
            if dpo_forward:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

                hidden_states = outputs[0]
                logits = self.lm_head(hidden_states)
                return logits, labels

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

        if images is not None:
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
    
    @torch.no_grad()
    def generate_parallel(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        speeches: Optional[torch.FloatTensor] = None,
        speech_lengths: Optional[List[List[int]]] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        streamer = kwargs.pop("streamer", None)
        tokenizer = kwargs.pop("tokenizer", None)
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            if speeches is None:
                (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
            else:
                (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal_av(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes, speeches=speeches, speech_lengths=speech_lengths)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        
        new_query = kwargs.pop("new_query", None)
        new_query_str = kwargs.pop("new_query_str", None)
        query_str = kwargs.pop("query_str", None)
        new_query_pos = kwargs.pop("new_query_pos", 0)
        new_speeches = kwargs.pop("new_speeches", None)
        new_speech_lengths = kwargs.pop("new_speech_lengths", None)
        if query_str is not None:
            query_str = query_str.split("<image>\n")[-1]
            print(f"Human: {query_str}")
        if new_query is not None or new_speeches is not None:
            if new_speeches is not None:
                # HACK batch_size = 1
                assert new_query.shape[0] == 1
                speech_token_indice = torch.where(new_query[0] == SPEECH_TOKEN_INDEX)[0].tolist()[0]
                new_inputs_embeds_1 = self.get_model().embed_tokens(new_query[:,:speech_token_indice])
                new_inputs_embeds_2 = self.get_model().embed_tokens(new_query[:,speech_token_indice+1:])
                speech_features = self.encode_speech(new_speeches, new_speech_lengths)
                speech_features = [speech_feature.to(dtype=new_inputs_embeds_1.dtype) for speech_feature in speech_features]
                speech_features = torch.stack(speech_features)
                new_inputs_embeds = torch.cat([new_inputs_embeds_1, speech_features, new_inputs_embeds_2], dim=1)
                
            else:
                new_inputs_embeds = self.get_model().embed_tokens(new_query)
        else:
            new_inputs_embeds = None
        bsz, seq = inputs_embeds.shape[:-1]

        # get channel ids 10: system; 14: 13: system+template
        prefix_length = 13 + 32 * 144 + 1 # system + video # prefix prompt + frame
        channel = [0] * prefix_length + [1] * (seq - prefix_length)
        # # construct new attention masks by flexattention
        def prefix_mask(b, h, q_idx, kv_idx):
            return kv_idx <= prefix_length
        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx
        def transition_attention_mask(b, h, q_idx, kv_idx):
            prefix_mask = kv_idx <= prefix_length
            causal_mask = q_idx >= kv_idx
            channel_tensor = torch.tensor(channel, dtype=torch.long, device=new_inputs_embeds.device)
            block_mask = channel_tensor[q_idx] == channel_tensor[kv_idx]
            return (prefix_mask | block_mask) & causal_mask

        
        past_key_values = DynamicCache()
        max_cache_length = past_key_values.get_max_length()
        cache_position = torch.arange(seq, dtype=torch.int64, device=inputs_embeds.device)
        max_new_tokens = kwargs.pop("max_new_tokens", 1024)
        for idx in range(max_new_tokens):
            
            # discern noise
            if idx == new_query_pos:
                start_time = time.time()
                print()
                if new_query_str is not None:
                    print(f"Human: {new_query_str}")
                else:
                    print(f"Human: <new query>")
                # # encounter new query 
                # new_attention_mask = create_mask(transition_attention_mask, 1, 1, bsz, seq)
                
                # outputs_ids = torch.cat([outputs_ids, new_query], dim=-1)
                
                ori_length = attention_mask.shape[1]
                
                # # (1) naive mask: mask previous input query
                # attention_mask[:, prefix_length:] = 0 # mask prev attention
                # cur_length = attention_mask.shape[1] - prefix_length
                # # print(attention_mask.shape)
                # attention_mask = torch.cat([attention_mask[:,:-1], attention_mask.new_ones((new_inputs_embeds.shape[0], new_inputs_embeds.shape[1]))], dim=-1)
                # (2) parallel mask
                seq = ori_length + new_inputs_embeds.shape[1]
                channel += [channel[-1]+1] * new_inputs_embeds.shape[1]
                dtype_min = torch.finfo(new_inputs_embeds.dtype).min
                
                # # 1. flexattention
                # attention_mask_4d = create_mask(transition_attention_mask, 1, 1, seq, seq)
                # attention_mask_4d = attention_mask_4d[:, :, -new_inputs_embeds.shape[1]:, :]
                # attention_mask_4d = torch.where(attention_mask_4d, torch.tensor(0.0, device=new_inputs_embeds.device, dtype=new_inputs_embeds.dtype), torch.tensor(dtype_min, device=new_inputs_embeds.device, dtype=new_inputs_embeds.dtype))
                # 2. torch implementation
                attention_mask_4d = transition_attention_mask_pt(1, 1, list(range(seq)), list(range(seq)), prefix_length, channel, new_inputs_embeds.device).unsqueeze(0).unsqueeze(0)
                attention_mask_4d = attention_mask_4d[:, :, -new_inputs_embeds.shape[1]-1:, :]
                
                
                attention_mask_4d = torch.where(
                    attention_mask_4d,
                    torch.tensor(0.0, device=new_inputs_embeds.device, dtype=new_inputs_embeds.dtype),
                    torch.tensor(dtype_min, device=new_inputs_embeds.device, dtype=new_inputs_embeds.dtype)
                )
                # attention_mask_4d = torch.where(attention_mask_4d, torch.tensor(0.0, device=new_inputs_embeds.device, dtype=new_inputs_embeds.dtype), torch.tensor(dtype_min, device=new_inputs_embeds.device, dtype=new_inputs_embeds.dtype))
                
                cache_position = torch.arange(cache_position[-1], cache_position[-1]+new_inputs_embeds.shape[1]+1, dtype=torch.int64, device=inputs_embeds.device)
                
                # print("*"*24)
                # print(new_inputs_embeds.shape)
                # print(attention_mask_4d.shape)
                # print("*"*24)
                
                outputs = super().forward(
                    attention_mask=attention_mask_4d,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    inputs_embeds=torch.cat([inputs_embeds, new_inputs_embeds],dim=1),
                    use_cache=True,
                )
                
                
                
                temperature = 1
                next_prob = torch.softmax(outputs.logits[:,-1]/temperature, dim=-1)
                next_entropy = -torch.sum(next_prob*torch.log(next_prob+1e-5), dim=-1)
                next_threshold = 0.09
                next_alpha = 0.3
                noise_threshold = torch.minimum(
                    torch.ones_like(next_entropy) * next_threshold,
                    torch.exp(-next_entropy) * next_alpha
                )
                noise_prob = next_prob[:, 151644] # |<im_start>|
                # HACK batchsize must be 1
                if noise_prob[0] > noise_threshold[0]:
                    print("Assistant: ", end='', flush=True)
                    # # (1) naive mask: if noise: continue generate, give up the query, revert attention
                    # attention_mask[:, prefix_length:prefix_length+cur_length] = 1 # revert prev attention
                    # attention_mask[:, prefix_length+cur_length:-1] = 0 # mask noise attention
                    # attention_mask = torch.cat([attention_mask, attention_mask.new_ones((bsz, 1))], dim=-1)
                    # cache_position = cache_position[-1:] + 1
                    # (2) parallel mask: revert mask and crop KVCache
                    attention_mask = torch.cat([attention_mask, attention_mask.new_ones(bsz, 1)], dim=-1)
                    past_key_values.crop(ori_length)
                    cache_position = cache_position[:1] + 1
                    channel += [channel[-1]-1]
                    next_token_ids = outputs.logits[:, :1].argmax(-1)
                    if tokenizer is not None:
                        print(tokenizer.decode(next_token_ids.squeeze(0), skip_special_tokens=True), end='', flush=True)
                    outputs_ids = torch.cat([outputs_ids, next_token_ids], dim=-1)
                    inputs_embeds = self.get_model().embed_tokens(next_token_ids)
                    
                    
                else:
                    # if not noise: change channel
                    # print("*******start new topic*******")
                    print("Assistant: ", end='', flush=True)
                    next_token_ids = outputs.logits[:, -1:].argmax(-1)
                    outputs_ids = torch.cat([outputs_ids, next_token_ids], dim=-1)
                    if tokenizer is not None:
                        print(tokenizer.decode(next_token_ids.squeeze(0), skip_special_tokens=True), end='', flush=True)
                    if streamer is not None:
                        streamer.stream(output_ids=next_token_ids)
                    inputs_embeds = self.get_model().embed_tokens(next_token_ids)
                    # # (1) naive mask: 2d mask
                    # attention_mask = torch.cat([attention_mask, attention_mask.new_ones((bsz, 1))], dim=-1)
                    # cache_position = cache_position[-1:] + 1
                    # channel += [channel[-1]]
                    # (2) paprallel mask for simplicity: mask previous query TODO: consider previous query
                    attention_mask[:, prefix_length:] = 0 # mask prev attention
                    attention_mask = torch.cat([attention_mask[:,:-1], attention_mask.new_ones((new_inputs_embeds.shape[0], new_inputs_embeds.shape[1]+1))], dim=-1)
                    attention_mask = torch.cat([attention_mask, attention_mask.new_ones((bsz, 1))], dim=-1)
                    cache_position = cache_position[-1:] + 1
                    channel += [channel[-1]]
                
                end_time = time.time()
                print("new_query inference time: ", end_time-start_time)
            
            else:
                outputs = super().forward(
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    inputs_embeds=inputs_embeds,
                    use_cache=True,
                )
                
                # greedy sample
                next_token_ids = outputs.logits[:, -1:].argmax(-1)
                if idx == 0:
                    outputs_ids = next_token_ids
                else:
                    outputs_ids = torch.cat([outputs_ids, next_token_ids], dim=-1)
                if tokenizer is not None:
                    if idx == 0: print("Assistant: ", end='', flush=True)
                    print(tokenizer.decode(next_token_ids.squeeze(0), skip_special_tokens=True), end='', flush=True)
                if streamer is not None:
                    streamer.stream(output_ids=next_token_ids)
                inputs_embeds = self.get_model().embed_tokens(next_token_ids)
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones((bsz, 1))], dim=-1)
                cache_position = cache_position[-1:] + 1
                channel += [channel[-1]]
                
            if next_token_ids.squeeze(0)[0].item() == tokenizer.eos_token_id:
                print()
                print("="*20,"COMPLETE","="*20)
                break

        return outputs_ids


    @torch.no_grad()
    def generate_stream(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        speeches: Optional[torch.FloatTensor] = None,
        speech_lengths: Optional[List[List[int]]] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        tokenizer = kwargs.pop("tokenizer", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        
        if images is not None:
            if speeches is None:
                (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
            else:
                (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal_av(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes, speeches=speeches, speech_lengths=speech_lengths)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        
        bsz, seq = inputs_embeds.shape[:-1]
        past_key_values = DynamicCache()
        cache_position = torch.arange(seq, dtype=torch.int64, device=inputs_embeds.device)
        max_new_tokens = kwargs.pop("max_new_tokens", 1024)
        for idx in range(max_new_tokens):
            if idx == 10 or idx == 0: start_time = time.time()
            outputs = super().forward(
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                inputs_embeds=inputs_embeds,
                use_cache=True,
            )
            
            # greedy sample
            next_token_ids = outputs.logits[:, -1:].argmax(-1)
            if idx == 0:
                outputs_ids = next_token_ids
            else:
                outputs_ids = torch.cat([outputs_ids, next_token_ids], dim=-1)
            if tokenizer is not None:
                if idx == 0: print("Assistant: ", end='', flush=True)
                print(tokenizer.decode(next_token_ids.squeeze(0), skip_special_tokens=True), end='', flush=True)
            inputs_embeds = self.get_model().embed_tokens(next_token_ids)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((bsz, 1))], dim=-1)
            cache_position = cache_position[-1:] + 1
            
            if idx == 10 or idx == 0: print("forward time: ", time.time()-start_time)
            
            if next_token_ids.squeeze(0)[0].item() == tokenizer.eos_token_id:
                print()
                print("="*20,"COMPLETE","="*20)
                break

        return outputs_ids

    
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


