import os
import torch, torchvision, transformers, collections
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# torchvision.set_video_backend('video_reader')
from dataclasses import asdict
# from torchvision.io import read_video
from decord import VideoReader, cpu

from transformers.cache_utils import (
    DynamicCache,
    SinkCache,
    StaticCache,
    SlidingWindowCache,
    QuantoQuantizedCache,
    QuantizedCacheConfig,
)


logger = transformers.logging.get_logger('liveinfer')


from open_omni.conversation import conv_templates, SeparatorStyle
from open_omni.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from open_omni.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token, KeywordsStoppingCriteria
from open_omni.vid_utils import load_video
from open_omni.model.builder import load_pretrained_model
from .arguments_live import parse_args
from .inference_util import MaxHeapDict



# python -m demo.cli --resume_from_checkpoint ... 

class LiveInfer:
    def __init__(self, ) -> None:
        # model
        args = parse_args()
        self.device = args.device
        model_name = get_model_name_from_path(args.model_path)
        llava_model_args = {"multimodal": True}
        if args.attn_implementation is not None:
            llava_model_args["attn_implementation"] = args.attn_implementation
        overwrite_config = {}
        overwrite_config["mm_spatial_pool_stride"] = 2
        overwrite_config["mm_spatial_pool_mode"] = "average"
        llava_model_args["overwrite_config"] = overwrite_config
        self.tokenizer, self.model, self.processor, self.context_len = load_pretrained_model(args.model_path, None, model_name, device_map=args.device, **llava_model_args)
        if 'qwen' in args.model_path.lower():
            conv_mode = 'qwen_1_5'
        elif 'llama3' in args.model_path.lower():
            conv_mode = 'llava_llama_3'
        elif 'mistral' in args.model_path.lower():
            conv_mode = 'mistral_instruct'
        else:
            conv_mode = "llava_v1"
        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
        else:
            args.conv_mode = conv_mode
        self.conv = conv_templates[args.conv_mode].copy()
            
        
        # self.model, self.tokenizer = build_model_and_tokenizer(is_training=False, set_vision_inside=True, **asdict(args))
        # self.model.to('cuda')
        
        # file
        self.video_file = args.video_file

        # visual
        # self.hidden_size = self.model.config.hidden_size
        self.frame_fps = args.frame_fps
        self.frame_interval = 1 / self.frame_fps
        self.frame_resolution = args.frame_resolution
        # self.frame_resolution = self.model.config.frame_resolution
        # self.frame_num_tokens = self.model.config.frame_num_tokens
        # self.frame_v_placeholder = self.model.config.v_placeholder * self.frame_num_tokens
        # self.frame_token_interval_id = self.model.config.frame_token_interval_id
        # self.frame_placeholder_ids = torch.tensor(self.model.config.v_placeholder_id).repeat(self.model.config.frame_num_tokens).reshape(1,-1)
        
        # generation
        self.system_prompt = args.system_prompt
        self.inplace_output_ids = torch.zeros(1, 100, device='cuda', dtype=torch.long)
        self.frame_token_interval_threshold = 0.725
        self.eos_token_id = self.tokenizer.eos_token_id
        # self._start_ids = self.tokenizer.apply_chat_template([{'role': 'system', 'content': self.system_prompt}], add_stream_prompt=True, return_tensors='pt').to('cuda')
        # self._added_stream_prompt_ids = self.tokenizer.apply_chat_template([{}], add_stream_prompt=True, return_tensors='pt').to('cuda')
        # self._added_stream_generation_ids = self.tokenizer.apply_chat_template([{}], add_stream_generation_prompt=True, return_tensors='pt').to('cuda')
        
        # app
        self.reset()
    
    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=batch_first, padding_value=padding_value
        )
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    @torch.no_grad()
    def _call_for_response(self, video_time, query):
        # when there is a query, just add it to the LLM prompt
        if query is not None:
            if self.conv.messages:
                self.conv.messages[-1][-1] = self.current_output
            self.conv.append_message(self.conv.roles[0], '\n'+query)
            self.conv.append_message(self.conv.roles[1], None)
            query = f'(Current Time = {video_time}s) User: {query}'
            return query, None
        prompt = self.conv.get_prompt()
        inputs_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").to(self.device)
        
        # TODO: support batch inference
        # input_ids = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")]
        # pad_token_ids = (
        #     self.tokenizer.pad_token_id
        #     if self.tokenizer.pad_token_id is not None
        #     else self.tokenizer.eos_token_id
        # )
        # input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_ids).to(self.device)
        # attention_masks = input_ids.ne(pad_token_ids).to(self.device)
        # stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2
        # keywords = [stop_str]
        # stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        
        # TODO: support interleaved image multi-turn
        # split input query: 
        # single turn: ['<|im_start|>', 'system', 'Ċ', 'You', 'Ġare', 'Ġa', 'Ġhelpful', 'Ġassistant', '.', '<|im_end|>', 'Ċ', '<|im_start|>', 'user', 'Ċ', None, 'query', '<|im_end|>', 'Ċ', '<|im_start|>', 'assistant', 'Ċ']
        # mutlip turn: ['<|im_start|>', 'system', 'Ċ', 'You', 'Ġare', 'Ġa', 'Ġhelpful', 'Ġassistant', '.', '<|im_end|>', 'Ċ', '<|im_start|>', 'user', 'Ċ', None, 'Ċ', 'query', '<|im_end|>', 'Ċ', '<|im_start|>', 'assistant', 'Ċ', 'yes', '<|im_end|>', 'Ċ', '<|im_start|>', 'user', 'Ċ', 'query2', '<|im_end|>', 'Ċ', '<|im_start|>', 'assistant', 'Ċ']
        # print(inputs_ids)
        
        # print("##before: ", self.tokenizer.decode(inputs_ids))
        inputs_ids = inputs_ids[self.vis_index:]
        # print("##after: ", self.tokenizer.decode(inputs_ids))
        
        inputs_embeds = self.model.get_model().embed_tokens(inputs_ids.unsqueeze(0))
        output_ids, self.past_key_values = self.model.generate_streaming(
            inputs_embeds,
            past_key_values=self.past_key_values,
            tokenizer=self.tokenizer
        )
        post_length = inputs_embeds.shape[-2] + output_ids.shape[-1] - 1
        self.past_key_values.crop(self.past_key_values.get_seq_length() - post_length)
        output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        self.current_output = output_text
        
        self.query_inputs_ids = inputs_ids
        if query:
            query = f'(Current Time = {video_time}s) User: {query}'
        if type(video_time) is tuple:
            response = f'(Current Time = {video_time[0]}s) Assistant: REQUIREMENT MEET AT {video_time[1]}s '
        else:
            response = f'(Current Time = {video_time}s) Assistant:{output_text}'
        return query, response
    
    @torch.no_grad()
    def _call_for_streaming(self, ):
        while self.frame_embeds_queue:
            # 1. encouter first query: add it to text prompt, print the query 
            if self.query_queue and self.frame_embeds_queue[0][0] > self.query_queue[0][0]:
                video_time, query = self.query_queue.popleft()
                return video_time, query
            video_time, frame_embeds = self.frame_embeds_queue.popleft() # frame_embeds, N_patch, H
            
            # add new frame to KVCache
            if self.past_key_values is None: # initialize
                # prefix_prompt
                # qwen -> CHATML
                prefix_prompt = "" if self.conv.system == "" else self.conv.system + self.conv.sep + "\n"
                prefix_prompt += self.conv.roles[0] + "\n"
                prefix_inputs_ids = self.tokenizer(prefix_prompt).input_ids
                prefix_inputs_ids = torch.tensor(prefix_inputs_ids, dtype=torch.long).to(self.device)
                self.vis_index = prefix_inputs_ids.shape[-1]
                # print("prefix: ")
                # print(self.tokenizer.decode(prefix_inputs_ids))
                # post_prompt + query
                post_prompt = "\n" + self.query_queue[-1][1] + self.conv.sep + "\n" + self.conv.roles[1] + "\n"
                query_inputs_ids = self.tokenizer(post_prompt).input_ids
                self.query_inputs_ids = torch.tensor(query_inputs_ids, dtype=torch.long).to(self.device)
                # kvcache
                self.past_key_values = DynamicCache()
                inputs_embeds = torch.cat([
                    self.model.get_model().embed_tokens(prefix_inputs_ids.unsqueeze(0)),
                    frame_embeds.unsqueeze(0),
                    self.model.get_model().embed_tokens(self.query_inputs_ids.unsqueeze(0))
                ], dim=1)
            else:
                inputs_embeds = torch.cat([
                    frame_embeds.unsqueeze(0),
                    self.model.get_model().embed_tokens(self.query_inputs_ids.unsqueeze(0))
                ], dim=1)
            self.frame_count += 1
            # TODO interleaved multiple turn
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                past_key_values=self.past_key_values,
                output_attentions=True,
                return_dict=True
            )
            self.past_key_values.crop(self.past_key_values.get_seq_length() - self.query_inputs_ids.shape[-1])
            
            # 2. encouter new query, add it to the text prompt of the LLM
            if self.query_queue and video_time >= self.query_queue[0][0]:
                video_time, query = self.query_queue.popleft()
                return video_time, query
            
            # 3. grounding to future
            if self.frame_count > 4:
                attentions = outputs.attentions[-1].squeeze(0)
                # print("nan shape: ", torch.nonzero(torch.isnan(attentions), as_tuple=False))
                non_nan_mask = ~torch.isnan(attentions)
                attentions = torch.where(non_nan_mask, attentions, torch.tensor(0.0))
                
                # attentions = attentions.mean(0)[-3, :]
                # ground_attention = attentions[self.vis_index:self.vis_index+144*self.frame_count]
                # ground_score = ground_attention.reshape(-1, 144).mean(dim=-1)
                attentions = attentions.mean(0)[-self.query_inputs_ids.shape[-1]:, :]
                attentions = attentions[:, self.vis_index:self.vis_index+144*self.frame_count]
                attentions = attentions.reshape(self.query_inputs_ids.shape[-1], -1, 144).mean(dim=-1)
                ground_score = attentions[-3]
                
                # grounding algorithm: when looking forward, when a frame always larger than expectation + variance, we tag it as a hit 
                std, mean = torch.std_mean(ground_score)
                threshold = mean + 1.5*std
                salients = ground_score > threshold
                salients = salients.nonzero().squeeze(-1).tolist()
                # # topk
                # salients = torch.topk(ground_score, 1).indices.sort()[0].tolist()
                # # print(salients)
                
                for sa in salients:
                    self.salient.add_or_update(sa, -self.salient.entry_finder.get(sa, (0, -1, -1))[0]+1)
                if self.salient.heap:
                    sa, cnt = self.salient.peek_max()
                    # print(self.salient.entry_finder)
                    # print(sa, cnt) # debug score
                    # delay 4 seconds to improve the accuracy of the grounding
                    if cnt > max(4, self.frame_fps * 3) and sa not in self.highlight_points: # forward step
                        self.highlight_points.append(sa)
                        # return video_time, None
                        return (video_time, sa / self.frame_fps), None
                
                def visualize_attention(attention_weights, idx=None):
                    """
                    Visualizes a single attention weight matrix.

                    Args:
                        attention_weights (torch.Tensor): The attention weights (T, T) for a sequence.
                    """
                    # Replace NaNs with zeros
                    attention_weights = torch.nan_to_num(attention_weights, nan=0.0)

                    # Convert to numpy array if needed
                    attention = attention_weights.cpu().float().numpy()
                    attention = (attention - np.min(attention)) / (np.max(attention) - np.min(attention))

                    # downsample
                    # l = attention.shape[0]
                    # attention = attention.reshape(l//144, 144, l//144, 144).mean(axis=(1,3))
                    ql, kl = attention.shape
                    # attention = attention.reshape(ql, kl//144, 144).sum(axis=2)

                    # Set up the matplotlib figure
                    plt.figure(figsize=(10, 8))

                    # Create a heatmap for the attention weights
                    # sns.heatmap(attention, cmap="viridis", cbar=True, mask=np.isnan(attention))
                    sns.heatmap(attention, cmap="viridis", cbar=True)

                    # Set titles and labels
                    plt.title('Attention Weights')
                    plt.xlabel('Key Index')
                    plt.ylabel('Query Index')

                    plt.tight_layout()

                    # plt.show()
                    if idx:
                        plt.savefig(f"visualization/attn_debug_{idx}.png")
                    else:
                        plt.savefig("visualization/attn_debug.png")
                
                # # attention visualization
                # if self.frame_count == 30:
                #     attentions = attentions.mean(0)[-self.query_inputs_ids.shape[-1]:, :]
                #     attentions = attentions[:, self.vis_index:self.vis_index+144*self.frame_count]
                #     attentions = attentions.reshape(self.query_inputs_ids.shape[-1], -1, 144).mean(dim=-1)
                #     visualize_attention(attentions)
                #     print(attentions[-3])

                # attentions = attentions.mean(0)[-self.query_inputs_ids.shape[-1]:, :]
                # attentions = attentions[:, self.vis_index:self.vis_index+144*self.frame_count]
                # attentions = attentions.reshape(self.query_inputs_ids.shape[-1], -1, 144).mean(dim=-1)
                # print(attentions[-3])
                # visualize_attention(attentions[-3].unsqueeze(0), self.frame_count)
            
        return None, None
    
    def reset(self, ):
        self.query_queue = collections.deque()
        self.frame_embeds_queue = collections.deque()
        self.salient = MaxHeapDict()
        self.highlight_points = collections.deque()
        self.current_output = None
        self.frame_count = 0
        self.video_time = 0
        self.last_frame_idx = -1
        self.video_tensor = None
        self.last_ids = torch.tensor([[]], device='cuda', dtype=torch.long)
        self.past_key_values = None

    def input_query_stream(self, query, history=None, video_time=None):
        if video_time is None:
            self.query_queue.append((self.video_time, query))
        else:
            self.query_queue.append((video_time, query))
        if not self.past_key_values:
            return f'(NOTE: No video stream here. Please select or upload a video. Then the assistant will answer "{query} (at {self.video_time}s)" in the video stream)'
        return f'(NOTE: Received "{query}" (at {self.video_time}s). Please wait until previous frames have been processed)'
    
    def input_video_stream(self, video_time):
        
        frame_idx = int(video_time * self.frame_fps)
        if frame_idx > self.last_frame_idx:
            # print('frame_idx ', frame_idx)
            # print('last_frame_idx ', self.last_frame_idx)
            ranger = range(self.last_frame_idx + 1, frame_idx + 1)
            
            # encode video
            video_idx_in_batch = [0]
            images_list = [self.video_tensor[ranger]] # self.video_tensor: L, H, W, C
            concat_images = torch.cat([image for image in images_list], dim=0)
            split_sizes = [image.shape[0] for image in images_list]
            frames_embeds = self.model.encode_multimodals(concat_images, video_idx_in_batch, split_sizes) #[ (L, N, H)
            # unires + video = flat
            # frames_embeds = [x.flatten(0, 1) for x in image_features] # [L*N, H]
            # TODO unires image
            
            frames_embeds = frames_embeds[0] # L, N, H
            self.frame_embeds_queue.extend([(r/self.frame_fps, frame_embeds) for r, frame_embeds in zip(ranger, frames_embeds)]) # 
            
            
        self.last_frame_idx = frame_idx
        self.video_time = video_time
    
    def load_videos(self, video_path):
        
        if os.path.isdir(video_path):
            video_tensor = load_video(video_path, video_decode_backend='frame', fps=self.frame_fps)
        elif video_path.endswith(".gif"):
            video_tensor = load_video(video_path, video_decode_backend='gif', fps=self.frame_fps)
        else:
            video_tensor = load_video(video_path, fps=self.frame_fps, max_frames=512) # T H W C
        self.video_tensor = self.processor.preprocess(video_tensor, return_tensors="pt")['pixel_values'].to(torch.bfloat16).to(self.device)
        
        # print(self.video_tensor.shape)
        self.num_video_frames = self.video_tensor.shape[0]
        self.video_duration = self.video_tensor.shape[0] / self.frame_fps
        logger.warning(f'{video_path} -> {self.video_tensor.shape}, {self.frame_fps} FPS')

    def __call__(self, ):
        while not self.frame_embeds_queue:
            continue
        video_time, query = self._call_for_streaming()
        response = None
        if video_time is not None:
            query, response = self._call_for_response(video_time, query)
        return query, response