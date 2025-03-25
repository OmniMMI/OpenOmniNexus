import argparse
import copy
import math
import warnings
import json
import os
from datetime import timedelta
from typing import List, Optional, Union, Tuple

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from decord import VideoReader, cpu
from packaging import version
from transformers import AutoConfig, TextIteratorStreamer, TextStreamer
from threading import Thread
from loguru import logger as eval_logger

import ChatTTS
chat = ChatTTS.Chat()
chat.load(source='local', compile=True)
from num2words import num2words
import re

import whisper
import torchaudio

def clean_text(text):
    def replace(match):
        num = match.group(0)
        return num2words(num)
    text = text.replace("-", "to").replace("\n", " ")
    return re.sub(r'\b\d+\b', replace, text)

torch.backends.cuda.matmul.allow_tf32 = True

warnings.filterwarnings("ignore")

try:
    from open_omni.constants import (
        IMAGE_TOKEN_INDEX,
        DEFAULT_IMAGE_TOKEN,
        SPEECH_TOKEN_INDEX,
        DEFAULT_SPEECH_TOKEN
    )
    from open_omni.conversation import conv_templates, SeparatorStyle
    from open_omni.mm_utils import (
        get_model_name_from_path,
        process_images,
        tokenizer_image_token,
        tokenizer_image_speech_tokens,
        KeywordsStoppingCriteria,
    )
    from open_omni.model.builder import load_pretrained_model
except Exception as e:
    eval_logger.debug(
        "open_omni is not installed. Please install open_omni to use this model.\nError: %s" % e
    )
    raise NotImplementedError("no valid open_omni")

if version.parse(torch.__version__) >= version.parse("2.1.2"):
    best_fit_attn_implementation = "sdpa"
else:
    best_fit_attn_implementation = "eager"


class OpenOmni:
    """
    OpenOmni Model
    """

    def __init__(
        self,
        pretrained: str = "ColorfulAI/M4-LongVA-7B-Qwen2",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        model_name: Optional[str] = None,
        attn_implementation: Optional[str] = best_fit_attn_implementation,
        device_map: Optional[str] = "cuda:0",
        conv_template: Optional[str] = "qwen_1_5",
        use_cache: Optional[bool] = True,
        truncate_context: Optional[
            bool
        ] = False,  # whether to truncate the context in generation, set it False for LLaVA-1.6
        customized_config: Optional[str] = None,  # ends in json
        max_frames_num: Optional[int] = 32,
        mm_spatial_pool_stride: Optional[int] = 2,
        mm_spatial_pool_mode: Optional[str] = "average",
        token_strategy: Optional[
            str
        ] = "single",  # could be "single" or "multiple", "multiple" denotes adding multiple <image> tokens for each frame in the context
        video_decode_backend: str = "decord",
        # video_decode_backend: str = "opencv",
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        # Replace accelerate with native PyTorch multi-GPU setup
        if torch.cuda.device_count() > 1:
            self._device = torch.device(f"cuda:{torch.distributed.get_rank()}")
            self.device_map = f"cuda:{torch.distributed.get_rank()}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map

        llava_model_args = {
            "multimodal": True,
        }

        if customized_config is not None:
            llava_model_args["customized_config"] = customized_config
        if attn_implementation is not None:
            llava_model_args["attn_implementation"] = attn_implementation
        if "use_flash_attention_2" in kwargs:
            llava_model_args["use_flash_attention_2"] = kwargs["use_flash_attention_2"]

        model_name = (
            model_name
            if model_name is not None
            else get_model_name_from_path(pretrained)
        )

        self.pretrained = pretrained
        self.token_strategy = token_strategy
        self.max_frames_num = max_frames_num
        self.mm_spatial_pool_stride = mm_spatial_pool_stride
        self.mm_spatial_pool_mode = mm_spatial_pool_mode
        self.video_decode_backend = video_decode_backend

        overwrite_config = {}
        overwrite_config["mm_spatial_pool_stride"] = self.mm_spatial_pool_stride
        overwrite_config["mm_spatial_pool_mode"] = self.mm_spatial_pool_mode
        cfg_pretrained = AutoConfig.from_pretrained(self.pretrained)

        llava_model_args["overwrite_config"] = overwrite_config
        try:
            # Try to load the model with the multimodal argument
            self._tokenizer, self._model, self._image_processor, self._max_length = (
                load_pretrained_model(
                    pretrained,
                    None,
                    model_name,
                    device_map=self.device_map,
                    **llava_model_args,
                )
            )
        except TypeError:
            # for older versions of LLaVA that don't have multimodal argument
            llava_model_args.pop("multimodal", None)
            self._tokenizer, self._model, self._image_processor, self._max_length = (
                load_pretrained_model(
                    pretrained,
                    None,
                    model_name,
                    device_map=self.device_map,
                    **llava_model_args,
                )
            )

        self._config = self._model.config
        self.model.eval()
        self.model.tie_weights()
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        assert (
            self.batch_size_per_gpu == 1
        ), "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."

        eval_logger.info(f"Using device: {self._device}")
        self.model.to(self._device)
        self._rank = 0
        self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model
        return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=batch_first, padding_value=padding_value
        )
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(
        self, string: str, left_truncate_len=None, add_special_tokens=None
    ) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        try:
            return self.tokenizer.decode(tokens)
        except:
            return self.tokenizer.decode([tokens])

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def load_video(self, video_path, max_frames_num):
        if type(video_path) == str:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        else:
            vr = VideoReader(video_path[0], ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(
            0, total_frame_num - 1, max_frames_num, dtype=int
        )
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        print(f"spare_frames: {spare_frames.shape}")
        return spare_frames  # (frames, height, width, channels)

    def output(self, requests: dict, gen_kwargs: dict):
        print("start generation")
        
        question_input = []

        visuals = requests["visuals"]
        context = requests["context"]
        task_type = requests["task_type"]
        
        print(f"################# requests ######################\n{requests}")
        print(f"################# gen_kwargs ######################\n{gen_kwargs}")

        if task_type == "text":
            image_tensor = None

        # encode, pad, and truncate contexts for this batch
        elif task_type == "image":  # For image task
            image_tensor = process_images(visuals, self._image_processor, self._config)
            if type(image_tensor) is list:
                image_tensor = [
                    _image.to(dtype=torch.float16, device=self.device)
                    for _image in image_tensor
                ]
            else:
                image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)

        elif task_type == "video":  # For video task
            image_tensor = []
            max_frames = gen_kwargs.get("sample_frames", self.max_frames_num)
            if "sample_frames" in gen_kwargs:
                gen_kwargs.pop("sample_frames")

            try:
                if self.video_decode_backend == "decord":
                    frames = self.load_video(visuals, max_frames)
                elif self.video_decode_backend == "opencv": # read from camera
                    video_data = []
                    cap = cv2.VideoCapture('http://localhost:1234')
                    for i in range(8):
                        ret, frame = cap.read()
                        if not ret:
                            raise ValueError(f"video error at localhost:1234")
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        video_data.append(frame)
                    cap.release()
                    cv2.destroyAllWindows()
                    frames = np.stack(video_data, axis=0) # (T, H, W, C)

                frames = (
                    self._image_processor.preprocess(frames, return_tensors="pt")[
                        "pixel_values"
                    ]
                    # .half()
                    .bfloat16()
                    .cuda()
                )
                image_tensor.append(frames)
            except Exception as e:
                eval_logger.error(f"Error {e} in loading video")
                image_tensor = None

            task_type = "video"

        # if image_tensor is not None and len(image_tensor) != 0:
        #     assert DEFAULT_IMAGE_TOKEN not in context and DEFAULT_SPEECH_TOKEN not in context
        question = context
        
        # This is much safer for llama3, as we now have some object type in it
        if "llama_3" in self.conv_template:
            conv = copy.deepcopy(conv_templates[self.conv_template])
        else:
            conv = conv_templates[self.conv_template].copy()

        for prev_conv in requests["prev_conv"]:
            conv.append_message(conv.roles[0], prev_conv[0])
            conv.append_message(conv.roles[1], prev_conv[1])

        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN)
        conv.append_message(conv.roles[0], DEFAULT_SPEECH_TOKEN)
        conv.append_message(conv.roles[1], None)

        prompt_question = conv.get_prompt()
        question_input.append(prompt_question)

        # preconfigure gen_kwargs with defaults
        if "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 1024
        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = 0
        if "do_sample" not in gen_kwargs:
            gen_kwargs["do_sample"] = False
        if "top_p" not in gen_kwargs:
            gen_kwargs["top_p"] = None
        if "num_beams" not in gen_kwargs:
            gen_kwargs["num_beams"] = 1

        input_ids_list = [
            # tokenizer_image_token(
            #     prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            # )
            tokenizer_image_speech_tokens(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, SPEECH_TOKEN_INDEX, return_tensors="pt"
            )
            for prompt in question_input
        ]
        pad_token_ids = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )
        input_ids = self.pad_sequence(
            input_ids_list, batch_first=True, padding_value=pad_token_ids
        ).to(self.device)
        attention_masks = input_ids.ne(pad_token_ids).to(self.device)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )
        
        # process speech for input question
        if requests["question_audio"] is not None:
            audio_path = requests["question_audio"]
        else:
            audio_path = "./local_demo/wav/" + visuals[0].split("/")[-1]+".wav"
            if os.path.exists(audio_path): os.remove(audio_path) # refresh
            if not os.path.exists(audio_path):
                wav = chat.infer(clean_text(context))
                try:
                    torchaudio.save(audio_path, torch.from_numpy(wav).unsqueeze(0), 24000)
                except:
                    torchaudio.save(audio_path, torch.from_numpy(wav), 24000)
        speech = whisper.load_audio(audio_path)
        speech = whisper.pad_or_trim(speech)
        speech = whisper.log_mel_spectrogram(speech, n_mels=128).permute(1, 0).to(device=self.model.device, dtype=torch.float16)
        speech_length = torch.LongTensor([speech.shape[0]]).to(self.model.device)
        
        # print("##"*20)
        # print(stop_str)
        # print(stopping_criteria)
        # print("##"*20)
        
        if task_type == "image":
            gen_kwargs["image_sizes"] = [
                visual.size for visual in visuals
            ]  # (width, height)
        elif task_type == "video":
            gen_kwargs["modalities"] = ["video"]
            gen_kwargs["stopping_criteria"] = [stopping_criteria]
            self._config.mm_spatial_pool_stride = self.mm_spatial_pool_stride
            self._config.mm_spatial_pool_mode = self.mm_spatial_pool_mode

        # These steps are not in LLaVA's original code, but are necessary for generation to work
        # TODO: attention to this major generation step...
        if "image_aspect_ratio" in gen_kwargs.keys():
            gen_kwargs.pop("image_aspect_ratio")

        max_context_length = getattr(self.model.config, "max_position_embeddings", 2048)
        # num_image_tokens = (
        #     question.count(DEFAULT_IMAGE_TOKEN)
        #     * self.model.get_vision_tower().num_patches
        # )
        num_image_tokens = self.model.get_vision_tower().num_patches

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15
        )

        gen_kwargs["max_new_tokens"] = min(
            gen_kwargs["max_new_tokens"],
            max_context_length - input_ids.shape[-1] - num_image_tokens,
        )

        if gen_kwargs["max_new_tokens"] < 1:
            print("yield error")
            # yield json.dumps(
            #     {
            #         "text": question
            #         + "Exceeds max token length. Please start a new conversation, thanks.",
            #         "error_code": 0,
            #     }
            # ).encode() + b"\0"
            return

        with torch.inference_mode():
            streamer = TextStreamer(self.tokenizer)
            # new_query = "Can you describe the video?"
            # new_query = "Oh. This video is about cooking."
            # new_query = "Sorry to interrupt."
            # new_query = "How to make a cup of wine?"
            
            new_query = requests["new_query"]
            new_query_pos = requests["new_query_pos"]
            
            conv = conv_templates[self.conv_template].copy()
            # conv.append_message(conv.roles[0], new_query)
            conv.append_message(conv.roles[0], DEFAULT_SPEECH_TOKEN)
            conv.append_message(conv.roles[1], None)
            new_prompt_question = [conv.get_prompt()]
            new_input_ids_list = [
                # tokenizer_image_token(
                #     new_query_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                # )
                tokenizer_image_speech_tokens(
                    new_query_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, SPEECH_TOKEN_INDEX, return_tensors="pt"
                )
                for new_query_prompt in new_prompt_question
            ]
            new_input_ids = self.pad_sequence(
                new_input_ids_list, batch_first=True, padding_value=pad_token_ids
            ).to(self.device)
            
            # process speech of new input query
            if requests["new_query_audio"] is not None:
                audio_path = requests["new_query_audio"]
            else:
                audio_path = "./local_demo/wav/new_" + visuals[0].split("/")[-1]+".wav"
                if os.path.exists(audio_path): os.remove(audio_path) # refresh
                if not os.path.exists(audio_path):
                    wav = chat.infer(clean_text(new_query))
                    try:
                        torchaudio.save(audio_path, torch.from_numpy(wav).unsqueeze(0), 24000)
                    except:
                        torchaudio.save(audio_path, torch.from_numpy(wav), 24000)
            new_speech = whisper.load_audio(audio_path)
            new_speech = whisper.pad_or_trim(new_speech)
            new_speech = whisper.log_mel_spectrogram(new_speech, n_mels=128).permute(1, 0).to(device=self.model.device, dtype=torch.float16)
            new_speech_length = torch.LongTensor([new_speech.shape[0]]).to(self.model.device)
            
            cont = self.model.generate_parallel(
                input_ids,
                attention_mask=attention_masks,
                pad_token_id=pad_token_ids,
                images=image_tensor,
                speeches=speech.unsqueeze(0),
                speech_lengths=speech_length,
                use_cache=self.use_cache,
                new_query=new_input_ids,
                new_query_str=new_query,
                new_query_pos=new_query_pos,
                new_speeches=new_speech.unsqueeze(0),
                new_speech_lengths=new_speech_length,
                query_str=question,
                tokenizer=self.tokenizer,
                **gen_kwargs,
            )

        print(cont)
        
        outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
        
        return outputs

def image_demo(model, args):
    visual_path = args.image_path
    image = Image.open(visual_path).convert("RGB")
    
    input_visuals = [image]
    input_context = args.question
    task_type = "image"
    gen_kwargs = {"max_new_tokens": 1024, "temperature": 0, "do_sample": False}
    query = {
        "visuals": input_visuals,
        "context": input_context,
        "task_type": task_type,
        "prev_conv": [],
    }
    try:
        prev = 0
        for x in model.stream_generate_until(query, gen_kwargs):
            output = json.loads(x.decode("utf-8").strip("\0"))["text"].strip()
            print(output[prev:], end="", flush=True)
            prev = len(output)

        print("\n")
    except Exception as e:
        print(e)

def video_demo(model, args):
    visual_path = args.video_path
    input_visuals = [visual_path]
    input_context = args.question
    task_type = "video"
    gen_kwargs = {"max_new_tokens": 1024, "temperature": 0, "do_sample": False, "sample_frames": args.num_sampled_frames}
    query = {
        "visuals": input_visuals,
        "context": input_context,
        "task_type": task_type,
        "prev_conv": [],
    }
    prev = 0
    
    print("start...")
    outputs = model.stream_generate_until(query, gen_kwargs)
    # print(outputs)
    
    print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default=None, required=True)
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--num_sampled_frames", type=int, default=32)
    parser.add_argument("--question", type=str)
    parser.add_argument("--question_audio", type=str)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--new_query", type=str)
    parser.add_argument("--new_query_audio", type=str)
    parser.add_argument("--new_query_pos", type=int, default=20)
    args = parser.parse_args()
    
    model = OpenOmni(pretrained="checkpoints/M4-Audio-LongVA-7B-Qwen2", model_name="llava_qwen", device_map=args.device, attn_implementation="eager")
    
    visual_path = args.video_path
    input_visuals = [visual_path]
    input_context = args.question
    task_type = "video"
    gen_kwargs = {"max_new_tokens": 1024, "temperature": 0, "do_sample": False, "sample_frames": args.num_sampled_frames}
    query = {
        "visuals": input_visuals,
        "context": input_context,
        "task_type": task_type,
        "prev_conv": [],
        "new_query": args.new_query,
        "new_query_pos": args.new_query_pos,
        "question_audio": args.question_audio,
        "new_query_audio": args.new_query_audio
    }
    prev = 0
    
    # print("start...")
    outputs = model.output(query, gen_kwargs)
    # print(outputs)

