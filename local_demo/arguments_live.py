from dataclasses import dataclass, field
from transformers import TrainingArguments
from transformers import HfArgumentParser

@dataclass
class LiveTrainingArguments(TrainingArguments):
    live_version: str = 'live1+'
    system_prompt: str = (
        "A multimodal AI assistant is helping users with some activities."
        " Below is their conversation, interleaved with the list of video frames received by the assistant."
    )
    train_datasets: list[str] = None
    eval_datasets: list[str] = None
    stream_loss_weight: float = 1.0
    llm_pretrained: str = 'Qwen2-7B-Instruct'
    vision_pretrained: str = 'clip-vit-large-patch14-336'
    lora_modules: str = "model.*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|lm_head$"
    lora_r: int = 128
    lora_alpha: int = 256
    finetune_modules: list[str] = field(default_factory=lambda: ['connector'])
    frame_fps: int = 1 # for training. inference can be 10
    frame_token_cls: bool = None
    frame_token_pooled: list[int] = None
    frame_resolution: int = 336
    frame_token_interval: str  = None
    frame_token_interval_threshold: float = 0.0
    augmentation: bool = False
    # attn_implementation: str = 'flash_attention_2'
    attn_implementation: str = 'eager' # nan from sdpa
    output_dir: str = 'outputs/debug'
    
    # OpenOmni parameters
    model_path: str = "ColorfulAI/OpenOmni"
    model_base: str = None
    video_file: str = None
    # device: str = "cuda"
    conv_mode: str = "qwen_1_5"
    num_frames: float = 8
    temperature: float = 0.2
    max_new_tokens: int = 512
    load_8bit: bool = None
    load_4bit: bool = None
    
    # inference args
    model_name: str = ''
    benchmark_name: str = ''
    cache_dir: str = ''
    video_dir: str = ''
    questions_file: str = ''
    num_chunks: int = 1
    chunk_idx: int = 0
    output_dir: str = ''
    

@dataclass
class LiveOneTrainingArguments(LiveTrainingArguments):
    live_version: str = 'live1'
    frame_token_cls: bool = True
    frame_num_tokens: int = 1
    frame_token_interval: str  = ''
    embed_mark: str = '2fps_384_1'
    max_num_frames: int = 7200 # 1h, 2fps, 7200 frames

@dataclass
class LiveOnePlusTrainingArguments(LiveTrainingArguments):
    live_version: str = 'live1+'
    frame_token_cls: bool = True
    frame_token_pooled: list[int] = field(default_factory=lambda: [3,3])
    frame_num_tokens: int = 10 # 1+3x3
    embed_mark: str = '2fps_384_1+3x3'
    frame_token_interval: str = ','
    max_num_frames: int = 1200 # 10min, 2fps, 1200 frames

def get_args_class(live_version: str):
    if live_version == 'live1':
        return LiveOneTrainingArguments
    elif live_version == 'live1+':
        return LiveOnePlusTrainingArguments
    raise NotImplementedError

def parse_args() -> LiveTrainingArguments:
    args, = HfArgumentParser(LiveTrainingArguments).parse_args_into_dataclasses()
    args, = HfArgumentParser(get_args_class(args.live_version)).parse_args_into_dataclasses()
    return args