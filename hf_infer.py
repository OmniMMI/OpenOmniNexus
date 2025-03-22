
import os
import json
from PIL import Image
import numpy as np
import torchaudio
import torch
from decord import VideoReader, cpu
import whisper
import soundfile as sf
# fix seed
torch.manual_seed(0)

from fairseq import utils as fairseq_utils
from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder

from open_gpt4o.model.builder import load_pretrained_model
from open_gpt4o.mm_utils import tokenizer_image_speech_tokens, process_images, ctc_postprocess
from open_gpt4o.constants import IMAGE_TOKEN_INDEX, SPEECH_TOKEN_INDEX

import warnings
warnings.filterwarnings("ignore")

# config OpenGPT4o
model_path = "checkpoints/OpenGPT4o-7B-Qwen2"
video_path = "local_demo/assets/water.mp4"
audio_path = "local_demo/wav/infer.wav"
audio_path = "local_demo/wav/water.mp4.wav"
max_frames_num = 16 # you can change this to several thousands so long you GPU memory can handle it :)
gen_kwargs = {"do_sample": True, "temperature": 0.5, "top_p": None, "num_beams": 1, "use_cache": True, "max_new_tokens": 1024}
tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, "llava_s2s_qwen", device_map="cuda:0")

# config vocoder
with open("checkpoints/vocoder/config.json") as f:
    vocoder_cfg = json.load(f)
vocoder = CodeHiFiGANVocoder("checkpoints/vocoder/g_00500000", vocoder_cfg).cuda()

# query input
query = "Give a detailed caption of the video as if I am blind."
query = None # comment this to use ChatTTS to convert the query to audio

#video input
prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image><|im_end|>\n<|im_start|>user\n<speech>\n<|im_end|>\n<|im_start|>assistant\n"
input_ids = tokenizer_image_speech_tokens(prompt, tokenizer, IMAGE_TOKEN_INDEX, SPEECH_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
vr = VideoReader(video_path, ctx=cpu(0))
total_frame_num = len(vr)
uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
frame_idx = uniform_sampled_frames.tolist()
frames = vr.get_batch(frame_idx).asnumpy()
video_tensor = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(model.device, dtype=torch.float16)

#audio input
# process speech for input question
if query is not None:
    import ChatTTS
    chat = ChatTTS.Chat()
    chat.load(source='local', compile=True)
    audio_path = "./local_demo/wav/" + "infer.wav"
    if os.path.exists(audio_path): os.remove(audio_path) # refresh
    if not os.path.exists(audio_path):
        wav = chat.infer(query)
        try:
            torchaudio.save(audio_path, torch.from_numpy(wav).unsqueeze(0), 24000)
        except:
            torchaudio.save(audio_path, torch.from_numpy(wav), 24000)
    print(f"Human: {query}")
    
else:
    print("Human: <audio>")
    
speech = whisper.load_audio(audio_path)
speech = whisper.pad_or_trim(speech)
speech = whisper.log_mel_spectrogram(speech, n_mels=128).permute(1, 0).to(device=model.device, dtype=torch.float16)
speech_length = torch.LongTensor([speech.shape[0]]).to(model.device)

with torch.inference_mode():
    output_ids, output_units = model.generate(input_ids, images=[video_tensor],  modalities=["video"], speeches=speech.unsqueeze(0), speech_lengths=speech_length, **gen_kwargs)
outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
print(f"Agent: {outputs}")

output_units = ctc_postprocess(output_units, blank=model.config.unit_vocab_size)
output_units = [(list(map(int, output_units.strip().split())))]
print(f"Units: {output_units}")
x = {"code": torch.LongTensor(output_units[0]).view(1,-1)}
x = fairseq_utils.move_to_cuda(x)
wav = vocoder(x, True)
output_file_path = "local_demo/wav/output.wav"
sf.write(
    output_file_path,
    wav.detach().cpu().numpy(),
    16000
)
print(f"The generated wav saved to {output_file_path}")

