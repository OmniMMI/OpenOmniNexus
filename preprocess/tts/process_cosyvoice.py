import json
import os
import torch
import torchaudio
from num2words import num2words
import re
from tqdm import tqdm
import argparse

# # optional: ChatTTS
# import ChatTTS
# chat = ChatTTS.Chat()
# chat.load(source="local", compile=True)
#chat.normalizer.register("en", normalizer_en_nemo_text())
#chat.normalizer.register("zh",normalizer_zh_tn())

from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

cosyvoice = CosyVoice("pretrained_models/CosyVoice-300M-25Hz")

voiceassistant = json.load(open("voiceassistant.json"))


import multiprocessing
from multiprocessing import Pool

import random
random.seed(43)

def convert_numbers_to_words(text):
    def replace(match):
        num = match.group(0)
        return num2words(num)
    return re.sub(r'\b\d+\b', replace, text)

def clean_text(text):
    text = text.replace("-", "to")
    text = text.replace("'", "\'")
    return text.replace("<image>\n", "").replace("\n", " ")


def text_to_speech(item_id, idx, text, output_dir="m4-it/"):
    os.makedirs(output_dir, exist_ok=True)
    audio_filename = f"{item_id}_{idx}.wav" 
    audio_path = os.path.join(output_dir, audio_filename)
    directory = os.path.dirname(audio_path)
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    if os.path.exists(audio_path):
        return audio_filename
    
    while 1:
        try:
            prompt = random.choice(voiceassistant)
            prompt_audio = prompt["speech"]
            prompt_text = prompt["question"]
            if not prompt_text.endswith("."):
                prompt_text += "."
            if len(text.split()) <= 0.5 * len(prompt_text.split()):
                raise ValueError("prompt text too long")
            prompt_audio = f"VoiceAssistant-400K/audios/{prompt_audio}"
            prompt_speech = load_wav(prompt_audio, 16000)
            
            if not text.endswith("."):
                text += '.'
            for i, j in enumerate(cosyvoice.inference_zero_shot(text, prompt_text=prompt_text, prompt_speech_16k=prompt_speech, stream=False)):
                torchaudio.save(audio_path, j["tts_speech"], cosyvoice.sample_rate)
            break
        except Exception as e:
            print(f"Encounter Error: {e}, resample the prompt from voiceassistant")
            
    
    return audio_filename

def process_conversation(items):
    speech_files = []
    for _, item in tqdm(enumerate(items), total=len(items)):
        count = 0
        for i, convo in enumerate(item["conversations"]): 
            if convo.get("from") == "human":
                clean_value = clean_text(convo["value"])
                clean_value = convert_numbers_to_words(clean_value)
                audio_filename = text_to_speech(item["id"], count, clean_value)
                count+=1
                speech_files.append(audio_filename)
        if len(speech_files) == 0:
            continue
        item["speech"] = speech_files
        torch.cuda.empty_cache()
    return item

def split_into_batches(data, num_batches):
    batch_size = len(data) // num_batches
    remainder = len(data) % num_batches

    batches = []
    start = 0
    for i in range(num_batches):
        end = start + batch_size + (1 if i < remainder else 0)
        batches.append(data[start:end])
        start = end

    return batches

def main():
    parser = argparse.ArgumentParser(description="Process data batches.")
    parser.add_argument('--batch_id', type=int, default=0, help="ID of the batch to process.")
    parser.add_argument('--num_batch', type=int, default=1, help="ID of the batch to process.")
    args = parser.parse_args()

    with open("m4-it-qwen.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    num_batches = args.num_batch
    batches = split_into_batches(data, num_batches)
    batch = batches[args.batch_id]
    new = process_conversation(batch)

if __name__ == "__main__":
    main()
