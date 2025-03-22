# This code is from https://github.com/facebookresearch/fairseq/blob/main/examples/textless_nlp/gslm/speech2unit/clustering/quantize_with_kmeans.py

import os
import argparse
from tqdm import tqdm
import joblib
import json
import librosa
import soundfile as sf

import torch

from examples.textless_nlp.gslm.speech2unit.pretrained.hubert_feature_reader import (
    HubertFeatureReader,
)

# from util import save_unit

def save_unit(unit, unit_path):
    os.makedirs(os.path.dirname(unit_path), exist_ok=True)
    with open(unit_path, "w") as f:
        f.write(unit)

def load_model(model_path, kmeans_path, use_cuda=False):
    hubert_reader = HubertFeatureReader(
        checkpoint_path=model_path,
        layer=11,
        use_cuda=use_cuda,
    )
    kmeans_model = joblib.load(open(kmeans_path, "rb"))
    kmeans_model.verbose = False

    return hubert_reader, kmeans_model

def s2u(items, hubert_reader, kmeans_model):
    
    new_items = []
    
    for item in tqdm(items, total=len(items)):
        idx = item["id"]
        if not os.path.exists(f"../../inputs/speech/voiceassistant_response/{idx}_0.wav"): continue
        count = 0
        for i, convo in enumerate(item["conversations"]):
            if convo.get("from") == "gpt":
                audio_filename = f"../../inputs/speech/voiceassistant_response/{idx}_{count}.wav"
                # resample
                tgt_sr = 16000
                y, orig_sr = librosa.load(audio_filename, sr=None, mono=True)
                if orig_sr != tgt_sr:
                    y_resampled = librosa.resample(y, orig_sr=orig_sr, target_sr=tgt_sr)
                    audio_filename = f"../../inputs/speech/voiceassistant_response/{idx}_{count}_16k.wav"
                    sf.write(audio_filename, y_resampled, tgt_sr)
                
                
                feats = hubert_reader.get_feats(audio_filename)
                feats = feats.cpu().numpy()
                pred = kmeans_model.predict(feats)
                
                convo["tgt_units"] = pred.tolist()
                # break
                # print(pred)
        # break
        new_items.append(item)
                
    return new_items

def main():
    
    items_path = "../../inputs/text/voiceassistant.json"
    mhubert_path = "../../checkpoints/quantizer/mhubert_base_vp_en_es_fr_it3.pt"
    kmeans_path = "../../checkpoints/quantizer/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin"
    
    use_cuda = torch.cuda.is_available()

    hubert_reader, kmeans_model = load_model(mhubert_path, kmeans_path, use_cuda=use_cuda)
    
    items = json.load(open(items_path))
    
    new_items = s2u(items, hubert_reader, kmeans_model)
    
    json.dump(new_items, open("voiceassistant_units.json", "w"))

if __name__ == "__main__":
    main()