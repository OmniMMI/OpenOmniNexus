import json
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("../../checkpoints/Qwen2-7B-Instruct")

rates = []

vas = json.load(open("voiceassistant_units.json"))
for va in vas:
    conv = va["conversations"][-1]
    tokens = tokenizer.tokenize(conv["value"])
    len_token = len(tokens)
    len_units = len(conv["tgt_units"])
    rates.append(len_units / len_token)
    
print("MAX: ", max(rates))
print("MIN: ", min(rates))
print("MEAN: ", sum(rates)/len(rates))
    


