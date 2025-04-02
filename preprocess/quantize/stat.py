import json
import statistics
from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("../../checkpoints/Qwen2-7B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("/data/prev_trained_models//Qwen2-7B-Instruct")

rates = []
len_units = []
len_tokens = []

vas = json.load(open("VoiceAssistant_units_gt.json"))
vas = json.load(open("VoiceAssistant_units_pred.json"))
vas = json.load(open("VoiceAssistant_units_pred_unmerged.json"))
vas = json.load(open("VoiceAssistant_units_gt_270K.json"))
for va in vas:
    conv = va["conversations"][-1]
    tokens = tokenizer.tokenize(conv["value"])
    len_token = len(tokens)
    len_unit = len(conv["tgt_units"])
    len_tokens.append(len_token)
    len_units.append(len_unit)
    rates.append(len_unit / len_token)

rates = sorted(rates, reverse=True)
print(rates[:1000][-100:])
print("MAX: ", max(rates))
print("MIN: ", min(rates))
print("MEAN: ", sum(rates)/len(rates))
print("MEDIAN: ", statistics.median(rates))

print("mean units: ", sum(len_units)/len(len_units))
print("mean tokens: ", sum(len_tokens)/len(len_tokens))
    


