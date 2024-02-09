import torch
import sentencepiece as spm
from models import TextCNN

tokenizer = spm.SentencePieceProcessor(model_file="tokenizer.model")
model = TextCNN()
model.load_state_dict(torch.load("checkpoint_14.pth"))
model.eval()
idx2label = {
    0: "អាហារតាមផ្លូវ",
    1: "អាហារល្បីតាមតំបន់",
    2: "អាហារពេលព្រឹក",
    3: "អាហារពេលយប់",
    4: "អាហារសម្រន់",
    5: "កាហ្វេ តែ នំ នំប៉័ង",
    6: "អាហារពេលថ្ងៃត្រង់",
    7: "អាហារពេលល្ងាច",
    8: "ស៊ុប/សាច់អាំង",
    9: "កាហ្វេ Coffee",
}

with torch.no_grad():
    text = "ភោជនីយដ្ឋាន ម៉ាលីន"
    input_ids = tokenizer.encode(text.lower().replace("\u200b", ""))
    input_ids = input_ids + [3] * (128 - len(input_ids))
    input_ids = torch.LongTensor(input_ids).unsqueeze(0)
    logits = model(input_ids)
    preds = torch.argmax(logits, dim=1).flatten()
    label_id = preds.item()
    
    print(f"{text} => {idx2label[label_id]}")
