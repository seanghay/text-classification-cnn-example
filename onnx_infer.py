import sentencepiece as spm
import numpy as np
import onnxruntime as rt

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


sess = rt.InferenceSession("./text_cnn.onnx")
tokenizer = spm.SentencePieceProcessor(model_file="tokenizer.model")
text = "បាយឆា"
input_ids = tokenizer.encode(text.lower().replace("\u200b", ""))
input_ids = input_ids + [3] * (128 - len(input_ids))
input_ids = np.expand_dims(input_ids, axis=0)

outputs = sess.run(
    None,
    {
        "input": input_ids,
    },
)

print(f"{text} => {idx2label[np.argmax(outputs)]}")
