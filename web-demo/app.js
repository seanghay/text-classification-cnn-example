import * as ort from 'onnxruntime-web';
import { sentencePieceProcessor, cleanText } from '@weblab-notebook/sentencepiece';

const spm = await sentencePieceProcessor("/tokenizer.model")
spm.loadVocabulary("/tokenizer.vocab")

/**
 * @param {string} text 
 */
function tokenize(text) {
  const max_length = 128
  const pad_id = 3
  text = text.replace(/\u200b/, "").toLowerCase().trim();
  const ids = spm.encodeIds(cleanText(text));
  const padding = Array.from({ length: max_length - ids.length }).fill(pad_id);
  const input_ids = [...ids, ...padding];
  return input_ids
}

function argmax(array) {
  return array.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
}

const idx2label = [
  "អាហារតាមផ្លូវ",
  "អាហារល្បីតាមតំបន់",
  "អាហារពេលព្រឹក",
  "អាហារពេលយប់",
  "អាហារសម្រន់",
  "កាហ្វេ តែ នំ នំប៉័ង",
  "អាហារពេលថ្ងៃត្រង់",
  "អាហារពេលល្ងាច",
  "ស៊ុប / សាច់អាំង",
  "កាហ្វេ",
];

const session = await ort.InferenceSession.create("/model.onnx")


const input_el = document.querySelector("#text_input")
const tokens_el = document.querySelector("#tokens")
input_el.disabled = false
input_el.placeholder = "Input"

input_el.addEventListener("input", async e => {
  if (!e.target.value) return;
  const input_ids = tokenize(e.target.value)
  const result = await session.run({
    "input": new ort.Tensor('int64', input_ids.map(i => BigInt(i)), [1, input_ids.length])
  });

  const idx = argmax(Array.from(result["output"].data))
  tokens_el.value = idx2label[idx]
})