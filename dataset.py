import torch
import sentencepiece as spm
import pandas as pd
from torch.utils.data import Dataset

class FoodDataset(Dataset):
    def __init__(
        self,
        file_path,
        model_file,
        max_text_len=128,
    ) -> None:
        super().__init__()
        self.tokenizer = spm.SentencePieceProcessor(model_file=model_file)
        self.max_text_len = max_text_len
        self._pad_idx = 3
        
        df = pd.read_csv(file_path, delimiter="\t")
        df = df.drop("label_name", axis=1)
        self.items = df.values.tolist()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        text, label = self.items[index]
        input_ids = self.tokenizer.encode(text)
        input_ids = input_ids + [self._pad_idx] * (self.max_text_len - len(input_ids))  
        return torch.LongTensor(input_ids), label

if __name__ == "__main__":
    dataset = FoodDataset("./data/result.tsv", model_file="tokenizer.model")
    print(len(dataset))
    print(dataset[0])
    
    
