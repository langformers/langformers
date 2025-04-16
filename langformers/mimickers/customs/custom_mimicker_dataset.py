from torch.utils.data import Dataset
from tqdm import tqdm


class CustomMimickerDataset(Dataset):
    def __init__(self, dataset_path, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

        if isinstance(dataset_path, str):
            with open(dataset_path, "r", encoding="utf-8") as f:
                self.texts = [text for text in tqdm(f.readlines(), desc="Processing data") if isinstance(text, str) and len(text.strip()) > 0]
        else:
            self.texts = [text for text in tqdm(dataset_path, desc="Processing data") if isinstance(text, str) and len(text.strip()) > 0]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx].strip() if isinstance(self.texts[idx], str) else self.texts[idx]
        encoding = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True,
                                  max_length=self.max_length, add_special_tokens=True)
        return {key: val.squeeze(0) for key, val in encoding.items()}
    