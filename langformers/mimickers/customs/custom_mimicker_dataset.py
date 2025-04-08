from torch.utils.data import Dataset


class CustomMimickerDataset(Dataset):
    def __init__(self, dataset_path, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

        if isinstance(dataset_path, str):
            with open(dataset_path, "r", encoding="utf-8") as f:
                self.texts = f.readlines()
        else:
            self.texts = dataset_path

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tweet = self.texts[idx].strip() if isinstance(self.texts[idx], str) else self.texts[idx]
        encoding = self.tokenizer(tweet, return_tensors="pt", padding="max_length", truncation=True,
                                  max_length=self.max_length)
        return {key: val.squeeze(0) for key, val in encoding.items()}

