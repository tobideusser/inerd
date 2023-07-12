from typing import List

from torch.utils.data import Dataset

from inerd.data_classes import Sentence


class GenerativeNERDataset(Dataset):
    def __init__(self, sentences: List[Sentence]):
        super().__init__()
        self.sentences = sentences

    def __getitem__(self, index) -> Sentence:
        return self.sentences[index]

    def __len__(self) -> int:
        return len(self.sentences)
