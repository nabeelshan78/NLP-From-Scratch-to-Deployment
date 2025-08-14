import torch
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from typing import List, Iterator

# ==============================================================================
#                               SETUP
# ==============================================================================

# global settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
# seed for reproducibility
RANDOM_SEED = 42

# tokenizer
tokenizer = get_tokenizer('basic_english')

# ==============================================================================
#                       DATASET CLASS DEFINITION
# ==============================================================================

class NewsDataset(Dataset):
    """
    Custom Dataset for news text classification.
    Sorts the data by text length to enable more efficient batching.
    """
    def __init__(self, df, sort_by_len=False):
        # Extract texts and labels from the DataFrame
        texts = df.iloc[:, 0].tolist()
        labels = df.iloc[:, 1].tolist()

        if sort_by_len:
            print("Sorting data by text length to reduce padding...")
            # Create a list of tuples (text, label, length)
            data_with_len = [(text, label, len(tokenizer(text))) for text, label in zip(texts, labels)]
            # Sort the list based on the length
            data_with_len.sort(key=lambda x: x[2])
            # Unpack the sorted data
            texts = [item[0] for item in data_with_len]
            labels = [item[1] for item in data_with_len]

        self.texts = list(texts)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.labels[idx], self.texts[idx]


def get_dataloaders(batch_size=64):
    """
    Loads data, builds vocab, and returns reproducible DataLoaders.

    1. Reproducible Splits: Uses a fixed random seed with torch.Generator
       to ensure the train/validation split is identical every time.
    2. Efficient Batching: The training DataLoader has shuffle=False because the
       dataset is pre-sorted by length. This minimizes padding.
    """
    # --- Set Seed for Reproducibility ---
    torch.manual_seed(RANDOM_SEED)

    # --- Load Raw Data ---
    print("Loading data from Parquet files...")
    
    if not os.path.exists("ag_news/train.parquet"):
        raise FileNotFoundError("Please download the ag_news dataset first.")
    train_df = pd.read_parquet("ag_news/train.parquet")
    test_df = pd.read_parquet("ag_news/test.parquet")

    # --- Vocabulary Setup (Load or Build) ---
    VOCAB_PATH = 'vocab.pth'
    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)

    if os.path.exists(VOCAB_PATH):
        print("Loading existing vocabulary...")
        vocab = torch.load(VOCAB_PATH)
    else:
        print("Building vocabulary from training data...")
        # Use a temporary dataset just for building the vocab
        temp_train_dataset = NewsDataset(train_df)
        vocab = build_vocab_from_iterator(yield_tokens(temp_train_dataset), specials=["<unk>"])
        vocab.set_default_index(vocab["<unk>"])
        torch.save(vocab, VOCAB_PATH)
        print(f"Vocabulary built and saved to {VOCAB_PATH}")
    
    print(f"Vocabulary Size: {len(vocab)}")

    # --- Define Pipelines and Collate Function ---
    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: int(x)

    def collate_batch(batch):
        label_list, text_list, len_list = [], [], []
        for (_label, _text) in batch:
            label_list.append(label_pipeline(_label))
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
        
        labels = torch.tensor(label_list, dtype=torch.int64)
        # pad_sequence creates a batch of sequences padded to the same length
        texts = pad_sequence(text_list, batch_first=True)
        return labels.to(DEVICE), texts.to(DEVICE)

    # --- Create Datasets and Split (Reproducibly) ---
    # The data is sorted by length to make padding more efficient.
    train_dataset_full = NewsDataset(train_df, sort_by_len=True)
    test_dataset = NewsDataset(test_df, sort_by_len=True)
    
    num_train = int(len(train_dataset_full) * 0.92)
    split_lengths = [num_train, len(train_dataset_full) - num_train]
    
    print(f"Splitting training data into {split_lengths[0]} training and {split_lengths[1]} validation samples.")
    
    # Use a generator with a fixed seed for a reproducible split
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    split_train, split_valid = random_split(train_dataset_full, split_lengths, generator=generator)
    
    # --- Create and Return DataLoaders ---
    # shuffle=False for the train_dataloader is crucial. Since the data is already
    # sorted by length, shuffling here would undo the benefit of reduced padding.
    train_dataloader = DataLoader(split_train, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    valid_dataloader = DataLoader(split_valid, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    
    return train_dataloader, valid_dataloader, test_dataloader, vocab
