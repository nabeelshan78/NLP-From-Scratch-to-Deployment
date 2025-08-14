import math
import torch
import torch.nn as nn
from tqdm import tqdm
import pickle


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, vocab_size=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(vocab_size, d_model)
        position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1) # [vocab_size, 1]
        div_term = torch.exp(             # [d_model/2]
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

        pe = pe.unsqueeze(0)    # [1, vocab_size, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):    # [batch_size, seq_len, d_model]
        # x shape: [batch_size, seq_len, d_model]
        # pe slice shape: [1, seq_len, d_model]
        # Broadcast 1 → batch_size
        # Resulting x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
        
        

class ClassificationNet(nn.Module):
    """
    Text classifier based on a pytorch TransformerEncoder.
    """
    def __init__(self, vocab_size, num_class):
        super().__init__()
        embedding_dim = 256
        nhead = 8
        dim_feedforward = 1024
        num_layers = 3
        dropout = 0.2

        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, vocab_size, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, num_class)
        )
        self.d_model = embedding_dim

    def forward(self, x):
        x = self.emb(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.classifier(x)

        return x
    
    

def train_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    """
    Trains the model for one epoch.
    """
    model.train()   # Set the model to training mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # Use tqdm on the dataloader for a batch-level progress bar
    progress_bar = tqdm(dataloader, desc=f"Training")

    for labels, texts in progress_bar:
        labels, texts = labels.to(device), texts.to(device)

        # Forward pass
        predicted_labels = model(texts)
        loss = criterion(predicted_labels, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1) # Gradient clipping
        optimizer.step()
        
        scheduler.step()

        # Update running loss and accuracy metrics
        running_loss += loss.item() * texts.size(0)
        _, preds = torch.max(predicted_labels, 1)
        correct_predictions += torch.sum(preds == labels)
        total_samples += texts.size(0)

        # Update the progress bar with the current average loss
        progress_bar.set_postfix(loss=f'{running_loss/total_samples:.4f}')

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples
    return epoch_loss, epoch_acc.item()


def evaluate_epoch(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc=f"Evaluating")
    
    with torch.no_grad():  # No need to track gradients during evaluation
        for labels, texts in progress_bar:
            labels, texts = labels.to(device), texts.to(device)

            predicted_labels = model(texts)
            loss = criterion(predicted_labels, labels)

            # Update metrics
            running_loss += loss.item() * texts.size(0)
            _, preds = torch.max(predicted_labels, 1)
            correct_predictions += torch.sum(preds == labels)
            total_samples += texts.size(0)

            # Update the progress bar
            progress_bar.set_postfix(loss=f'{running_loss/total_samples:.4f}')

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples
    return epoch_loss, epoch_acc.item()




def save_list_to_file(lst, filename):
    """
    Save a list to a file using pickle serialization.
    """
    with open(filename, 'wb') as file:
        pickle.dump(lst, file)

def load_list_from_file(filename):
    """
    Load a list from a file using pickle deserialization.
    """
    with open(filename, 'rb') as file:
        loaded_list = pickle.load(file)
    return loaded_list