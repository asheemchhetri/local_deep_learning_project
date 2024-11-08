import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ClaimsDataset(Dataset):
    """
        A custom PyTorch Dataset class for processing insurance claims data.

        Attributes:
            tokenizer (PreTrainedTokenizer): The tokenizer used for encoding text data.
            max_length (int): The maximum sequence length for text encoding.
            texts (pd.Series): The concatenated text data (diagnosis description and provider notes) for each claim.
            severity (pd.Series): The severity of the claim mapped to numeric values.
            risk_scores (pd.Series): The normalized risk scores for the claims.
            labels (pd.Series): The claim status mapped to numeric values.
    """
    def __init__(self, data, tokenizer, max_length=384):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = data['diagnosis_description'] + " [SEP] " + data['provider_notes']
        self.severity = data['severity'].map({'mild': 0, 'moderate': 1, 'severe': 2})
        self.risk_scores = data['risk_score'] / 100.0
        self.labels = data['claim_status'].map({'Approved': 0, 'Denied': 1, 'Pending': 2})

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts.iloc[idx]),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'severity': torch.tensor(self.severity.iloc[idx], dtype=torch.long),
            'risk_score': torch.tensor(self.risk_scores.iloc[idx], dtype=torch.float),
            'label': torch.tensor(self.labels.iloc[idx], dtype=torch.long)
        }


class BertClaimClassifier(nn.Module):
    """
    BertClaimClassifier

    A PyTorch neural network module designed to classify insurance claims based on textual content, severity, and risk score.
    The model utilizes a pre-trained BERT model for text embeddings and additional embeddings for severity and risk scores.

    Attributes:
        bert (BertModel): An instance of a pre-trained BERT model.
        severity_embedding (nn.Embedding): An embedding layer for severity input.
        risk_score_fc (nn.Sequential): A fully connected network for risk score input.
        classifier (nn.Sequential): A fully connected classification network.

    Methods:
        __init__(dropout=0.1):
            Initializes the BertClaimClassifier with a specified dropout rate.
            Unfreezes all layers of the BERT model.
            Sets up embedding layers for severity and risk scores.
            Configures the classifier network with specified dimensions and activation functions.

        forward(input_ids, attention_mask, severity, risk_score):
            Forward pass of the network.
            Processes the input textual data through the BERT model to obtain features.
            Embeds the severity and applies a fully connected network to the risk score.
            Concatenates the features from BERT, severity embedding, and risk score embedding.
            Passes the combined features through the classification network to get the final output.
    """
    def __init__(self, dropout=0.1):
        super(BertClaimClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Unfreeze all layers
        for param in self.bert.parameters():
            param.requires_grad = True

        self.severity_embedding = nn.Embedding(3, 16)
        self.risk_score_fc = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        combined_dim = self.bert.config.hidden_size + 16 + 16

        # self.classifier = nn.Sequential(
        #     nn.Dropout(dropout),
        #     nn.Linear(combined_dim, 512),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(512, 3)
        # )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 3)
        )

    def forward(self, input_ids, attention_mask, severity, risk_score):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use CLS token hidden state
        bert_features = bert_output.last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)

        severity_emb = self.severity_embedding(severity)
        risk_score_emb = self.risk_score_fc(risk_score.unsqueeze(1))

        combined = torch.cat([bert_features, severity_emb, risk_score_emb], dim=1)
        return self.classifier(combined)


def train(params):
    """
    :param params: Dictionary containing hyperparameters and configuration for training the model. Must include keys such as 'max_length', 'batch_size', 'dropout', 'bert_learning_rate', 'learning_rate', 'num_epochs', 'warmup_ratio', and 'early_stopping_patience'.
    :return: A tuple containing the trained model and the best validation accuracy achieved during training.
    """
    # Load data
    print("Loading data...")
    data = pd.read_csv('data/medical_claims_synthetic.csv')
    print(f"Total dataset size: {len(data):,} rows")

    # Check class distribution
    print("\nClass Distribution:")
    print(data['claim_status'].value_counts(normalize=True))

    # Split data
    train_data, temp = train_test_split(data, test_size=0.2, random_state=42, stratify=data['claim_status'])
    val_data, test_data = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp['claim_status'])
    print(f"\nTraining set: {len(train_data):,} rows")
    print(f"Validation set: {len(val_data):,} rows")
    print(f"Test set: {len(test_data):,} rows")

    # Save test data for later predictions
    test_data.to_csv('data/test_data.csv', index=False)

    # Create datasets
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = ClaimsDataset(train_data, tokenizer, max_length=params['max_length'])
    val_dataset = ClaimsDataset(val_data, tokenizer, max_length=params['max_length'])

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'])

    # Initialize model
    model = BertClaimClassifier(dropout=params['dropout']).to(device)

    # Calculate class weights for balanced loss
    class_counts = train_data['claim_status'].value_counts().sort_index()
    total = len(train_data)
    class_weights = torch.tensor(
        [total / class_counts[i] for i in ['Approved', 'Denied', 'Pending']],
        dtype=torch.float
    ).to(device)

    # Initialize training components
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW([
        {'params': model.bert.parameters(), 'lr': params['bert_learning_rate']},
        {'params': model.severity_embedding.parameters(), 'lr': params['learning_rate']},
        {'params': model.risk_score_fc.parameters(), 'lr': params['learning_rate']},
        {'params': model.classifier.parameters(), 'lr': params['learning_rate']}
    ])

    # Setup scheduler with warmup
    num_training_steps = len(train_loader) * params['num_epochs']
    num_warmup_steps = int(num_training_steps * params['warmup_ratio'])

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # Training loop
    best_val_accuracy = 0
    patience_counter = 0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    print("\nStarting training...")
    for epoch in range(params['num_epochs']):
        # Training phase
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            severity = batch['severity'].to(device)
            risk_score = batch['risk_score'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, severity, risk_score)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            train_loss += loss.item()

            if (batch_idx + 1) % 500 == 0:
                print(f'Epoch {epoch + 1}, Batch {batch_idx + 1}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Accuracy: {100 * correct / total:.2f}%')

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                severity = batch['severity'].to(device)
                risk_score = batch['risk_score'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask, severity, risk_score)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        print(f'\nEpoch {epoch + 1}:')
        print(f'Training Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.2f}%')
        print(f'Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')
        print('-' * 60)

        # Early stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'article_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= params['early_stopping_patience']:
                print("Early stopping triggered!")
                break

    # Plot training and validation loss
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Training Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label='Training Accuracy')
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Evaluate on validation set
    print("\nValidation Set Evaluation:")
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            severity = batch['severity'].to(device)
            risk_score = batch['risk_score'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask, severity, risk_score)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(classification_report(all_labels, all_preds, target_names=['Approved', 'Denied', 'Pending']))

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    return model, best_val_accuracy


if __name__ == "__main__":
    # Adjusted hyperparameters
    # hyperparams= {
    #     'max_length': 256,
    #     'batch_size': 16,
    #     'learning_rate': 3e-5,
    #     'bert_learning_rate': 2e-5,
    #     'dropout': 0.1,
    #     'num_epochs': 10,
    #     'warmup_ratio': 0.1,
    #     'early_stopping_patience': 3
    # }

    hyperparams = {
        'max_length': 256,
        'batch_size': 32,  # Increased batch size
        'learning_rate': 1e-4,  # Increased classifier learning rate
        'bert_learning_rate': 5e-5,  # Increased BERT learning rate
        'dropout': 0.3,  # Increased dropout rate
        'num_epochs': 25,  # Increased number of epochs
        'warmup_ratio': 0.1,
        'early_stopping_patience': 5  # Increased patience
    }

    model, best_accuracy = train(hyperparams)
    print(f"\nTraining completed! Best validation accuracy: {best_accuracy:.2f}%")
