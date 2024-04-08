import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_scheduler
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import gradio as gr
from tqdm import tqdm
import matplotlib.image as mpimg

# Load Financial PhraseBank dataset with a specific configuration
dataset = load_dataset('financial_phrasebank', split='train', name='sentences_allagree')

# Extract training data and labels
train_texts = dataset['sentence']
train_labels = dataset['label']

# Split dataset into train and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.2,
                                                                    random_state=42)

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                      num_labels=3)  # 3 labels: positive, neutral, negative

# Tokenize inputs
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)


# Define PyTorch datasets
class FinancialDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = FinancialDataset(train_encodings, train_labels)
val_dataset = FinancialDataset(val_encodings, val_labels)

# Define data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Define optimizer, scheduler, and loss function
optimizer = AdamW(model.parameters(), lr=5e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Create directories if not exist
if not os.path.exists("saved_models/bert_model"):
    os.makedirs("saved_models/bert_model")
if not os.path.exists("plt"):
    os.makedirs("plt")


# Define function to train model and plot accuracy curve
def train_model(total_epochs, save_every, learning_rate):
    global model
    global optimizer
    global loss_fn
    global train_loader
    global val_loader

    train_accuracies = []
    val_accuracies = []

    # Move model to CUDA device if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Define optimizer with given learning rate
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0,
                              num_training_steps=len(train_loader) * total_epochs)

    # Training loop
    for epoch in range(1, total_epochs + 1):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs}"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        epoch_train_accuracy = train_correct / train_total
        train_accuracies.append(epoch_train_accuracy)

        # Evaluation loop
        model.eval()
        val_correct = 0
        val_total = 0
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

        epoch_val_accuracy = val_correct / val_total
        val_accuracies.append(epoch_val_accuracy)

        # Plot accuracy curve and save model every save_every epochs
        if epoch % save_every == 0 or epoch == total_epochs:
            epochs = range(1, epoch + 1)
            plt.figure(figsize=(10, 5))
            plt.plot(epochs, train_accuracies, marker='o', label='Training Accuracy')
            plt.plot(epochs, val_accuracies, marker='o', label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Training and Validation Accuracy')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"plt/accuracy_plot_epoch_{epoch}.png")  # Save accuracy plot as image
            plt.close()

            # Save model
            model_name = f"bert_model_epoch_{epoch}"
            model_path = os.path.join("saved_models/bert_model", model_name)
            model.save_pretrained(model_path)
            print(f"Model saved at epoch {epoch}")

            # Read the saved image and return it
            image = mpimg.imread(f"plt/accuracy_plot_epoch_{epoch}.png")
            yield image


# Launch Gradio interface
iface = gr.Interface(fn=train_model, inputs=["number", "number", "number"], outputs="image", title="Training Progress",
                     description="Specify the number of epochs for training, epochs per save, and learning rate.")
iface.launch()
