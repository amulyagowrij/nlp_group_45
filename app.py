import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from flask import Flask, request, jsonify
import pickle
from datasets import load_dataset
import gzip
import shutil
import pandas as pd

# Function to decompress the GloVe file
def decompress_glove_file(input_path, output_path):
    try:
        with gzip.open(input_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    except OSError as e:
        print(f"Error decompressing the file: {e}")

# Paths to the GloVe files (update these paths accordingly)
compressed_glove_file_path = '/user/HS401/aj01496/gensim-data/glove-wiki-gigaword-300_tmp/glove-wiki-gigaword-300.gz'
decompressed_glove_file_path = '/user/HS401/aj01496/gensim-data/glove-wiki-gigaword-300_tmp/glove.6B.300d.txt'

# Decompress the GloVe file if it hasn't been decompressed already
if not os.path.exists(decompressed_glove_file_path):
    decompress_glove_file(compressed_glove_file_path, decompressed_glove_file_path)

# Function to load GloVe vectors from a file
def load_glove_vectors(filepath):
    glove_vectors = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            glove_vectors[word] = vector
    return glove_vectors

# Load GloVe word vectors
word_vectors = load_glove_vectors(decompressed_glove_file_path)
print(f"Loaded {len(word_vectors)} word vectors.")

# Load dataset
dataset = load_dataset("surrey-nlp/PLOD-CW")
train_df = dataset['train'].to_pandas()
test_df = dataset['test'].to_pandas()
print("Loaded dataset.")

# Prepare data by mapping tokens to indices and pad sequences for uniformity
word_to_index = {}
index_to_vector = []

for token in pd.concat([train_df['tokens'], test_df['tokens']]).explode().unique():
    if token in word_vectors:
        word_to_index[token] = len(word_to_index)
        index_to_vector.append(word_vectors[token])
    else:
        word_to_index[token] = len(word_to_index)
        index_to_vector.append(np.random.normal(size=(300,)))

index_to_vector = np.array(index_to_vector)
print(f"Created word_to_index mapping with {len(word_to_index)} words.")

train_tokens = [[word_to_index[token] for token in tokens] for tokens in train_df['tokens']]
test_tokens = [[word_to_index[token] for token in tokens] for tokens in test_df['tokens']]

# Convert labels to numeric format and pad sequences
label_to_index = {'B-O': 0, 'B-LF': 1, 'B-AC': 2, 'I-LF': 3}
num_classes = len(label_to_index)
train_labels_numeric = [[label_to_index[label] for label in labels] for labels in train_df['ner_tags']]
test_labels_numeric = [[label_to_index[label] for label in labels] for labels in test_df['ner_tags']]

# Split train data into train and validation sets
train_tokens, val_tokens, train_labels, val_labels = train_test_split(train_tokens, train_labels_numeric, test_size=0.1, random_state=42)

# Pad token sequences
max_len = max(max(len(tokens) for tokens in train_tokens), max(len(tokens) for tokens in test_tokens))
train_tokens_padded = pad_sequence([torch.tensor(tokens, dtype=torch.long) for tokens in train_tokens], batch_first=True, padding_value=0)
val_tokens_padded = pad_sequence([torch.tensor(tokens, dtype=torch.long) for tokens in val_tokens], batch_first=True, padding_value=0)
test_tokens_padded = pad_sequence([torch.tensor(tokens, dtype=torch.long) for tokens in test_tokens], batch_first=True, padding_value=0)

train_labels_padded = pad_sequence([torch.tensor(labels, dtype=torch.long) for labels in train_labels], batch_first=True, padding_value=0)
val_labels_padded = pad_sequence([torch.tensor(labels, dtype=torch.long) for labels in val_labels], batch_first=True, padding_value=0)
test_labels_padded = pad_sequence([torch.tensor(labels, dtype=torch.long) for labels in test_labels_numeric], batch_first=True, padding_value=0)
print("Padded token and label sequences.")

# Define Model: Improved BiLSTM with additional layers and techniques
class BiLSTM(nn.Module):
    def __init__(self, embedding_matrix, hidden_size, output_size):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float32), freeze=False)  # Use pre-trained GloVe embeddings
        self.lstm = nn.LSTM(embedding_matrix.shape[1], hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(0.5)  # Add dropout layer

    def forward(self, x):
        embedded = self.embedding(x)  # Embed the input
        out, _ = self.lstm(embedded)
        out = self.fc(self.dropout(out))  # Apply dropout before the final layer
        return out

# Convert tokens and labels to PyTorch tensors
train_tokens_tensor = train_tokens_padded
val_tokens_tensor = val_tokens_padded
train_labels_tensor = train_labels_padded
val_labels_tensor = val_labels_padded
test_tokens_tensor = test_tokens_padded
test_labels_tensor = test_labels_padded

# Define function for training and evaluating the model with hyperparameters and activation function
def train_and_evaluate_model(lr, epochs, batch_size, hidden_size, train_tokens_tensor, train_labels_tensor, val_tokens_tensor, val_labels_tensor, test_tokens_tensor, test_labels_tensor, embedding_matrix):
    # Initialize model
    model = BiLSTM(embedding_matrix=embedding_matrix, hidden_size=hidden_size, output_size=num_classes)

    # Define loss function and optimizer
    class_weights = torch.tensor([0.1, 0.5, 0.2, 0.2])
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=0)  # Weighted loss function to handle class imbalance
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.5, verbose=True)

    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0

    # Training loop
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        total_loss = 0
        for i in range(0, len(train_tokens_tensor), batch_size):
            optimizer.zero_grad()
            batch_X = train_tokens_tensor[i:i+batch_size]
            batch_y = train_labels_tensor[i:i+batch_size]

            # Forward pass
            outputs = model(batch_X)

            # Compute loss
            loss = criterion(outputs.permute(0, 2, 1), batch_y)  # Permute output tensor to match expected shape

            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()

            total_loss += loss.item()

        # Store the average loss per epoch
        avg_loss = total_loss / len(train_tokens_tensor)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss}")

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_tokens_tensor)
            _, val_predicted = torch.max(val_outputs, 2)
            val_predicted = val_predicted.view(-1).cpu().numpy()
            val_true = val_labels_tensor.view(-1).cpu().numpy()
            val_loss = criterion(val_outputs.permute(0, 2, 1), val_labels_tensor).item()
            val_report = classification_report(val_true, val_predicted, target_names=list(label_to_index.keys()), zero_division=0)
            print(f"Validation Report:\n{val_report}")

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), 'bilstm_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping")
                    break

    # Save the model
    with open('word_to_index.pkl', 'wb') as f:
        pickle.dump(word_to_index, f)
    with open('label_to_index.pkl', 'wb') as f:
        pickle.dump(label_to_index, f)

    return model

# Define specific hyperparameters
lr = 0.001
epochs = 20
batch_size = 32
hidden_size = 256  # Increase the hidden size

# Train the model
model = train_and_evaluate_model(lr, epochs, batch_size, hidden_size, train_tokens_tensor, train_labels_tensor, val_tokens_tensor, val_labels_tensor, test_tokens_tensor, test_labels_tensor, index_to_vector)

# Define the Flask application
app = Flask(__name__)

# Load the trained BiLSTM model
model = BiLSTM(embedding_matrix=index_to_vector, hidden_size=hidden_size, output_size=num_classes)
model.load_state_dict(torch.load('bilstm_model.pth', map_location=torch.device('cpu')))
model.eval()

# Load the word_to_index and label_to_index mappings
with open('word_to_index.pkl', 'rb') as f:
    word_to_index = pickle.load(f)
with open('label_to_index.pkl', 'rb') as f:
    label_to_index = pickle.load(f)
index_to_label = {v: k for k, v in label_to_index.items()}  # Create reverse mapping

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    tokens = data['tokens']
    
    # Preprocess the input text
    tokens_numeric = [word_to_index.get(token, 0) for token in tokens]  # Convert tokens to indices, default to 0 if not found
    input_tensor = torch.tensor(tokens_numeric, dtype=torch.long).unsqueeze(0)  # Create a batch with a single sequence
    
    # Convert to tensor and predict using BiLSTM
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 2)
        predicted = predicted.view(-1).cpu().numpy()
    
    # Decode the prediction, only consider the length of the input tokens
    decoded_prediction = [index_to_label[idx] for idx in predicted[:len(tokens)]]
    print("Decoded predictions:", decoded_prediction)
    
    return jsonify({
        'prediction': decoded_prediction
    })

if __name__ == '__main__':
    app.run(debug=True)
