import torch
import torch.nn as nn
from flask import Flask, request, jsonify
import pickle
import numpy as np
import time

# Define Model: Improved BiLSTM with additional layers and techniques
class BiLSTM(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, output_size, activation_function):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.activation_function = activation_function

    def forward(self, x):
        embedded = self.embedding(x)
        out, _ = self.lstm(embedded)
        out = self.fc(out)
        if self.activation_function == 'relu':
            out = nn.ReLU()(out)
        elif self.activation_function == 'sigmoid':
            out = nn.Sigmoid()(out)
        return out

# Load the model and mappings
def load_model_and_mappings(model_path, mappings_path, activation_function, hidden_size):
    with open(mappings_path, 'rb') as f:
        mappings = pickle.load(f)
    
    word_to_index = mappings['word_to_index']
    label_to_index = mappings['label_to_index']
    
    input_size = len(word_to_index)
    output_size = len(label_to_index)
    
    model = BiLSTM(input_size, 300, hidden_size, output_size, activation_function)  # Set embedding dimension to 300
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    return model, word_to_index, label_to_index

# Initialize Flask application
app = Flask(__name__)

# Load model and mappings
model_path = '/user/HS401/aj01496/model_relu_CrossEntropyLoss_Adagrad.pth'
mappings_path = '/user/HS401/aj01496/mappings_relu_CrossEntropyLoss_Adagrad.pkl'
activation_function = 'relu'
hidden_size = 64
model, word_to_index, label_to_index = load_model_and_mappings(model_path, mappings_path, activation_function, hidden_size)

# Global variables to track min and max response times
min_response_time = float('inf')
max_response_time = float('-inf')

def log_request(tokens, predictions, start_time, end_time, response_time):
    global min_response_time, max_response_time

    with open("server_log.txt", "a") as f:
        f.write(f"Start Time: {start_time}\n")
        f.write(f"End Time: {end_time}\n")
        f.write(f"Tokens: {tokens}\n")
        f.write(f"Predictions: {predictions}\n")
        f.write(f"Response Time: {response_time:.2f} seconds\n")
        f.write("\n")

    # Update global min and max response times
    if response_time < min_response_time:
        min_response_time = response_time
    if response_time > max_response_time:
        max_response_time = response_time

@app.route('/predict', methods=['POST'])
def predict():
    global min_response_time, max_response_time

    data = request.json
    tokens_list = data['tokens']  # tokens_list is either a list of tokens or a list of lists of tokens

    # Check if the input is a single list of tokens
    if all(isinstance(token, str) for token in tokens_list):
        tokens_list = [tokens_list]  # Convert to a list of lists

    predictions = []
    start_time = time.strftime("%Y-%m-%d %H:%M:%S")
    start = time.time()

    for tokens in tokens_list:
        tokens_numeric = [word_to_index.get(token, 0) for token in tokens]  # Convert tokens to indices
        input_tensor = torch.tensor(tokens_numeric, dtype=torch.long).unsqueeze(0)  # Create a batch with a single sequence

        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 2)
            predicted = predicted.view(-1).cpu().numpy()

        decoded_prediction = [list(label_to_index.keys())[label] for label in predicted[:len(tokens)]]
        predictions.append(decoded_prediction)


    end_time = time.strftime("%Y-%m-%d %H:%M:%S")
    end = time.time()
    response_time = end - start

    log_request(tokens_list, predictions, start_time, end_time, response_time)

    return jsonify({
        'prediction': predictions,
        'min_response_time': f"{min_response_time:.2f} seconds",
        'max_response_time': f"{max_response_time:.2f} seconds"
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
