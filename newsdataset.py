import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity

# Load and preprocess your data
df1 = pd.read_json('/Users/shagverdiyev/Downloads/one_week/20170101', lines=True, nrows=100000)
"""df2 = pd.read_json('/Users/shagverdiyev/Downloads/one_week/20170102', lines=True, nrows=200000)
df3 = pd.read_json('/Users/shagverdiyev/Downloads/one_week/20170103', lines=True, nrows=200000)
df4 = pd.read_json('/Users/shagverdiyev/Downloads/one_week/20170104', lines=True, nrows=200000)
df5 = pd.read_json('/Users/shagverdiyev/Downloads/one_week/20170105', lines=True, nrows=200000)

df = pd.concat([df1, df2,df3,df4,df5])

# Reset the index of the new dataframe
df.reset_index(drop=True, inplace=True)|"""

df_cleaned = df1.dropna(subset=['title', 'userId'])
unique_title_count = df1['title'].nunique()


# tokenizing and embedding the titles

title_to_embedding = {}

#loading pre-trained model tokenizer and model(for norwegian language)
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
model = BertModel.from_pretrained('bert-base-multilingual-uncased')

unique_titles = df_cleaned['title'].unique()

for title in unique_titles:
    # Tokenize the title

    input_ids = tokenizer.encode(title, add_special_tokens=True)
    input_ids = torch.tensor(input_ids).unsqueeze(0)  # Add batch dimension

    # Generate embeddings

    with torch.no_grad():
        last_hidden_states = model(input_ids)
    embedding = last_hidden_states[0][:, 0, :].numpy()
    title_to_embedding[title] = embedding



# Getting sessions based on activity
sessions_df = df_cleaned[['userId', 'title', 'time']].copy()
sessions_df = sessions_df.sort_values(['userId', 'time'])

grouped = sessions_df.groupby('userId')

sessions = []  # List to hold the sessions

sessions_df['time'] = pd.to_datetime(sessions_df['time'], unit='s')

for _, group in grouped:
    session = []  # List to hold the current session
    last_timestamp = None  # Variable to hold the timestamp of the last interaction

    # Iterate over each row in the group
    for _, row in group.iterrows():
        # If this is the first interaction or the time since the last interaction is less than 30 minutes
        if last_timestamp is None or (row['time'] - last_timestamp).total_seconds() < 1800:
            # Add this interaction to the current session
            session.append(row['title'])
        else:
            # If the time since the last interaction is 30 minutes or more, end the current session and start a new one
            sessions.append(session)
            session = [row['title']]

        last_timestamp = row['time']

    # Add the last session of this user
    if session:
        sessions.append(session)

#deleting sessions which has length of 1
sessions = [session for session in sessions if len(session) > 1]


embedded_sessions = []

for session in sessions:
    # Map each title in the session to its embedding and add the sequence to the new list
    embedded_session = [title_to_embedding[title] for title in session]
    embedded_sessions.append(embedded_session)


# Convert the sessions to tensors
tensor_sessions = [torch.tensor(session) for session in embedded_sessions]

# Store the original lengths of the sequences
original_lengths = [len(tensor) for tensor in tensor_sessions]



# Pad the sessions and create a tensor of shape (batch_size, sequence_length, input_size)
padded_sessions = nn.utils.rnn.pad_sequence(tensor_sessions, batch_first=True)

# The input sequences are all elements except the last one
inputs = padded_sessions[:, :-1, :]

targets = []
# Loop over the padded_sessions and original_lengths
for i, length in enumerate(original_lengths):
    # The target is the last non-zero element in the sequence
    target = padded_sessions[i, length-1, :]
    targets.append(target)
# Convert the list of targets to a tensor
targets = torch.stack(targets).squeeze(1)

# Split the data into training and testing datasets
train_ratio = 0.8  # 80% for training
total_sessions = len(inputs)
train_index = int(train_ratio * total_sessions)
train_inputs = inputs[:train_index]
train_targets = targets[:train_index]

test_inputs = inputs[train_index:]
test_targets = targets[train_index:]


num_all_zeros = 0
for tensor in targets:
    if torch.all(tensor == 0):
        num_all_zeros += 1

print(f"Number of all-zero tensors in targets: {num_all_zeros}")




# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define your LSTM model
class SessionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(SessionModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)  # The output size is set to the input size

    def forward(self, x):
        batch_size = x.size(0)  # Get the batch size
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)  # Initialize hidden state
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)  # Initialize cell state
        x = x.squeeze(dim=2)

        out, _ = self.lstm(x, (h0, c0))  # LSTM with input, hidden, and internal state
        out = self.fc(out[:, -1, :])  # Classify only the last output of the sequence

        return out


# Create an instance of SessionModel
input_size = train_inputs.shape[-1]
hidden_size = 128
num_layers = 2
model = SessionModel(input_size, hidden_size, num_layers).to(device)

# Define a loss function and an optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Number of epochs
epochs = 100

# Train the model
model.train()

for epoch in range(epochs):
    # Forward pass
    outputs = model(train_inputs.to(device))
    loss = criterion(outputs, train_targets.to(device))

    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, loss.item()))
    
  
    
  
model.eval() 
# Compute the predicted embeddings in batches
predicted_embeddings = []
batch_size = 128
for i in range(0, len(test_inputs), batch_size):
    batch = test_inputs[i:i+batch_size].to(device)
    with torch.no_grad():
        output_list = model(batch).tolist()
        predicted_embeddings.extend(output_list)

predicted_embeddings = torch.stack([torch.Tensor(i) for i in predicted_embeddings])
        
    
# Compute the cosine similarity in batches
cos = nn.CosineSimilarity(dim=1)
similarities = []
title_embeddings = torch.Tensor(list(title_to_embedding.values())).squeeze(1) # Remove unnecessary dimension

for i in range(0, len(predicted_embeddings), batch_size):
    batch = predicted_embeddings[i:i+batch_size]
    
    # For each element in the batch, we want to compute similarity with all title embeddings
    for embedding in batch:
        embedding = embedding.unsqueeze(0)  # Add an extra dimension
        similarity = cos(embedding, title_embeddings)  # Calculate cosine similarity
        similarities.append(similarity)

# Convert list to tensor
similarities = torch.stack(similarities)


# Parameters
N = 3
cos = torch.nn.CosineSimilarity(dim=1)

# Flatten test targets tensor
test_targets_flattened = test_targets.reshape(-1, test_targets.size(-1))

# Convert test targets flattened to list of tensors
test_targets_list = [torch.Tensor(i) for i in test_targets_flattened.numpy()]

def check_similarity(array1, array2, threshold=0.983):
    # calculate cosine similarity
    similarity = cosine_similarity([array1], [array2])
    # return whether similarity is above the threshold
    return similarity >= threshold

## Set of actual targets
actual_targets_set = set(test_targets_list)

precision = 0
recall = 0
MRR = 0
for i, predicted_embedding in enumerate(predicted_embeddings):
    # Print out the iteration number
    print(f'Processing embedding #{i}')    # Calculate cosine similarity between predicted embedding and all target embeddings
    
    
    similarities = cos(predicted_embedding.unsqueeze(0), test_targets_flattened)
    
    # Get top N most similar target embeddings
    top_n_indices = similarities.topk(N)[1]
    top_n_targets = test_targets_flattened[top_n_indices]
    # Convert tensors to list for set operations
    top_n_targets_list = [tensor.numpy() for tensor in top_n_targets]
    actual_targets_list = [tensor.numpy() for tensor in actual_targets_set]
   
    # Calculate true positives, false positives, and false negatives
    true_positives = [target for target in top_n_targets_list if any(check_similarity(target, actual_target) for actual_target in actual_targets_list)]
    false_positives = [target for target in top_n_targets_list if not any(check_similarity(target, actual_target) for actual_target in actual_targets_list)]
    
    false_negatives = [target for target in actual_targets_list if not any(check_similarity(target, top_n_target) for top_n_target in top_n_targets_list)]
    
   
    # Calculate precision and recall
    precision += len(true_positives) / (len(true_positives) + len(false_positives) + 0.7) #added a small number to denominator to prevent division by zero
    recall += len(true_positives) / (len(true_positives) + len(false_negatives) + 1e-10) #added a small number to denominator to prevent division by zero
    
    # Calculate MRR
    rank_list = [i for i, target in enumerate(top_n_targets_list) if any(check_similarity(target, actual_target) for actual_target in actual_targets_list)]
    MRR += sum(1 / (rank + 1) for rank in rank_list) / len(rank_list) if rank_list else 0


# Average precision, recall, and MRR
precision /= len(predicted_embeddings)
recall /= len(predicted_embeddings)
MRR /= len(predicted_embeddings)

print('Precision: {:.16f}'.format(precision))
print('Recall: {:.16f}'.format(recall))
print('MRR: {:.16f}'.format(MRR))


