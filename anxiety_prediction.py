import os
from pathlib import Path
import random

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from nn_models import NetWithRE, NetWithoutRE

def map_ids(ids_tensor, id_mapping):
    # Apply the mapping to each element
    mapped_ids = ids_tensor.apply_(lambda x: id_mapping.get(x, 0))  # Use .get() with a default value if any ID is not found
    return mapped_ids

def evaluate_model(model, data_loader):
    model.eval()
    total_accuracy = 0
    total_f1 = 0
    total_mcc = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for X_batch, ids, y_batch in data_loader:
            y_pred = model(X_batch, ids)
            y_batch = torch.mode(y_batch, 1).values.long()
            _, predicted_classes = torch.max(y_pred, 1)
            total_accuracy += accuracy_score(y_batch.numpy(), predicted_classes.numpy())
            total_f1 += f1_score(y_batch.numpy(), predicted_classes.numpy(), average='weighted')
            total_mcc += matthews_corrcoef(y_batch.numpy(), predicted_classes.numpy())
            all_predictions.extend(predicted_classes.numpy())
            all_targets.extend(y_batch.numpy())

    # Compute the confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    cm = cm / cm.sum(axis=1, keepdims=True)
    accuracy = total_accuracy / len(data_loader)
    f1 = total_f1 / len(data_loader)
    mcc = total_mcc / len(data_loader)
    print(f"Accuracy: {accuracy}, F1 Score: {f1}, MCC: {mcc}")
    # Plot the confusion matrix as a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True,  cmap='Blues', xticklabels=np.unique(all_targets), yticklabels=np.unique(all_targets))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix')
    plt.savefig(f"confusion_matrix.png")
    plt.close()

def train_model(loss_fn, model, optimizer, data_loader, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X_batch, ids, y_batch in data_loader:
            optimizer.zero_grad()
            predictions = model(X_batch, ids)
            loss = loss_fn(predictions, y_batch.squeeze().long())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Average Loss: {total_loss / len(data_loader)}")

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, ids, sequence_length):
        self.y = y
        self.X = X
        self.ids = ids
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.X) // self.sequence_length

    def __getitem__(self, idx):
        start_idx = idx * self.sequence_length
        sequence = self.X[idx*self.sequence_length : (idx+1)*self.sequence_length]
        target = torch.mode(self.y[idx*self.sequence_length : (idx+1)*self.sequence_length]).values.unsqueeze(0)
        ids = self.ids[idx*self.sequence_length : (idx+1)*self.sequence_length]
        return sequence, ids,  target


def main():
    path_to_data_folder = Path(__file__).parent / "data" / "climbing_data"
    features = ["IEMG","MDF", "acc_x_mean", "acc_y_mean", "acc_z_mean","HR_mean","x","y"]
    files_train, files_test = train_test_split(os.listdir(path_to_data_folder), test_size=0.2)
    data_train = pd.concat([pd.read_csv(path_to_data_folder / file) for file in files_train], axis=0)
    X_train = data_train.loc[:, features]
    y_train = data_train.loc[:, "Fear of falling due to fatigue"]
    ids_train = data_train.loc[:, "sub_id"]
    print(set(y_train))
    
    data_test = pd.concat([pd.read_csv(path_to_data_folder / file) for file in files_test], axis=0)
    X_test = data_test.loc[:, features]
    y_test = data_test.loc[:, "Fear of falling due to fatigue"]
    ids_test = data_train.loc[:, "sub_id"]
    a = set(ids_train)
    a.update(set(ids_test))
    id_mapping = {old_id: new_id for new_id, old_id in enumerate(a)}
    mapped_subject_ids = torch.tensor([id_mapping[old_id] for old_id in a])
    num_subjects = len(mapped_subject_ids)
    X_train, X_test, y_train, y_test, ids_train, ids_test = map(torch.tensor, (X_train.to_numpy().astype(np.float32), X_test.to_numpy().astype(np.float32), 
                                                          y_train.to_numpy().astype(np.float32), y_test.to_numpy().astype(np.float32), 
                                                          ids_train.to_numpy().astype(np.int32), ids_test.to_numpy().astype(np.int32)))

    # Apply ID mapping
    ids_train = map_ids(ids_train, id_mapping)
    ids_test = map_ids(ids_test, id_mapping)

    train_dataset = TimeSeriesDataset(X_train, y_train, ids_train, sequence_length=20)
    test_dataset = TimeSeriesDataset(X_test, y_test, ids_test, sequence_length=20)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

        # Initialize and train Neural Network with RE
    NN = NetWithoutRE(input_size=len(features), num_classes=9, num_subjects=num_subjects, embedding_dim=1)
    optimizer_re = optim.Adam(NN.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()
    train_model(loss_function, NN, optimizer_re, train_loader, 25)
    evaluate_model(NN, test_loader)

if __name__ == "__main__":
    main()

    
