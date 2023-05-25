import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

sns.set_theme()

class EEGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=21, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Take the output from the final timetep
        final_time_step_out = lstm_out[:, -1, :]
        y_pred = self.fc(final_time_step_out)
        return y_pred


class ConvLSTM1(nn.Module):
    def __init__(self):
        super(ConvLSTM1, self).__init__()
        self.conv1 = nn.Conv1d(21, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=2, batch_first=True, dropout=0.5)
        self.fc1 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # rearrange to (batch_size, num_channels, sequence_length)
        x = F.relu(self.bn1(self.conv1(x)))
        x = x.permute(0, 2, 1)  # rearrange back to (batch_size, sequence_length, num_channels)
        lstm_out, _ = self.lstm(x)
        final_time_step_out = lstm_out[:, -1, :]  # take the output from the final timestep
        x = F.relu(self.bn2(self.fc1(final_time_step_out)))  # ReLU activation and batch normalization
        y_pred = self.fc2(x)
        return y_pred


class ConvLSTM(nn.Module):
    def __init__(self):
        super(ConvLSTM, self).__init__()
        self.conv1 = nn.Conv1d(21, 32, kernel_size=3, stride=1, padding=1)
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=2, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # rearrange to (batch_size, num_channels, sequence_length)
        x = F.relu(self.conv1(x))
        x = x.permute(0, 2, 1)  # rearrange back to (batch_size, sequence_length, num_channels)
        lstm_out, _ = self.lstm(x)
        final_time_step_out = lstm_out[:, -1, :]  # take the output from the final timestep
        y_pred = self.fc(final_time_step_out)
        return y_pred


def get_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    train_data = np.load("out/fft_train_full_1000.npy")
    eval_data = np.load("out/fft_eval_full_1000.npy")

    train_age = np.load("out/fft_train_ages_1000.npy")
    eval_age = np.load("out/fft_eval_ages_1000.npy")

    train_data = torch.from_numpy(train_data)
    train_labels = torch.from_numpy(train_age)
    eval_data = torch.from_numpy(eval_data)
    eval_labels = torch.from_numpy(eval_age)

    # Standardize data
    train_mean = train_data.mean()
    train_std = train_data.std()
    train_data = (train_data - train_mean) / train_std
    eval_data = (eval_data - train_mean) / train_std

    train_dataset = EEGDataset(train_data, train_labels.view(-1, 1))
    eval_dataset = EEGDataset(eval_data, eval_labels.view(-1, 1))

    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=128)

    return train_dataloader, eval_dataloader


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


if __name__ == '__main__':
    train_dataloader, eval_dataloader = get_data()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = LSTM()
    model.apply(init_weights)
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)  # learning rate scheduler

    epochs = 200
    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        for i, (train_data, train_labels) in enumerate(train_dataloader):
            train_data = train_data.to(device, dtype=torch.float)
            train_labels = train_labels.to(device, dtype=torch.float)

            optimizer.zero_grad()

            outputs = model(train_data)
            loss = torch.sqrt(criterion(outputs, train_labels))

            loss.backward()
            optimizer.step()

        scheduler.step()

        model.eval()
        with torch.no_grad():
            eval_losses = []
            for i, (inputs, labels) in enumerate(eval_dataloader):
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.float)

                outputs = model(inputs)
                loss = torch.sqrt(criterion(outputs, labels))
                eval_losses.append(loss.item())
            avg_eval_loss = sum(eval_losses) / len(eval_losses)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f} - Eval loss: {avg_eval_loss:.4f}")

        # Save the model if the evaluation loss decreased.
        if avg_eval_loss < best_loss:
            print(f"New best model found! Eval loss: {avg_eval_loss:.4f}")
            best_loss = avg_eval_loss
            torch.save(model.state_dict(), 'best_model.pth')

    # Show predictions
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    with torch.no_grad():
        predictions = []
        for i, (inputs, labels) in enumerate(eval_dataloader):
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)

            outputs = model(inputs)
            out = outputs.cpu().numpy()
            predictions.append(out)
        predictions = np.concatenate(predictions)

    for i, (pred, label) in enumerate(zip(predictions, eval_dataloader.dataset.labels)):
        print(f"Actual: {label[0]:.2f} - Predicted: {pred[0]:.2f}")

"""
Example output from predictions - predicted age is basically the mean of the training data :(
--------------------------------
Actual: 65.00 - Predicted: 44.05
Actual: 79.00 - Predicted: 44.05
Actual: 56.00 - Predicted: 44.05
Actual: 58.00 - Predicted: 44.05
Actual: 31.00 - Predicted: 44.05
Actual: 48.00 - Predicted: 44.05
Actual: 58.00 - Predicted: 44.05
Actual: 43.00 - Predicted: 44.05
Actual: 61.00 - Predicted: 44.05
Actual: 29.00 - Predicted: 44.05
Actual: 48.00 - Predicted: 44.05
Actual: 71.00 - Predicted: 44.05
Actual: 62.00 - Predicted: 44.05
Actual: 43.00 - Predicted: 44.04
Actual: 36.00 - Predicted: 44.05
Actual: 36.00 - Predicted: 44.04
Actual: 55.00 - Predicted: 44.03
Actual: 52.00 - Predicted: 44.04
Actual: 81.00 - Predicted: 44.05
"""