import torch
import pandas as pd
from torch import nn


class AgePrediction(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(AgePrediction, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    

def get_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    train_dataset = pd.read_csv("out/fft_train.csv").to_numpy()  # 1171x1000x22
    eval_dataset = pd.read_csv("out/fft_eval.csv").to_numpy()  # 126x1000x22

    train_data = train_dataset[:, :, :-1]
    train_labels = train_dataset[:, :, -1]
    eval_data = eval_dataset[:, :, :-1]
    eval_labels = eval_dataset[:, :, -1]

    train_data = torch.tensor(train_data, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.float32)
    eval_data = torch.tensor(eval_data, dtype=torch.float32)
    eval_labels = torch.tensor(eval_labels, dtype=torch.float32)

    return train_data, train_labels, eval_data, eval_labels


if __name__ == '__main__':
    train_data, train_labels, eval_data, eval_labels = get_data()
    model = AgePrediction(input_size=21, hidden_size=150, num_layers=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        

