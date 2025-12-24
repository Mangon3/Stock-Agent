import torch.nn as nn

class LSTM_StockNet(nn.Module):
    def __init__(self, input_size: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(LSTM_StockNet, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout
        )
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # LSTM returns output, (hidden_state, cell_state)
        out, (h_n, c_n) = self.lstm(x)
        last_step = out[:, -1, :]
        last_step = self.bn(last_step)
        last_step = self.dropout(last_step)
        
        return self.fc(last_step)