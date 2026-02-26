import os
import torch
import torch.nn as nn
import numpy as np

INT_TO_ACTION = {0: "up", 1: "down", 2: "left", 3: "right"}

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class AgentCNN(nn.Module):
    def __init__(self, num_blocks=4, channels=128):
        super(AgentCNN, self).__init__()
        self.conv_in = nn.Conv2d(16, channels, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 4)
        )

    def forward(self, x):
        x = self.conv_in(x)
        x = self.bn_in(x)
        x = self.relu(x)
        x = self.res_blocks(x)
        x = self.fc(x)
        return x

class Agent:
    def __init__(self, seed=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AgentCNN().to(self.device)
        
        model_path = os.path.join(os.path.dirname(__file__), "agente_2048.pt")
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval() 
        else:
            print(f"No hay modelo. Revisar {model_path}")

    def _board_to_one_hot(self, board):
        one_hot = np.zeros((16, 4, 4), dtype=np.float32)
        for i in range(4):
            for j in range(4):
                val = board[i, j]
                if val > 0:
                    idx = int(np.log2(val))
                    if idx < 16:
                        one_hot[idx, i, j] = 1.0
                else:
                    one_hot[0, i, j] = 1.0
        return one_hot

    def act(self, board: np.ndarray, legal_actions: list) -> str:
        if not legal_actions:
            return "up"
        
        state = self._board_to_one_hot(board)
        state_tensor = torch.tensor(state).unsqueeze(0).to(self.device) 
        
        with torch.no_grad():
            q_values = self.model(state_tensor).squeeze(0).cpu().numpy()
        
        for i in range(4):
            if INT_TO_ACTION[i] not in legal_actions:
                q_values[i] = -np.inf
                
        best_action_idx = int(np.argmax(q_values))
        return INT_TO_ACTION[best_action_idx]
