import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.amp import autocast, GradScaler

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Ruta al dataset en Scratch')
    parser.add_argument('--output_dir', default='./', help='Ruta para guardar artefactos en Home')
    parser.add_argument('--epochs', type=int, default=30, help='Numero de epocas')
    args = parser.parse_args()

    x_path = os.path.join(args.data_dir, 'dataset', 'X.npy')
    y_path = os.path.join(args.data_dir, 'dataset', 'y.npy')

    print(f"Cargando dataset desde: {x_path}")
    X_data = np.load(x_path)
    y_data = np.load(y_path)

    tensor_x = torch.Tensor(X_data)
    tensor_y = torch.LongTensor(y_data)
    dataset = TensorDataset(tensor_x, tensor_y)
    
    loader = DataLoader(dataset, batch_size=4096, shuffle=True, num_workers=8, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entrenando ResNet en: {device}")
    
    model = AgentCNN().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    print(f"Iniciando entrenamiento por {args.epochs} epocas")
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            with autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f} - LR: {optimizer.param_groups[0]['lr']}")
        scheduler.step(avg_loss)

    save_path = os.path.split(args.output_dir)[0] if os.path.isfile(args.output_dir) else args.output_dir
    model_file = os.path.join(save_path, "agente_2048.pt")
    torch.save(model.state_dict(), model_file)
    print(f"Modelo guardado en: {model_file}")

if __name__ == '__main__':
    main()
