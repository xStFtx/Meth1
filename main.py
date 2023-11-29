import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
from torch.utils.tensorboard import SummaryWriter

# Initialize logging
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

dim_A = 5

class TensorDiagramNet(nn.Module):
    def __init__(self):
        super(TensorDiagramNet, self).__init__()
        self.fc1 = nn.Linear(dim_A**3 * 2, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(p=0.2)
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, dim_A**2)

    def forward(self, x):
        x = x.view(-1, dim_A**3 * 2)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.fc5(x)
        return x.view(-1, dim_A, dim_A)

model = TensorDiagramNet()

# Use NumPy for data generation and augmentation
np_dataset = [((np.random.randn(dim_A, dim_A, dim_A), np.random.randn(dim_A, dim_A)), np.random.randn(dim_A, dim_A)) for _ in range(200)]
augmented_dataset = []
for (input1, input2), target in np_dataset:
    noisy_input1 = input1 + np.random.normal(0, 0.1, input1.shape)
    noisy_input2 = input2 + np.random.normal(0, 0.1, input2.shape)
    augmented_dataset.append(((noisy_input1, noisy_input2), target))

# Convert to PyTorch tensors
torch_dataset = [((torch.from_numpy(input1).float(), torch.from_numpy(input2).float()), torch.from_numpy(target).float()) for (input1, input2), target in augmented_dataset]
data_loader = torch.utils.data.DataLoader(torch_dataset, batch_size=32, shuffle=True)

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

# TensorBoard writer
writer = SummaryWriter()

best_loss = float('inf')

# Training loop
for epoch in range(100):
    total_loss = 0
    for i, ((input1, input2), targets) in enumerate(data_loader):
        optimizer.zero_grad()

        # Flatten and concatenate inputs
        inputs_reshaped = torch.cat((input1.view(input1.size(0), -1), input2.view(input2.size(0), -1)), dim=1)
        
        # Forward pass
        outputs = model(inputs_reshaped)
        loss = criterion(outputs, targets)
        total_loss += loss.item()

        # Backward and optimize
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            logging.info(f'Epoch {epoch}, Batch {i}, Loss: {loss.item()}')
            writer.add_scalar('Training Loss', loss.item(), epoch * len(data_loader) + i)

    average_loss = total_loss / len(data_loader)
    scheduler.step(average_loss)
    writer.add_scalar('Average Training Loss', average_loss, epoch)

    # Save model if it has improved
    if average_loss < best_loss:
        best_loss = average_loss
        torch.save(model.state_dict(), 'best_model.pth')
        logging.info(f'Model improved and saved at epoch {epoch} with loss {average_loss}')

writer.close()
logging.info("Training complete.")

# Remember to run `tensorboard --logdir=runs` in your terminal to view the TensorBoard.
