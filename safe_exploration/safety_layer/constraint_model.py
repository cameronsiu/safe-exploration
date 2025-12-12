import torch
from torch.nn.init import uniform_

from safe_exploration.core.config import Config
from safe_exploration.core.net import Net
import torch.nn.functional as F
import torch.nn as nn

class ConstraintModel(nn.Module):
    def __init__(self, observation_dim, action_dim, model_file:str=None):
        super(ConstraintModel, self).__init__()
        config = Config.get().safety_layer.constraint_model
        self.lidar_dim = 60
        self.velocity_dim = 2

        # 1 to 16 channels
        # reduces 60 to 30
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2)
        self.batch_norm1 = nn.BatchNorm1d(16)

        # 16 to 32 channels
        # reduces 30 to 15
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
        self.batch_norm2 = nn.BatchNorm1d(32)

        # 32 to 64 channels
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)

        # Compute feature dimension after flatten
        cnn_output_dim = 64 * self.lidar_dim

        # Fully connected head combines lidar features + velocity
        self.fc = nn.Sequential(
            nn.Linear(cnn_output_dim + self.velocity_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

        if model_file is not None:
            state_dict = torch.load(model_file)
            self.load_state_dict(state_dict)

    def save(self, output_folder:str, model_index:int):
        state_dict = self.state_dict()
        torch.save(state_dict, f"{output_folder}/constraint_{model_index}.pt")

    def forward(self, obs):
        lidar = obs[:, :self.lidar_dim].unsqueeze(1)     # (B, 1, lidar_dim)
        vel   = obs[:, self.lidar_dim:]                  # (B, vel_dim)

        # CNN features
        # (B, 64, lidar_dim)
        lidar = F.relu(self.batch_norm1(self.conv1(lidar)))
        lidar = F.relu(self.batch_norm2(self.conv2(lidar)))
        lidar = F.relu(self.conv3(lidar))
        cnn_feat = lidar.view(obs.size(0), -1)

        # Combine CNN + velocity
        fused = torch.cat((cnn_feat, vel), dim=1)

        g = self.fc(fused)
        return g
