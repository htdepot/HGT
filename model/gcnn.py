import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv



class GCNN(nn.Module):
    def __init__(self, in_channel, features, out_channel, K, hidden):
        super(GCNN, self).__init__()
        self.conv1_1 = ChebConv(1, features, K=K)
        self.conv1_2 = ChebConv(features, features, K=K)
        self.conv1_3 = ChebConv(features, features, K=K)
        self.conv2_1 = ChebConv(features, features, K=K)
        self.conv2_2 = ChebConv(features, features, K=K)
        self.conv3_1 = ChebConv(features, features, K=K)
        self.lin1 = nn.Linear(features * in_channel, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.lin3 = nn.Linear(hidden, out_channel)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1_1(x, edge_index))
        temp = x
        x = F.relu(self.conv1_2(x, edge_index))
        x = self.conv1_3(x, edge_index)
        x = temp + x
        temp = x
        x = F.relu(self.conv2_1(x, edge_index))
        x = self.conv2_2(x, edge_index)
        x = temp + x
        x = F.relu(self.conv3_1(x, edge_index))
        batch_size = int(batch.max().item() + 1)
        x = torch.reshape(x, (batch_size, -1))
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        return x