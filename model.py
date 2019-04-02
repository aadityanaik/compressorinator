import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # self.inputLayer = nn.Linear(64, 32)
        self.hiddenLayer = nn.Linear(64, 16)
        # self.hiddenLayer1 = nn.Linear(16, 8)
        # self.hiddenLayer2 = nn.Linear(16, 64)

    def forward(self, x):
        # x = F.relu(self.inputLayer(x))
        x = self.hiddenLayer(x)
        # x = F.relu(self.hiddenLayer1(x))
        # x = self.hiddenLayer2(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # self.inputLayer = nn.Linear(64, 16)
        # self.hiddenLayer = nn.Linear(32, 16)
        # self.hiddenLayer1 = nn.Linear(16, 32)
        self.hiddenLayer2 = nn.Linear(16, 64)

    def forward(self, x):
        # x = F.relu(self.inputLayer(x))
        # x = F.relu(self.hiddenLayer(x))
        # x = F.relu(self.hiddenLayer1(x))
        x = self.hiddenLayer2(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        # self.inputLayer = nn.Linear(64, 16)
        # self.hiddenLayer = nn.Linear(32, 16)
        # self.hiddenLayer1 = nn.Linear(16, 8)
        # self.hiddenLayer2 = nn.Linear(16, 64)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        # x = F.relu(self.inputLayer(x))
        # x = F.relu(self.hiddenLayer(x))
        # x = F.relu(self.hiddenLayer1(x))
        # x = self.hiddenLayer2(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s

        return num_features
