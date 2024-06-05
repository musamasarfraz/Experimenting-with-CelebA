import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, img_channels=3, img_size=64, z_dim=20, attr_dim=40):
        super(VAE, self).__init__()
        self.attr_dim = attr_dim
        self.enc_conv1 = nn.Conv2d(img_channels, 32, 4, 2, 1)
        self.enc_conv2 = nn.Conv2d(32, 64, 4, 2, 1)
        self.enc_conv3 = nn.Conv2d(64, 128, 4, 2, 1)
        self.fc1 = nn.Linear(128*8*8 + attr_dim, 256)
        self.fc_mu = nn.Linear(256, z_dim)
        self.fc_logvar = nn.Linear(256, z_dim)
        self.fc2 = nn.Linear(z_dim + attr_dim, 256)
        self.dec_fc = nn.Linear(256, 128*8*8)
        self.dec_conv1 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.dec_conv2 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.dec_conv3 = nn.ConvTranspose2d(32, img_channels, 4, 2, 1)

    def encode(self, x, attrs):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.cat((x, attrs), dim=1)
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, attrs):
        z = torch.cat((z, attrs), dim=1)
        x = F.relu(self.fc2(z))
        x = F.relu(self.dec_fc(x))
        x = x.view(x.size(0), 128, 8, 8)
        x = F.relu(self.dec_conv1(x))
        x = F.relu(self.dec_conv2(x))
        x = torch.sigmoid(self.dec_conv3(x))
        return x

    def forward(self, x, attrs):
        mu, logvar = self.encode(x, attrs)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, attrs), mu, logvar