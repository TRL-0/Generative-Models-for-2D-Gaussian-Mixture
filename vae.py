import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from utils import set_seed, visualize_data, normalize_data


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim, block_num=2):
        super(Encoder, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            *([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU()
            ]*block_num)
        )
        self.mu = nn.Linear(hidden_dim, z_dim)
        self.log_var = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        hidden = self.feature(x)
        z_mu = self.mu(hidden)
        z_log_var = self.log_var(hidden)
        return z_mu, z_log_var


class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, output_dim,block_num=2):
        super(Decoder, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.LeakyReLU(),
            *([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU()
            ]*block_num)
        )
        self.out_mu = nn.Linear(hidden_dim, output_dim)
        self.out_log_var = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        hidden = self.feature(z)
        mu = self.out_mu(hidden)
        log_var = self.out_log_var(hidden)
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        predicted = mu + eps * std
        return predicted


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim,block_num=2):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, z_dim,block_num=block_num)
        self.decoder = Decoder(z_dim, hidden_dim, input_dim,block_num=block_num)

    def forward(self, x):
        z_mu, z_log_var = self.encoder(x)
        std = torch.exp(z_log_var / 2)
        eps = torch.randn_like(std)
        z = z_mu + eps * std
        return self.decoder(z), z_mu, z_log_var


def loss_function(recon_x, x, mu, log_var,beta=1.0):
    
    mse = F.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + log_var - mu*mu - torch.exp(log_var))
    return (mse+kld*beta)/x.shape[0]


def train(vae, data_loader, epochs=100):
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)
    vae.train()
    for epoch in range(epochs):
        par = tqdm(data_loader, ncols=100)
        for x in par:
            x = x[0]
            recon_x, mu, log_var = vae(x)
            loss = loss_function(recon_x, x, mu, log_var,beta=1.0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if not scheduler is None:
                scheduler.step()
            par.set_description(
                f"Epoch[{epoch + 1}/{epochs}], Loss: {loss.item()}")
            
        if epoch % 20 == 0:
            save_model(vae, './ckpt/vae.pth')


def validate(vae, data_loader):
    vae.eval()
    loss = 0
    with torch.no_grad():
        for x in data_loader:
            x = x[0]
            recon_x, mu, log_var = vae(x)
            visualize_data(x.cpu().numpy(), './tmp/x.png')
            visualize_data(recon_x.cpu().numpy(), './tmp/x_recon.png')
            visualize_data(mu.cpu().numpy(), './tmp/z.png')
            loss += loss_function(recon_x, x, mu, log_var)
            break
    return loss


def generate(vae, num=1800):
    vae.eval()
    with torch.no_grad():
        z = torch.randn(num, z_dim).to(device)
        generated = vae.decoder(z).cpu().numpy()
    visualize_data(generated, './ckpt/vae_generated.png')
    return generated


def save_model(vae, path):
    torch.save(vae.state_dict(), path)


def load_model(vae, path):
    vae.load_state_dict(torch.load(path))
    return vae


def main(is_train=False):

    vae = VAE(input_dim, hidden_dim, z_dim,block_num).to(device)

    if not is_train:
        vae = load_model(vae, './ckpt/vae.pth').to(device)

    
    train_data = torch.from_numpy(
        np.load('./data/train.npy')).to(device=device, dtype=torch.float32)
    train_dataset = TensorDataset(train_data)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_data = torch.from_numpy(
        np.load('./data/test.npy')).to(device=device, dtype=torch.float32)
    test_dataset = TensorDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=1500, shuffle=False)
    if is_train:
        train(vae, train_loader,epochs=epochs)

    save_model(vae, './ckpt/vae.pth')
    validate(vae, test_loader)
    generate(vae)


set_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 1500
lr = 3e-4
block_num = 4
input_dim = 2
hidden_dim = 2048
z_dim = 2
epochs = 30000
main(is_train=True)
