import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from utils import set_seed, visualize_data, compute_kl_divergence_2d, compute_fid


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
    def __init__(self, z_dim, hidden_dim, output_dim, block_num=2):
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
        # if self.training:
        #     log_var = self.out_log_var(hidden)
        #     std = torch.exp(log_var / 2)
        #     eps = torch.randn_like(std)
        #     predicted = mu + eps * std
        # else:
        predicted = mu
        return predicted


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim, block_num=2):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim,
                               z_dim, block_num=block_num)
        self.decoder = Decoder(
            z_dim, hidden_dim, input_dim, block_num=block_num)

    def forward(self, x):
        z_mu, z_log_var = self.encoder(x)
        std = torch.exp(z_log_var / 2)
        eps = torch.randn_like(std)
        z = z_mu + eps * std
        return self.decoder(z), z_mu, z_log_var


def loss_function(recon_x, x, mu, log_var, beta=1.0):

    mse = F.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + log_var - mu*mu - torch.exp(log_var))
    return (mse+kld*beta)/x.shape[0]


def train(vae, data_loader, epochs=100):
    optimizer = torch.optim.AdamW(vae.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)
    vae.train()
    for epoch in range(epochs):
        par = tqdm(data_loader, ncols=100)
        for x in par:
            x = x[0]
            recon_x, mu, log_var = vae(x)
            loss = loss_function(recon_x, x, mu, log_var, beta=1.0)
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
            # visualize_data(x.cpu().numpy()[:1800], './tmp/x.png')
            # visualize_data(recon_x.cpu().numpy()[:1800], './tmp/x_recon.png')
            # visualize_data(mu.cpu().numpy()[:1800], './tmp/z.png')
            # loss += loss_function(recon_x, x, mu, log_var)
            loss += F.mse_loss(recon_x, x, reduction='sum')

    loss /= len(data_loader.dataset)
    return loss


def generate(vae, num=15000):
    vae.eval()
    with torch.no_grad():
        z = torch.randn(num, cfg.z_dim).to(device)
        generated = vae.decoder(z)
    visualize_data(generated[:1800].cpu().numpy(),
                   './results/vae_generated.png')
    return generated.cpu()


def save_model(vae, path):
    torch.save(vae.state_dict(), path)


def load_model(vae, path):
    vae.load_state_dict(torch.load(path))
    return vae


def main():

    vae = VAE(cfg.input_dim, cfg.hidden_dim,
              cfg.z_dim, cfg.block_num).to(device)

    if cfg.inference_only:
        vae = load_model(vae, './ckpt/vae.pth').to(device)

    train_data = torch.from_numpy(
        np.load('./data/train.npy')).to(device=device, dtype=torch.float32)
    val_data = torch.from_numpy(
        np.load('./data/val.npy')).to(device=device, dtype=torch.float32)
    test_data = torch.from_numpy(
        np.load('./data/test.npy')).to(device=device, dtype=torch.float32)

    train_dataset = TensorDataset(train_data)
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True)

    val_dataset = TensorDataset(val_data)
    val_loader = DataLoader(val_dataset, batch_size=15000, shuffle=False)

    test_dataset = TensorDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=15000, shuffle=False)

    if not cfg.inference_only:
        train(vae, train_loader, epochs=cfg.epochs)

    save_model(vae, './ckpt/vae.pth')

    # reconstruction loss
    train_loss = validate(vae, train_loader)
    val_loss = validate(vae, val_loader)
    test_loss = validate(vae, test_loader)
    print(f"Train: {train_loss}, Val: {val_loss}, Test: {test_loss}")

    generated = generate(vae).cpu().numpy()
    # FID
    fid_train = compute_fid(generated, train_data.cpu().numpy())
    fid_val = compute_fid(generated, val_data.cpu().numpy())
    fid_test = compute_fid(generated, test_data.cpu().numpy())
    print(f"Train: {fid_train}, Val: {fid_val}, Test: {fid_test}")

    # KL divergence
    kl_train = compute_kl_divergence_2d(generated, train_data.cpu().numpy())
    kl_val = compute_kl_divergence_2d(generated, val_data.cpu().numpy())
    kl_test = compute_kl_divergence_2d(generated, test_data.cpu().numpy())
    print(f"Train: {kl_train}, Val: {kl_val}, Test: {kl_test}")

    with open('./results/vae.txt', 'w') as f:
        f.write(
            f"Reconstrution loss: Train: {train_loss}, Val: {val_loss}, Test: {test_loss}\n")
        f.write(f"FID: Train: {fid_train}, Val: {fid_val}, Test: {fid_test}\n")
        f.write(
            f"KL Divergence: Train: {kl_train}, Val: {kl_val}, Test: {kl_test}")


if __name__ == '__main__':
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('./configs/vae.yaml')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(cfg.seed)
    main()
