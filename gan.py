import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from utils import set_seed, visualize_data, compute_kl_divergence_2d, compute_fid


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, block_num=4):
        super(Discriminator, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            *([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU()
            ]*block_num)
        )
        self.logits = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        hidden = self.feature(x)
        logits = torch.sigmoid(self.logits(hidden))
        return logits


class Generator(nn.Module):
    def __init__(self, z_dim, hidden_dim, output_dim, block_num=4):
        super(Generator, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.LeakyReLU(),
            *([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU()
            ]*block_num)
        )
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        hidden = self.feature(z)
        predicted = self.out(hidden)
        return predicted


def train(generator, discriminator, data_loader, epochs=100):

    adversarial_loss = torch.nn.BCELoss()

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=cfg.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=cfg.lr)

    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_G, T_max=epochs)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_D, T_max=epochs)

    generator.train()
    discriminator.train()
    for epoch in range(epochs):
        par = tqdm(data_loader, ncols=100)
        for x in par:
            x = x[0]
            valid = torch.ones(x.shape[0], 1).to(device)
            fake = torch.zeros(x.shape[0], 1).to(device)

            for _ in range(2):
                optimizer_G.zero_grad()
                z = torch.randn((x.shape[0], cfg.z_dim)).to(device)
                gen = generator(z)
                g_loss = adversarial_loss(discriminator(gen), valid)
                g_loss.backward()
                optimizer_G.step()
                scheduler_G.step()

            z = torch.randn((x.shape[0], cfg.z_dim)).to(device)
            optimizer_D.zero_grad()
            gen = generator(z).detach()
            real_loss = adversarial_loss(discriminator(x), valid)
            fake_loss = adversarial_loss(discriminator(gen), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
            scheduler_D.step()

            par.set_description(
                f"Epoch[{epoch + 1}/{epochs}], G-Loss: {g_loss.item()}, D-Loss: {d_loss.item()}")

        if epoch % 20 == 0:
            visualize_data(gen.cpu().detach().numpy(),
                           f'./tmp/gan_generated.png')
            save_model(generator, './ckpt/gan-generator.pth')
            save_model(discriminator, './ckpt/gan-discriminator.pth')


def generate(generator, num=1800):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num, cfg.z_dim).to(device)
        generated = generator(z)
    visualize_data(generated[:1800].cpu().numpy(),
                   './results/gan_generated.png')
    return generated.cpu()


def save_model(vae, path):
    torch.save(vae.state_dict(), path)


def load_model(vae, path):
    vae.load_state_dict(torch.load(path))
    return vae


def main():

    generator = Generator(cfg.z_dim, cfg.hidden_dim,
                          cfg.input_dim, cfg.block_num).to(device)
    discriminator = Discriminator(
        cfg.input_dim, cfg.hidden_dim, cfg.block_num).to(device)

    if cfg.inference_only:
        generator = load_model(
            generator, './ckpt/gan-generator.pth').to(device)
        discriminator = load_model(
            discriminator, './ckpt/gan-discriminator.pth').to(device)

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
        train(generator, discriminator, train_loader,
              epochs=cfg.epochs)

    save_model(generator, './ckpt/gan-generator.pth')
    save_model(discriminator, './ckpt/gan-discriminator.pth')

    generated = generate(generator).cpu().numpy()
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

    with open('./results/gan.txt', 'w') as f:
        f.write(f"FID: Train: {fid_train}, Val: {fid_val}, Test: {fid_test}\n")
        f.write(
            f"KL Divergence: Train: {kl_train}, Val: {kl_val}, Test: {kl_test}")


if __name__ == '__main__':
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('./configs/gan.yaml')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(cfg.seed)
    main()
