import pickle
import torch
import torch.nn
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from ad.utils import plot_losses

@dataclass
class VAEOutput:
    z: torch.Tensor
    mu: torch.Tensor
    std: torch.Tensor
    x_recon: torch.Tensor
    loss: torch.Tensor
    loss_recon: torch.Tensor
    loss_kl: torch.Tensor

class VAENet(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        hidden_dim = 16
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_std = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        # Softplus + epsilon for stable std deviation
        std = F.softplus(self.fc_std(h)) + 1e-6
        return mu, std

    def reparameterize(self, mu, std):
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, kl_weight=1.0):
        mu, std = self.encode(x)
        z = self.reparameterize(mu, std)
        x_recon = self.decode(z)

        # 1. Reconstruction Loss
        # Sum over features, mean over batch
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')

        # 2. KL Divergence
        # Analytic KL for Normal distributions
        kl_loss = -0.5 * torch.sum(1 + torch.log(std**2) - mu**2 - std**2, dim=1).mean()

        # 3. Total Loss (ELBO)
        loss = recon_loss + (kl_weight * kl_loss)

        return VAEOutput(z, mu, std, x_recon, loss, recon_loss, kl_loss)


class VAE:

    def __init__(self, input_dim = None, rejection_rate = 0.01, device = "auto", **kwargs):
        self.model = None
        self.rejection_rate = rejection_rate
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ) if device == "auto" else torch.device(device)
        self.epochs = 80
        self.batch_size = 32
        self.lr = 1e-4
        self.threshold = None
        self.input_dim = input_dim
        self.latent_dim = 4

        self.train_losses = []
        self.train_recon_losses = []
        self.train_kl_losses = []

        self.val_losses = []
        self.val_recon_losses = []
        self.val_kl_losses = []

    def train(self, data, seed):
        data = torch.tensor(data, dtype=torch.float32)
        train_data, val_data = train_test_split(data, test_size=0.2, random_state=seed)
        train_data = torch.tensor(train_data, dtype=torch.float32, device=self.device)
        val_data = torch.tensor(val_data, dtype=torch.float32, device=self.device)
        kl_weight = 1.0

        # Training loop
        self.model = VAENet(input_dim=train_data.shape[1], latent_dim=self.latent_dim).to(self.device)
        
        # use L2Loss for reconstruction loss
        optimizer =  torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        dataset = torch.utils.data.TensorDataset(train_data)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        
        #progress_bar = tqdm(range(epochs), desc="Training AE with L2 Loss")
        best_val_loss = float("inf")
        patience = 10
        counter = 0
        best_state = None

        for _ in range(self.epochs):
            self.model.train()
            avg_loss = 0.0
            avg_recon_loss = 0.0
            avg_kl_loss = 0.0

            for batch in dataloader:
                x = batch[0]  # Extract the first element of the batch (the data)
                optimizer.zero_grad()
                # Forward pass
                output = self.model(x, kl_weight)
                # Backward pass
                output.loss.backward()
                avg_loss += output.loss.item()
                avg_recon_loss += output.loss_recon.item()
                avg_kl_loss += output.loss_kl.item()
                # Gradient clipping (recommended)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
            
            avg_loss /= len(dataloader)
            avg_recon_loss /= len(dataloader)
            avg_kl_loss /= len(dataloader)
            self.train_losses.append(avg_loss)
            self.train_recon_losses.append(avg_recon_loss)
            self.train_kl_losses.append(avg_kl_loss)

            self.model.eval()
            
            if val_data is not None:
                with torch.no_grad():
                    output = self.model(val_data, kl_weight)
                    val_loss = output.loss.item()
                    val_recon_loss = output.loss_recon.item()
                    val_kl_loss = output.loss_kl.item()
                    self.val_losses.append(val_loss)
                    self.val_recon_losses.append(val_recon_loss)
                    self.val_kl_losses.append(val_kl_loss)
                    # early stopping
                    if val_loss + 1e-5 < best_val_loss:
                        best_val_loss = val_loss
                        counter = 0
                        best_state = {
                            name: tensor.detach().clone()
                            for name, tensor in self.model.state_dict().items()
                        }
                    else:
                        counter += 1
                    if counter >= patience:
                        break
        
        if best_state is None:
            raise RuntimeError("VAE training did not produce a valid validation checkpoint")
        self.model.load_state_dict(best_state)

        # get reconstruction error on the train set
        self.model.eval()
        
        with torch.no_grad():
            X_cls = val_data
            output = self.model(X_cls, kl_weight)
            recon_error = F.mse_loss(output.x_recon, X_cls, reduction='mean').detach().cpu().numpy()
            self.threshold = np.percentile(recon_error, 100 * (1 - self.rejection_rate))
    
    def load(self, exp_name, unknown_cls, cls):
        self.model = VAENet(input_dim=self.input_dim, latent_dim=self.latent_dim).to(self.device)
        state = torch.load(
            f"results/{exp_name}/ad/no_{unknown_cls}/vae_{cls}.pth",
            map_location="cpu", weights_only=True,
        )
        self.model.load_state_dict(state)
        self.model.eval()
        self.threshold = pickle.load(open(f"results/{exp_name}/ad/no_{unknown_cls}/threshold_{cls}.pkl", "rb"))

    def save(self, exp_name, unknown_cls, class_name):
        torch.save(self.model.state_dict(), f"results/{exp_name}/ad/no_{unknown_cls}/vae_{class_name}.pth")
        with open(f"results/{exp_name}/ad/no_{unknown_cls}/threshold_{class_name}.pkl", "wb") as f:
                pickle.dump(self.threshold, f)
        plot_losses(self.train_losses, self.val_losses, 
                    self.train_recon_losses, self.val_recon_losses,
                    self.train_kl_losses, self.val_kl_losses,
                    dir=f"results/{exp_name}/ad/no_{unknown_cls}/", class_name=class_name)
    
    @staticmethod
    def test(models, data, gts, preds, unknown_cls):
        data = torch.tensor(data, dtype=torch.float32)
        y_gt_bin = []
        y_pred_bin = []
        y_gt_mul = []
        y_pred_mul = []
        ad_scores = []

        with torch.no_grad():
            # for each data in test
            for i in range(data.shape[0]):
                x = data[i].unsqueeze(0)
                gt = gts[i]
                pred = preds[i]

                y_gt_mul.append(gt)
                y_gt_bin.append(1 if gt == unknown_cls else 0)

                x = x.to(next(models[pred].model.parameters()).device)
                output = models[pred].model(x)
                recon_error = torch.mean((output.x_recon - x) ** 2).detach().cpu().numpy()

                if recon_error > models[pred].threshold:
                    y_pred_bin.append(1)
                    y_pred_mul.append(unknown_cls)
                else:
                    y_pred_bin.append(0)
                    y_pred_mul.append(pred)
        
        return y_gt_bin, y_pred_bin, y_gt_mul, y_pred_mul, ad_scores

    @staticmethod
    def recon_error(models, data, preds):
        data = torch.tensor(data, dtype=torch.float32)
        
        recon_errors = []

        with torch.no_grad():
            # for each data in test
            for i in range(data.shape[0]):
                x = data[i].unsqueeze(0)
                pred = preds[i]

                x = x.to(next(models[pred].model.parameters()).device)
                output = models[pred].model(x)
                recon_error = torch.mean((output.x_recon - x) ** 2).detach().cpu().numpy()
                recon_errors.append(recon_error)
        
        return recon_errors
