import pickle
import torch
import torch.nn
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from ad.utils import calc_anomaly_score

class AENet(torch.nn.Module):
    
    def __init__(self, input_dim, latent_dim):
        super(AENet, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 16),
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU(),
            torch.nn.Linear(16,8),
            torch.nn.BatchNorm1d(8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, latent_dim),
        )
        
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 8),
            torch.nn.BatchNorm1d(8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 16),
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, input_dim),
            #torch.nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon
    
class AE:

    def __init__(self, input_dim = None, rejection_rate = 0.01, score = "mse", device = "auto"):
        self.model = None
        self.rejection_rate = rejection_rate
        self.score = score
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ) if device == "auto" else torch.device(device)
        self.epochs = 80
        self.batch_size = 32
        self.lr = 1e-4
        self.threshold = None
        self.residual_mean = None
        self.residual_cov = None
        self.input_dim = input_dim
        self.latent_dim = 4

    def train(self, data, seed):
        #data = torch.tensor(data, dtype=torch.float32)
        train_data, val_data = train_test_split(data, test_size=0.2, random_state=seed)
        train_data = torch.tensor(train_data, dtype=torch.float32, device=self.device)
        val_data = torch.tensor(val_data, dtype=torch.float32, device=self.device)

        # Training loop
        self.model = AENet(input_dim=train_data.shape[1], latent_dim=self.latent_dim).to(self.device)
        # use L2Loss for reconstruction loss
        loss_fn = torch.nn.MSELoss()
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
            for batch in dataloader:
                x = batch[0]  # Extract the first element of the batch (the data)
                optimizer.zero_grad()
                x_recon = self.model(x)
                loss = loss_fn(x_recon, x)
                loss.backward()
                optimizer.step()
            
            
            self.model.eval()
            
            if val_data is not None:
                with torch.no_grad():
                    x_val = val_data
                    x_val_recon = self.model(x_val)
                    val_loss = loss_fn(x_val_recon, x_val).item()
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
                        #print(f"Early stopping at epoch {_+1}")
                        break
        
        if best_state is None:
            raise RuntimeError("AE training did not produce a valid validation checkpoint")
        self.model.load_state_dict(best_state)

        # get anomaly score on the validation set
        self.model.eval()
        
        with torch.no_grad():
            X_cls = val_data
            X_recon = self.model(X_cls)
            residuals = torch.abs(X_cls - X_recon)
            self.residual_mean = residuals.mean(dim=0)
            centered = residuals - self.residual_mean
            self.residual_cov = centered.T @ centered / (len(residuals) - 1)
            anomaly_scores = calc_anomaly_score(X_cls, X_recon, score=self.score, 
                                                residual_mean=self.residual_mean, residual_cov=self.residual_cov)
            self.threshold = np.percentile(anomaly_scores, 100 * (1 - self.rejection_rate))
    
    def load(self, exp_name, unknown_cls, cls):
        self.model = AENet(input_dim=self.input_dim, latent_dim=self.latent_dim).to(self.device)
        state = torch.load(
            f"results/{exp_name}/ad/no_{unknown_cls}/ae_{cls}.pth",
            map_location="cpu", weights_only=True,
        )
        self.model.load_state_dict(state)
        self.model.eval()
        self.threshold = pickle.load(open(f"results/{exp_name}/ad/no_{unknown_cls}/threshold_{cls}.pkl", "rb"))
        self.residual_mean = pickle.load(open(f"results/{exp_name}/ad/no_{unknown_cls}/residual_mean_{cls}.pkl", "rb"))
        self.residual_cov = pickle.load(open(f"results/{exp_name}/ad/no_{unknown_cls}/residual_cov_{cls}.pkl", "rb"))

    def save(self, exp_name, unknown_cls, class_name):
        torch.save(self.model.state_dict(), f"results/{exp_name}/ad/no_{unknown_cls}/ae_{class_name}.pth")
        with open(f"results/{exp_name}/ad/no_{unknown_cls}/threshold_{class_name}.pkl", "wb") as f:
            pickle.dump(self.threshold, f)
        with open(f"results/{exp_name}/ad/no_{unknown_cls}/residual_mean_{class_name}.pkl", "wb") as f:
            pickle.dump(self.residual_mean, f)
        with open(f"results/{exp_name}/ad/no_{unknown_cls}/residual_cov_{class_name}.pkl", "wb") as f:
            pickle.dump(self.residual_cov, f)

    @staticmethod
    def test(models, data, gts, preds, unknown_cls):
        data = torch.tensor(data, dtype=torch.float32)
        
        y_gt_bin = []
        y_pred_bin = []
        y_gt_mul = []
        y_pred_mul = []

        with torch.no_grad():
            # for each data in test
            for i in range(data.shape[0]):
                x = data[i].unsqueeze(0)
                gt = gts[i]
                pred = preds[i]

                y_gt_mul.append(gt)
                y_gt_bin.append(1 if gt == unknown_cls else 0)

                x = x.to(next(models[pred].model.parameters()).device)
                x_recon = models[pred].model(x)
                anomaly_score = calc_anomaly_score(x, x_recon, score=models[pred].score, 
                                                   residual_mean=models[pred].residual_mean, residual_cov=models[pred].residual_cov)

                if anomaly_score > models[pred].threshold:
                    y_pred_bin.append(1)
                    y_pred_mul.append(unknown_cls)
                else:
                    y_pred_bin.append(0)
                    y_pred_mul.append(pred)
        
        return y_gt_bin, y_pred_bin, y_gt_mul, y_pred_mul

    @staticmethod
    def get_anomaly_scores(models, data, preds):
        data = torch.tensor(data, dtype=torch.float32)
        anomaly_scores = []
        with torch.no_grad():
            # for each data in test
            for i in range(data.shape[0]):
                x = data[i].unsqueeze(0)
                pred = preds[i]
                x = x.to(next(models[pred].model.parameters()).device)
                x_recon = models[pred].model(x)
                anomaly_score = calc_anomaly_score(x, x_recon, score=models[pred].score,
                                                   residual_mean=models[pred].residual_mean, residual_cov=models[pred].residual_cov)
                anomaly_scores.append(anomaly_score.item())
        
        return anomaly_scores