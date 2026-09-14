import pickle
import torch
import torch.nn
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

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

    def __init__(self, input_dim = None):
        self.model = None
        self.epochs = 80
        self.batch_size = 32
        self.lr = 1e-4
        self.threshold = None
        self.input_dim = input_dim
        self.latent_dim = 4

    def train(self, data):
        #data = torch.tensor(data, dtype=torch.float32)
        train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
        train_data = torch.tensor(train_data, dtype=torch.float32)
        val_data = torch.tensor(val_data, dtype=torch.float32)

        # Training loop
        self.model = AENet(input_dim=train_data.shape[1], latent_dim=self.latent_dim)
        # use L2Loss for reconstruction loss
        loss_fn = torch.nn.MSELoss()
        optimizer =  torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        dataset = torch.utils.data.TensorDataset(train_data)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        
        #progress_bar = tqdm(range(epochs), desc="Training AE with L2 Loss")
        best_val_loss = float("inf")
        patience = 10
        counter = 0
        checkpoint_model = None

        for _ in range(self.epochs):
            self.model.train()
            total_loss = 0
            for batch in dataloader:
                x = batch[0]  # Extract the first element of the batch (the data)
                optimizer.zero_grad()
                x_recon = self.model(x)
                loss = loss_fn(x_recon, x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * x.size(0)
            
            #avg_loss = total_loss / len(dataloader.dataset)
            #progress_bar.set_postfix({"Avg Loss": f"{avg_loss:.4f}"})
            
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
                        # copy model object to checkpoint_model
                        checkpoint_model = AENet(input_dim=train_data.shape[1], latent_dim=self.latent_dim)
                        checkpoint_model.load_state_dict(self.model.state_dict())
                    else:
                        counter += 1
                    if counter >= patience:
                        #print(f"Early stopping at epoch {_+1}")
                        break
        
        self.model = checkpoint_model

        # get reconstruction error on the validation set
        self.model.eval()
        
        with torch.no_grad():
            X_cls = val_data
            X_recon = self.model(X_cls)
            recon_error = torch.mean((X_recon - X_cls) ** 2, dim=1).numpy()
            # get threshold as 99th percentile of reconstruction error
            self.threshold = np.percentile(recon_error, 99)
    
    def load(self, exp_name, unknown_cls, cls):
        self.model = AENet(input_dim=self.input_dim, latent_dim=self.latent_dim)
        self.model.load_state_dict(torch.load(f"results/{exp_name}/ad/no_{unknown_cls}/ae_{cls}.pth"))
        self.model.eval()
        self.threshold = pickle.load(open(f"results/{exp_name}/ad/no_{unknown_cls}/threshold_{cls}.pkl", "rb"))

    def save(self, exp_name, unknown_cls, class_name):
        torch.save(self.model.state_dict(), f"results/{exp_name}/ad/no_{unknown_cls}/ae_{class_name}.pth")
        with open(f"results/{exp_name}/ad/no_{unknown_cls}/threshold_{class_name}.pkl", "wb") as f:
            pickle.dump(self.threshold, f)
    
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

                x_recon = models[pred].model(x)
                recon_error = torch.mean((x_recon - x) ** 2).numpy()

                if recon_error > models[pred].threshold:
                    y_pred_bin.append(1)
                    y_pred_mul.append(unknown_cls)
                else:
                    y_pred_bin.append(0)
                    y_pred_mul.append(pred)
        
        return y_gt_bin, y_pred_bin, y_gt_mul, y_pred_mul

    @staticmethod
    def recon_error(models, data, preds):
        data = torch.tensor(data, dtype=torch.float32)
        
        recon_errors = []

        with torch.no_grad():
            # for each data in test
            for i in range(data.shape[0]):
                x = data[i].unsqueeze(0)
                pred = preds[i]

                x_recon = models[pred].model(x)
                recon_error = torch.mean((x_recon - x) ** 2).numpy()
                recon_errors.append(recon_error)
        
        return recon_errors