import pickle
import torch
import torch.nn
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from ad.utils import calc_anomaly_score, finalize_threshold

class AENet(torch.nn.Module):
    
    def __init__(self, input_dim, latent_dim, type = 1):
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
        )

        if type is not None and type % 3 == 1: # adding sigmoid for stability if type is 1 or 4 (output in [0,1]) 
            self.decoder.append(torch.nn.Sigmoid())

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon
    
class AE:

    def __init__(self, input_dim = None, rejection_rate = 0.01, 
                 EVT_rejection_rate = None, calibration_split = "same", 
                 loss = "l2", score = "mse", type = 1, device = "auto"):
        self.model = None
        self.loss = loss
        self.rejection_rate = rejection_rate
        self.EVT_rejection_rate = EVT_rejection_rate
        self.calibration_split = calibration_split
        self.score = score
        self.type = type
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
        self.num_exceedances = None # Number of exceedances above the threshold in the calibration set (for EVT diagnostics)

    def train(self, data, seed, only_load = False, correctness_mask=None):

        if correctness_mask is not None:
            correctness_mask = np.asarray(correctness_mask)
            if correctness_mask.shape != (len(data),) or correctness_mask.dtype.kind != "b":
                raise ValueError("correctness_mask must contain one Boolean per training sample.")
        if self.rejection_rate == "auto" and correctness_mask is None:
            raise ValueError("rejection_rate='auto' requires a correctness_mask for Youden calibration.")

        indices = np.arange(len(data))
        if self.calibration_split == "same":
            train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=seed)
            calib_indices = val_indices
        else:
            train_indices, heldout_indices = train_test_split(indices, test_size=0.3, random_state=seed)
            calib_indices, val_indices = train_test_split(heldout_indices, test_size=1/3, random_state=seed)
        
        calib_correctness = None if correctness_mask is None else correctness_mask[calib_indices]
        
        # strict requirement for Youden calibration: both correct and incorrect samples must be present in the calibration split
        '''
        if self.rejection_rate == "auto" and (not calib_correctness.any() or calib_correctness.all()):
            raise ValueError(
                "Youden calibration requires both correct and incorrect samples in the calibration split; "
                "include misclassified samples and ensure both groups are represented."
            )
        '''
        train_data = torch.tensor(data[train_indices], dtype=torch.float32, device=self.device)
        val_data = torch.tensor(data[val_indices], dtype=torch.float32, device=self.device)
        calib_data = torch.tensor(data[calib_indices], dtype=torch.float32, device=self.device)

        # Training loop
        self.model = AENet(input_dim=train_data.shape[1], latent_dim=self.latent_dim, type=self.type).to(self.device)

        if not only_load:
            if self.loss == "l2":
                loss_fn = torch.nn.MSELoss()
            elif self.loss == "huber":
                loss_fn = torch.nn.HuberLoss(delta=1.0)
            else:
                raise ValueError("Invalid loss function")

            optimizer =  torch.optim.Adam(self.model.parameters(), lr=self.lr)
            
            dataset = torch.utils.data.TensorDataset(train_data)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
            
            #progress_bar = tqdm(range(epochs), desc="Training AE with L2 Loss")
            best_val_loss = float("inf")
            patience = 10
            counter = 0
            eps = 1e-5
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
                        if val_loss + eps < best_val_loss:
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

        self.calibrate(calib_data, calib_correctness)

    def calibrate(self, calib_data, calib_correctness):
        # get anomaly score on the validation set
        self.model.eval()
        
        with torch.no_grad():
            X_cls = calib_data
            X_cls = X_cls.to(self.device)
            X_recon = self.model(X_cls)
            residuals = torch.abs(X_cls - X_recon).double()
            self.residual_mean = residuals.mean(dim=0)
            centered = residuals - self.residual_mean
            assert len(residuals) >= 2, "At least two validation samples are required to compute residual covariance."
            self.residual_cov = centered.T @ centered / (len(residuals) - 1)
            # Remove tiny numerical asymmetry.
            self.residual_cov = 0.5 * (
                self.residual_cov + self.residual_cov.T
            )
            anomaly_scores = calc_anomaly_score(X_cls, X_recon, score=self.score, 
                                                residual_mean=self.residual_mean, residual_cov=self.residual_cov)

            self.threshold, num_exceedances = finalize_threshold(
                anomaly_scores, self.rejection_rate, self.EVT_rejection_rate,
                correctness_mask=calib_correctness,
            )
            self.num_exceedances = num_exceedances

    def load(self, exp_name, unknown_cls, cls):
        self.model = AENet(input_dim=self.input_dim, latent_dim=self.latent_dim, type=self.type).to(self.device)
        state = torch.load(
            f"results/{exp_name}/ad/no_{unknown_cls}/ae_{cls}.pth",
            map_location="cpu", weights_only=True,
        )
        self.model.load_state_dict(state)
        self.model.eval()
        self.threshold = pickle.load(open(f"results/{exp_name}/ad/no_{unknown_cls}/threshold_{cls}.pkl", "rb"))
        self.residual_mean = pickle.load(open(f"results/{exp_name}/ad/no_{unknown_cls}/residual_mean_{cls}.pkl", "rb"))
        self.residual_cov = pickle.load(open(f"results/{exp_name}/ad/no_{unknown_cls}/residual_cov_{cls}.pkl", "rb"))
        self.num_exceedances = pickle.load(open(f"results/{exp_name}/ad/no_{unknown_cls}/num_exceedances_{cls}.pkl", "rb"))

    def save(self, exp_name, unknown_cls, class_name):
        torch.save(self.model.state_dict(), f"results/{exp_name}/ad/no_{unknown_cls}/ae_{class_name}.pth")
        with open(f"results/{exp_name}/ad/no_{unknown_cls}/threshold_{class_name}.pkl", "wb") as f:
            pickle.dump(self.threshold, f)
        with open(f"results/{exp_name}/ad/no_{unknown_cls}/residual_mean_{class_name}.pkl", "wb") as f:
            pickle.dump(self.residual_mean, f)
        with open(f"results/{exp_name}/ad/no_{unknown_cls}/residual_cov_{class_name}.pkl", "wb") as f:
            pickle.dump(self.residual_cov, f)
        with open(f"results/{exp_name}/ad/no_{unknown_cls}/num_exceedances_{class_name}.pkl", "wb") as f:
            pickle.dump(self.num_exceedances, f)

    @staticmethod
    def test(models, data, gts, preds, unknown_cls):
        """Score each predicted class together and return lists in input order."""
        data = torch.tensor(data, dtype=torch.float32)
        gts = np.asarray(gts)
        preds = np.asarray(preds)
        if data.ndim != 2:
            raise ValueError("Test data must have shape (samples, features).")
        if gts.shape != (len(data),) or preds.shape != (len(data),):
            raise ValueError("gts and preds must contain one label per test sample.")

        y_gt_bin = (gts == unknown_cls).astype(int)
        y_pred_bin = np.empty(len(data), dtype=int)
        # Object dtype accommodates an unknown label longer than known labels.
        y_pred_mul = preds.astype(object)
        ad_scores = np.empty(len(data), dtype=float)

        atol = 1e-2
        rtol = 1e-3

        with torch.no_grad():
            for pred in np.unique(preds):
                indices = np.flatnonzero(preds == pred)
                detector = models[pred]
                # Integer row indexing preserves (1, features) for singleton groups.
                x = data[torch.from_numpy(indices)].to(next(detector.model.parameters()).device)
                x_recon = detector.model(x)
                scores = calc_anomaly_score(
                    x, x_recon, score=detector.score,
                    residual_mean=detector.residual_mean, residual_cov=detector.residual_cov,
                )
                threshold = detector.threshold
                tolerance = 0
                #tolerance = atol + rtol * abs(threshold)
                rejected = scores > threshold + tolerance
                ad_scores[indices] = scores
                y_pred_bin[indices] = rejected.astype(int)
                y_pred_mul[indices[rejected]] = unknown_cls

        return (
            y_gt_bin.tolist(), y_pred_bin.tolist(), gts.tolist(),
            y_pred_mul.tolist(), ad_scores.tolist(),
        )
