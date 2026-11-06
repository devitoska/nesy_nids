import pickle

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from ad.utils import finalize_threshold


class DeepSVDDNet(torch.nn.Module):
    """Bias-free encoder with unbounded activations to avoid trivial collapse."""

    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 16, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(8, latent_dim, bias=False),
        )

    def forward(self, x):
        return self.encoder(x)


class DeepSVDD:
    """One-class Deep SVDD with a fixed center and calibrated squared distances.

    Uses the one-class objective from Ruff et al. (2018), with weight decay.
    AE-specific loss, score and type options are accepted but ignored: both
    training and scoring always use squared Euclidean distance to the center.
    """

    def __init__(self, input_dim=None, rejection_rate=0.01,
                 EVT_rejection_rate=None, calibration_split="same",
                 device="auto", **kwargs):
        self.model = None
        self.input_dim = input_dim
        self.rejection_rate = rejection_rate
        self.EVT_rejection_rate = EVT_rejection_rate
        self.calibration_split = calibration_split
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ) if device == "auto" else torch.device(device)
        self.latent_dim = 4
        self.epochs = 80
        self.batch_size = 32
        self.lr = 1e-4
        self.weight_decay = 1e-6
        self.center = None
        self.threshold = None
        self.num_exceedances = None

    def train(self, data, seed, only_load=False, correctness_mask=None):
        data = np.asarray(data, dtype=np.float32)
        if data.ndim != 2 or data.shape[1] == 0 or not np.isfinite(data).all():
            raise ValueError("Training data must be a finite array of shape (samples, features).")
        if correctness_mask is not None:
            correctness_mask = np.asarray(correctness_mask)
            if correctness_mask.shape != (len(data),) or correctness_mask.dtype.kind != "b":
                raise ValueError("correctness_mask must contain one Boolean per training sample.")
        if (self.rejection_rate == "auto" or self.EVT_rejection_rate == "auto") and correctness_mask is None:
            raise ValueError("Automatic calibration requires a correctness_mask.")

        indices = np.arange(len(data))
        if self.calibration_split == "same":
            train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=seed)
            calib_indices = val_indices
        elif self.calibration_split == "separate":
            train_indices, heldout_indices = train_test_split(indices, test_size=0.3, random_state=seed)
            calib_indices, val_indices = train_test_split(heldout_indices, test_size=1/3, random_state=seed)
        else:
            raise ValueError("calibration_split must be 'same' or 'separate'.")

        if not only_load:
            torch.manual_seed(seed)
            self.input_dim = data.shape[1]
            self.model = DeepSVDDNet(self.input_dim, self.latent_dim).to(self.device)
            train_data = torch.from_numpy(data[train_indices])
            dataset = torch.utils.data.TensorDataset(train_data)
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True,
                generator=torch.Generator().manual_seed(seed),
            )

            # Initialize only from the training partition and keep the center fixed.
            self.model.eval()
            with torch.no_grad():
                self.center = torch.zeros(self.latent_dim, device=self.device)
                for batch in train_data.split(self.batch_size):
                    self.center += self.model(batch.to(self.device)).sum(dim=0)
                self.center /= len(train_data)
                # Push near-zero coordinates (including exact zeros) away from zero.
                self.center = torch.where(
                    self.center.abs() < 0.1,
                    torch.where(self.center < 0, -0.1, 0.1), self.center,
                )

            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
            )
            best_val_loss = float("inf")
            best_state = None
            counter = 0
            for _ in range(self.epochs):
                self.model.train()
                for (batch,) in dataloader:
                    optimizer.zero_grad()
                    outputs = self.model(batch.to(self.device))
                    loss = (outputs - self.center).square().sum(dim=1).mean()
                    loss.backward()
                    optimizer.step()

                val_loss = self._scores(data[val_indices]).mean()
                if val_loss + 1e-5 < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {
                        name: tensor.detach().clone()
                        for name, tensor in self.model.state_dict().items()
                    }
                    counter = 0
                else:
                    counter += 1
                if counter >= 10:
                    break

            if best_state is None:
                raise RuntimeError("Deep SVDD training did not produce a valid validation checkpoint.")
            self.model.load_state_dict(best_state)
        elif self.model is None or self.center is None:
            raise RuntimeError("Load a fitted DeepSVDD before using only_load=True.")

        calib_correctness = None if correctness_mask is None else correctness_mask[calib_indices]
        self.calibrate(data[calib_indices], calib_correctness)

    def _scores(self, data):
        self.model.eval()
        data = torch.as_tensor(data, dtype=torch.float32)
        scores = np.empty(len(data), dtype=float)
        with torch.no_grad():
            for start in range(0, len(data), self.batch_size):
                batch = data[start:start + self.batch_size].to(self.device)
                distances = (self.model(batch) - self.center).square().sum(dim=1)
                scores[start:start + len(batch)] = distances.cpu().numpy()
        return scores

    def calibrate(self, calib_data, calib_correctness=None):
        self.threshold, self.num_exceedances = finalize_threshold(
            self._scores(calib_data), self.rejection_rate, self.EVT_rejection_rate,
            correctness_mask=calib_correctness,
        )

    def load(self, exp_name, unknown_cls, cls):
        path = f"results/{exp_name}/ad/no_{unknown_cls}"
        state = torch.load(f"{path}/deepsvdd_{cls}.pth", map_location="cpu", weights_only=True)
        self.input_dim = state["input_dim"]
        self.latent_dim = state["latent_dim"]
        self.model = DeepSVDDNet(self.input_dim, self.latent_dim).to(self.device)
        self.model.load_state_dict(state["model"])
        self.model.eval()
        self.center = state["center"].to(self.device)
        with open(f"{path}/threshold_{cls}.pkl", "rb") as f:
            self.threshold = pickle.load(f)
        with open(f"{path}/num_exceedances_{cls}.pkl", "rb") as f:
            self.num_exceedances = pickle.load(f)

    def save(self, exp_name, unknown_cls, cls):
        path = f"results/{exp_name}/ad/no_{unknown_cls}"
        torch.save({
            "model": {name: tensor.detach().cpu() for name, tensor in self.model.state_dict().items()},
            "center": self.center.detach().cpu(),
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
        }, f"{path}/deepsvdd_{cls}.pth")
        with open(f"{path}/threshold_{cls}.pkl", "wb") as f:
            pickle.dump(self.threshold, f)
        with open(f"{path}/num_exceedances_{cls}.pkl", "wb") as f:
            pickle.dump(self.num_exceedances, f)

    @staticmethod
    def test(models, data, gts, preds, unknown_cls):
        """Score each predicted class together and return lists in input order."""
        data = np.asarray(data)
        gts = np.asarray(gts)
        preds = np.asarray(preds)
        if data.ndim != 2:
            raise ValueError("Test data must have shape (samples, features).")
        if gts.shape != (len(data),) or preds.shape != (len(data),):
            raise ValueError("gts and preds must contain one label per test sample.")

        y_gt_bin = (gts == unknown_cls).astype(int)
        y_pred_bin = np.empty(len(data), dtype=int)
        y_pred_mul = preds.astype(object)
        ad_scores = np.empty(len(data), dtype=float)

        for pred in np.unique(preds):
            indices = np.flatnonzero(preds == pred)
            detector = models[pred]
            scores = detector._scores(data[indices])
            rejected = scores > detector.threshold
            ad_scores[indices] = scores
            y_pred_bin[indices] = rejected.astype(int)
            y_pred_mul[indices[rejected]] = unknown_cls

        return (
            y_gt_bin.tolist(), y_pred_bin.tolist(), gts.tolist(),
            y_pred_mul.tolist(), ad_scores.tolist(),
        )
