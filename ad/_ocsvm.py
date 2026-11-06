import pickle

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM

from ad.utils import finalize_threshold


class OCSVM:
    """RBF One-Class SVM with held-out threshold calibration.

    Scores are negated decision values, so larger values indicate anomalies.
    ``nu`` controls fitting independently of the calibrated rejection rate.
    AE-specific constructor options are accepted through kwargs and ignored.
    """

    def __init__(self, input_dim=None, rejection_rate=0.01,
                 EVT_rejection_rate=None, calibration_split="same",
                 nu=0.01, kernel="rbf", gamma="scale", **kwargs):
        self.model = None
        self.input_dim = input_dim
        self.rejection_rate = rejection_rate
        self.EVT_rejection_rate = EVT_rejection_rate
        self.calibration_split = calibration_split
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.threshold = None
        self.num_exceedances = None

    def train(self, data, seed, only_load=False, correctness_mask=None):
        data = np.asarray(data)
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
            train_indices, calib_indices = train_test_split(indices, test_size=0.2, random_state=seed)
        elif self.calibration_split == "separate":
            # Match AE's training/calibration partitions; SVM needs no validation set.
            train_indices, heldout_indices = train_test_split(indices, test_size=0.3, random_state=seed)
            calib_indices, _ = train_test_split(heldout_indices, test_size=1/3, random_state=seed)
        else:
            raise ValueError("calibration_split must be 'same' or 'separate'.")

        if not only_load:
            self.input_dim = data.shape[1]
            self.model = OneClassSVM(nu=self.nu, kernel=self.kernel, gamma=self.gamma).fit(data[train_indices])
        elif self.model is None:
            raise RuntimeError("Load a fitted OCSVM before using only_load=True.")

        calib_correctness = None if correctness_mask is None else correctness_mask[calib_indices]
        self.calibrate(data[calib_indices], calib_correctness)

    def calibrate(self, calib_data, calib_correctness=None):
        scores = -self.model.decision_function(calib_data)
        self.threshold, self.num_exceedances = finalize_threshold(
            scores, self.rejection_rate, self.EVT_rejection_rate,
            correctness_mask=calib_correctness,
        )

    def load(self, exp_name, unknown_cls, cls):
        path = f"results/{exp_name}/ad/no_{unknown_cls}"
        with open(f"{path}/ocsvm_{cls}.pkl", "rb") as f:
            self.model = pickle.load(f)
        self.input_dim = self.model.n_features_in_
        with open(f"{path}/threshold_{cls}.pkl", "rb") as f:
            self.threshold = pickle.load(f)
        with open(f"{path}/num_exceedances_{cls}.pkl", "rb") as f:
            self.num_exceedances = pickle.load(f)

    def save(self, exp_name, unknown_cls, cls):
        path = f"results/{exp_name}/ad/no_{unknown_cls}"
        with open(f"{path}/ocsvm_{cls}.pkl", "wb") as f:
            pickle.dump(self.model, f)
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
            scores = -detector.model.decision_function(data[indices])
            rejected = scores > detector.threshold
            ad_scores[indices] = scores
            y_pred_bin[indices] = rejected.astype(int)
            y_pred_mul[indices[rejected]] = unknown_cls

        return (
            y_gt_bin.tolist(), y_pred_bin.tolist(), gts.tolist(),
            y_pred_mul.tolist(), ad_scores.tolist(),
        )
