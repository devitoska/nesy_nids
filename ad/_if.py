import pickle
import numpy as np
from sklearn.ensemble import IsolationForest

class IF:

    def __init__(self, rejection_rate=0.01, **kwargs):
        self.model = None
        self.rejection_rate = rejection_rate

    def train(self, data, seed):
        self.model = IsolationForest(contamination=self.rejection_rate, random_state=seed).fit(data)
    
    def load(self, exp_name, unknown_cls, cls):
        with open(f"results/{exp_name}/ad/no_{unknown_cls}/if_{cls}.pkl", "rb") as f:
            self.model = pickle.load(f)

    def save(self, exp_name, unknown_cls, cls):
        with open(f"results/{exp_name}/ad/no_{unknown_cls}/if_{cls}.pkl", "wb") as f:
            pickle.dump(self.model, f)

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
            scores = models[pred].model.decision_function(data[indices])
            # IsolationForest.predict rejects negative decision scores. Reuse
            # these scores instead of traversing the forest a second time.
            rejected = scores < 0
            ad_scores[indices] = scores
            y_pred_bin[indices] = rejected.astype(int)
            y_pred_mul[indices[rejected]] = unknown_cls

        return (
            y_gt_bin.tolist(), y_pred_bin.tolist(), gts.tolist(),
            y_pred_mul.tolist(), ad_scores.tolist(),
        )
