import pickle
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
        y_gt_bin = []
        y_pred_bin = []
        y_gt_mul = []
        y_pred_mul = []
        ad_scores = []

        # for each data in test
        for i in range(data.shape[0]):
            x = data[i].reshape(1, -1)
            gt = gts[i]
            pred = preds[i]

            y_gt_mul.append(gt)
            y_gt_bin.append(1 if gt == unknown_cls else 0)

            out = models[pred].model.predict(x)
            anomaly_score = models[pred].model.decision_function(x)

            y_pred_bin.append(1 if out[0] == -1 else 0)
            y_pred_mul.append(unknown_cls if out[0] == -1 else pred)
            ad_scores.append(anomaly_score[0])

        return y_gt_bin, y_pred_bin, y_gt_mul, y_pred_mul, ad_scores