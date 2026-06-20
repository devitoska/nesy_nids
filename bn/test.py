from pgmpy.models import DiscreteBayesianNetwork
from tqdm import tqdm
import json
import os
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

from bn.inference import InferenceEngine

def test_bn(bn : DiscreteBayesianNetwork, data : pd.DataFrame, base_path : str, unknown_class : str):
    class_values = bn.get_cpds("class").state_names["class"]
    inference_engine = InferenceEngine(bn, class_values=class_values)
    preds = []
    gts = []
    progress_bar = tqdm(total=len(data), desc="Testing BN")
    
    for _, row in data.iterrows():
        predicted = inference_engine.predict_from_row(row)
        ground_truth = row['class']
        preds.append(predicted)
        gts.append(ground_truth)
        progress_bar.update(1)
    progress_bar.close()
    
    metrics = {}

    # Classification report
    cr = classification_report(gts, preds, output_dict=True)
    metrics["classification_report"] = cr
    cf = confusion_matrix(gts, preds)
    metrics["confusion_matrix"] = cf.tolist()

    # Save metrics to json
    with open(os.path.join(base_path, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)