from tqdm import tqdm
import json
import os
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

from bn.inference import InferenceEngine, PCEngine
from bn.utils import load_bn
from bn.pc_inf import load_pc

def test_bn(exp_name, config, data_path):
    inference = config.get("inference", "pgmpy")
    partitions = sorted(
        name for name in os.listdir(data_path)
        if name.startswith("no_") and os.path.isdir(os.path.join(data_path, name))
    )
    if not partitions:
        raise ValueError(f"No held-out-class partitions found in {data_path}")

    for partition in partitions:
        cls = partition.removeprefix("no_")
        partition_path = os.path.join(data_path, partition)
        test_data = pd.read_csv(
            os.path.join(partition_path, "test_data.csv"), dtype=str,
        )
        base_save_path = f"results/{exp_name}/bn/no_{cls}"
        bn = load_bn(base_save_path)
        mb_list = sorted(bn.get_markov_blanket("class"))
        data = test_data[mb_list + ["class"]]
        class_values = bn.get_cpds("class").state_names["class"]

        batched = inference != "pgmpy"
        if not batched:
            inference_engine = InferenceEngine(bn, class_values=class_values)
        else:
            root = load_pc(os.path.join(base_save_path, "pc_root.pkl"))
            inference_engine = PCEngine(bn, root, class_values=class_values)

        preds = []
        gts = []
        progress_bar = tqdm(total=len(data), desc="Testing BN")

        if batched:
            batch_size = 50_000
            for start in range(0, len(data), batch_size):
                batch = data.iloc[start:start + batch_size]
                queries = []
                for _, row in batch.iterrows():
                    evidence = row.to_dict()
                    evidence.pop('class', None)
                    for node, value in list(evidence.items()):
                        if value not in inference_engine.bn.get_cpds(node).state_names[node]:
                            evidence.pop(node)
                    queries.append(evidence)

                probabilities = inference_engine.predict_probs_from_evidence(queries)
                preds.extend(
                    inference_engine.class_values[int(index)]
                    for index in probabilities.argmax(axis=1)
                )
                gts.extend(batch['class'].tolist())
                progress_bar.update(len(batch))
        else:
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
        with open(os.path.join(base_save_path, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
