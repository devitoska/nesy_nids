from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination
import pandas as pd

# using exact algorithm Variable Elimination for inference
class InferenceEngine:

    def __init__(self, bn: DiscreteBayesianNetwork, class_values = None):
        self.bn = bn
        # possible values of the Class variable
        self.class_values = class_values
        self.infer = VariableElimination(bn)
        assert self.class_values is not None, "class_values must be provided"
    
    def infer_from_row(self, row : pd.Series, black_list = None):
        # Infererence P(Class | evidence from row)
        evidence = row.to_dict()
        evidence.pop('class', None)  # remove Class from evidence if present, we want to predict it

        ### Check if observed values are valid in the BN, if not pop them 
        new_evidence = evidence.copy()
        for node, value in evidence.items():
            if value not in self.bn.get_cpds(node).state_names[node]:
                # pop invalid evidence => treat as unobserved
                new_evidence.pop(node)
        
        # if black_list is provided remove those variables from evidence
        if black_list is not None:
            new_evidence = {k: v for k, v in new_evidence.items() if k not in black_list}

        q = self.infer.query(variables=["class"], evidence=new_evidence)
        return q.values # numpy array of probabilities

    def predict_from_row(self, row : pd.Series):
        probs = self.infer_from_row(row)
        return self.class_values[int(probs.argmax())]

    # Comp explanation vector
    def get_explanation_vec(self, row: pd.Series, target_value: str, evidence_vars: list) -> dict:
        
        if target_value not in self.class_values:
            raise ValueError(f"target value {target_value} not in class values {self.class_values}")
        
        post_probs = self.infer_from_row(row)
        target_index = self.class_values.index(target_value)
        p1 = post_probs[target_index]
        explanation_vec = []

        for ev_var in evidence_vars:
            post_probs_no_ev = self.infer_from_row(row, black_list=[ev_var])
            p2 = post_probs_no_ev[target_index]
            eps = 1e-9  # to avoid division by zero
            # compute Relative Risk
            rr = p1 / (p2 + eps)
            explanation_vec.append(rr)
        
        '''
        #EXP: compute RR when all evidence variables are removed, to get a baseline RR without any evidence

        post_probs_no_all_ev = self.infer_from_row(row, black_list=evidence_vars)
        p2 = post_probs_no_all_ev[target_index]
        rr_all = p1 / (p2 + eps)
        explanation_vec.append(rr_all)
        '''

        return explanation_vec