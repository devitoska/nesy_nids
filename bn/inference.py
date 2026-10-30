from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination
import pandas as pd

from pyjuice.nodes import CircuitNodes
from bn.pc_build import PCInference

# using exact algorithm Variable Elimination for inference
class InferenceEngine:

    def __init__(self, bn: DiscreteBayesianNetwork, class_values = None):
        self.bn = bn
        # possible values of the Class variable
        self.class_values = class_values
        self.infer = VariableElimination(bn)
        assert self.class_values is not None, "class_values must be provided"

    # predict the class label from a row of data
    # check if the observed values are valid in the BN, if not pop them
    # (invoke with data projected on the Markov blanket of the Class variable)
    def predict_from_row(self, row : pd.Series):
        # Infererence P(Class | evidence from row)
        evidence = row.to_dict()
        evidence.pop('class', None)  # remove Class from evidence if present, we want to predict it

        ### Check if observed values are valid in the BN, if not pop them 
        new_evidence = evidence.copy()
        for node, value in evidence.items():
            if value not in self.bn.get_cpds(node).state_names[node]:
                # pop invalid evidence => treat as unobserved
                new_evidence.pop(node)

        q = self.infer.query(variables=["class"], evidence=new_evidence)
        probs = q.values # numpy array of probabilities
        return self.class_values[int(probs.argmax())]

    # predict probabilities of the class variable from evidence (row of data)
    # evidence varies depending on the "unobserved" flag
    def predict_probs_from_evidence(self, evidence):
        q = self.infer.query(variables=["class"], evidence=evidence)
        return q.values # numpy array of probabilities
    
    # Compute explanation vector and predicted class
    def get_pred_expl(self, row: pd.Series, mb_list: list,
                            unobserved: bool = False) -> dict:
        
        evidence = row.to_dict()
        evidence.pop('class', None)  # remove Class from evidence if present, we want to predict it
        # order evidence dict by key as the order of mb_list
        evidence = {k: evidence[k] for k in mb_list if k in evidence}

        ### Check if observed values are valid in the BN, if not pop them
        evidence_ = evidence.copy()
        for node, value in evidence.items():
            if value not in self.bn.get_cpds(node).state_names[node]:
                evidence_.pop(node)
        
        post_probs = self.predict_probs_from_evidence(evidence_)
        target_value = self.class_values[int(post_probs.argmax())]  # predicted class
        target_index = self.class_values.index(target_value)
        p1 = post_probs[target_index]
        explanation_vec = [p1, ] # first element is the posterior probability of the target class

        for var in mb_list:
            if var not in evidence_:
                if unobserved:
                    # if unobserved is True and the variable is not observed, put -1 in the explanation vector
                    p2 = -1
                else:
                    # otherwise p2 = p1
                    p2 = p1
            else:
                # otherwise compute the posterior probability without the variable in the evidence
                evidence_no_var = evidence_.copy()
                evidence_no_var.pop(var, None)
                post_probs = self.predict_probs_from_evidence(evidence_no_var)
                p2 = post_probs[target_index]

            explanation_vec.append(p2)

        # Explanation vector is a list of posterior probabilities: 
        # e = [P(target|evidence), P(target|evidence - ev1), P(target|evidence - ev2), ...]
        return target_value, explanation_vec


class PCEngine:
    """InferenceEngine-compatible queries on an exact probabilistic circuit.

    Builds once per instance, uses PyJuice on CUDA when available, and evaluates
    the same circuit on CPU otherwise. Explanation variants are evaluated in a
    batch. A sequence of evidence dictionaries is also accepted by
    predict_probs_from_evidence, returning an array of shape (rows, classes).
    """

    def __init__(self, bn : DiscreteBayesianNetwork, root: CircuitNodes, class_values = None):
        self.bn = bn
        self.root = root
        self.class_values = class_values
        self.infer = PCInference(root)

    def predict_from_row(self, row : pd.Series):
        evidence = row.to_dict()
        evidence.pop('class', None)
        for node, value in list(evidence.items()):
            if value not in self.bn.get_cpds(node).state_names[node]:
                evidence.pop(node)
        probs = self.predict_probs_from_evidence(evidence)
        return self.class_values[int(probs.argmax())]

    def predict_probs_from_evidence(self, evidence):
        """Return a class-probability vector, or a matrix for a list of evidence dicts."""
        return self.infer.posterior(evidence, variable="class")

    def get_pred_expl(self, row: pd.Series, mb_list: list,
                            unobserved: bool = False) -> dict:
        return self._get_pred_expl_rows([row], mb_list, unobserved)[0]

    def get_pred_expl_batch(self, rows: pd.DataFrame, mb_list: list,
                            unobserved: bool = False) -> list:
        """Return (predicted class, explanation) for each row, in input order."""
        return self._get_pred_expl_rows(
            (row for _, row in rows.iterrows()), mb_list, unobserved
        )

    def _get_pred_expl_rows(self, rows, mb_list, unobserved):
        evidence_rows = []
        for row in rows:
            evidence = row.to_dict()
            evidence.pop('class', None)
            evidence = {k: evidence[k] for k in mb_list if k in evidence}
            for node, value in list(evidence.items()):
                if value not in self.bn.get_cpds(node).state_names[node]:
                    evidence.pop(node)
            evidence_rows.append(evidence)
        if not evidence_rows:
            return []

        # Select each row's target using its original evidence only. Duplicate
        # rows are handled by PCInference's query deduplication.
        base_probabilities = self.predict_probs_from_evidence(evidence_rows)
        targets = base_probabilities.argmax(axis=1)
        queries, states, row_queries = [], [], []
        for evidence, target_index in zip(evidence_rows, targets):
            query_index = {}
            for var in mb_list:
                if var in evidence and var not in query_index:
                    query_index[var] = len(queries)
                    queries.append({k: v for k, v in evidence.items() if k != var})
                    states.append(self.root.bn_state_names['class'][int(target_index)])
            row_queries.append(query_index)

        # Each retraction needs only the chosen class's joint probability and
        # the evidence likelihood, rather than the posterior of every class.
        probabilities = self.infer.posterior_for_states(queries, states, variable='class')
        results = []
        for row_index, (target_index, query_index) in enumerate(zip(targets, row_queries)):
            p1 = base_probabilities[row_index, target_index]
            explanation = [p1] + [
                probabilities[query_index[var]] if var in query_index
                else (-1 if unobserved else p1)
                for var in mb_list
            ]
            results.append((self.class_values[int(target_index)], explanation))
        return results
