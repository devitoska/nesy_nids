from pgmpy.readwrite import XMLBIFReader
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import ExpertKnowledge
import os
import pandas as pd
from IPython.display import Image

from bn.structure_learning import learn_structure
from bn.parameter_learning import estimate_parameters

def print_bn_info(bn, base_path : str = None, 
                  search_strategy: str = None,
                  structure_learning_params: dict = None,
                  estimator_type: str = None,
                  parameter_learning_params: dict = None,
                  console: bool = False):

    output = f"""Bayesian Network Params:
            Structure Learning Method: {search_strategy}
            Structure Learning Parameters: {structure_learning_params}
            Parameter Learning Method: {estimator_type}
            Parameter Learning Parameters: {parameter_learning_params}
            """

    # Number of nodes
    num_nodes = len(bn.nodes())
    # Number of edges
    num_edges = len(bn.edges())
    # Max number of parents
    max_parents = max([len(bn.get_parents(node)) for node in bn.nodes()])
    # Average number of parents
    avg_parents = sum([len(bn.get_parents(node)) for node in bn.nodes()]) / num_nodes
    # Max number of children
    max_children = max([len(bn.get_children(node)) for node in bn.nodes()])
    # Number of root nodes
    num_root_nodes = len([node for node in bn.nodes() if len(bn.get_parents(node)) == 0])
    # Number of leaf nodes
    num_leaf_nodes = len([node for node in bn.nodes() if len(bn.get_children(node)) == 0])
    
    output += f"""Bayesian Network Model Info:
                Number of nodes: {num_nodes}
                Number of edges: {num_edges}
                Max number of parents: {max_parents}
                Average number of parents: {avg_parents:.2f}
                Max number of children: {max_children}
                Number of root nodes: {num_root_nodes}
                Number of leaf nodes: {num_leaf_nodes}
                Model Edges:
                """
    for edge in bn.edges():
        output += f"{edge[0]} -> {edge[1]}\n"

    if console:
        print(output)
    else:
        # Save model info to a text file
        file_path = os.path.join(base_path, "bn_model_info.txt")
        with open(file_path, 'w') as f:
            f.write(output)

def save_bn_image(bn, base_path, file_name: str = "bn_model.png"):
    file_path = os.path.join(base_path, file_name)
    viz = bn.to_graphviz()
    viz.draw(file_path, prog='dot') # dot layout => bipartite graph
    Image(file_path)

def save_bn_model(bn, base_path):
    file_path = os.path.join(base_path, "bn_model.xmlbif")
    # Save the Bayesian Network model
    bn.save(file_path, filetype='xmlbif') # save as xmlbif format

# In BN, Class variable is the target variable
def get_expert_knowledge(data: pd.DataFrame, class_column: str = "class") -> ExpertKnowledge:
    # get columns
    cols = data.columns.tolist()
    # remove Class column
    cols.remove(class_column)
    temporal_order = [ [class_column], cols ] # class before all features
    expert_knowledge = ExpertKnowledge(temporal_order=temporal_order)
    return expert_knowledge

def init_bn(data: pd.DataFrame, search_strategy: str = "hill_climbing",
                                structure_learning_params: dict = {},
                                estimator_type: str = "bayesian",
                                parameter_learning_params: dict = {}) -> DiscreteBayesianNetwork:

    # Structure Learning
    expert_knowledge = get_expert_knowledge(data)
    print("Structure Learning...")
    bn = learn_structure(data, method=search_strategy, expert_knowledge=expert_knowledge, **structure_learning_params)
    
    # Keep only Markov Blanket variables
    mb_list = sorted(bn.get_markov_blanket("class"))
    # memo full bn
    full_bn = bn.copy()
    bn = bn.subgraph(mb_list + ["class"])

    # in data, keep only markov blanket variables + class
    data_mb = data[mb_list + ["class"]]

    # Parameter Learning
    print("Parameter Learning...")
    bn = estimate_parameters(bn, data_mb, estimator_type=estimator_type, **parameter_learning_params)

    # Check model validity
    assert bn.check_model(), "The Bayesian Network model is invalid."

    return bn, full_bn

def load_bn(base_path: str):
    # Load existing bn model
    file_path = os.path.join(base_path, "bn_model.xmlbif")
    reader = XMLBIFReader(file_path)
    bn = reader.get_model()
     # Check model validity
    assert bn.check_model(), "The Bayesian Network model is invalid."
    return bn
