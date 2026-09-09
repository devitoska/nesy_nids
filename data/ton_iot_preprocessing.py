import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Preprocessing parameters

# seed
random_state = 42

# Split ratios for train1, train2 and test datasets
train_1_split = 0.4
train_2_split = 0.4
test_split = 0.2

# Number of bins for discretization of numeric columns
num_bins = 50

def preprocess(data) -> pd.DataFrame:

    '''
    Create the following dataset splits:
    - Train 1: known data used to create the Bayesian Network
    - Train 2: known data used to generate explanations
    - Test: data used for testing
    '''

    # rename columns to lowercase
    data.columns = [col.lower() for col in data.columns]

    # rename 'type' column to 'class'
    data = data.rename(columns={'type': 'class'})

    # remove 'label', 'src_ip', 'dst_ip' columns
    data = data.drop(columns=['label', 'src_ip', 'dst_ip', 'dns_query'])

    # get categorical columns, i.e. those with dtype 'object'
    cat_columns = data.select_dtypes(include=['object']).columns.tolist() + ['dns_qclass', 'dns_qtype', 'dns_rcode', 'http_status_code']

    # get numeric columns to normalize and discretize
    numeric_columns = [col for col in data.columns if col not in cat_columns]
    
    # Convert categorical columns to numerical using alphabetical encoding
    for col in cat_columns:
        if col != 'class':
            data[col] = data[col].astype('category').cat.codes

    # Split data into a training partition and a test partition with stratification.
    train_data, test_data = train_test_split(
        data,
        test_size=test_split,
        stratify=data['class'],
        random_state=random_state,
    )
    train_data = train_data.copy()
    test_data = test_data.copy()

    drop_columns = []

    # Fit normalization parameters and quantile-bin boundaries on the training
    # partition, then apply exactly the same transformation to the test set.
    for col in numeric_columns:
        train_min = train_data[col].min()
        train_max = train_data[col].max()
        train_range = train_max - train_min

        if train_range == 0:
            drop_columns.append(col)
            continue

        normalized_train = (train_data[col] - train_min) / train_range
        _, bin_edges = pd.qcut(
            normalized_train,
            labels=False,
            q=num_bins,
            duplicates='drop',
            retbins=True,
        )

        # Values outside the range observed during training are assigned to the
        # first or last bin instead of becoming missing values.
        bin_edges[0] = float('-inf')
        bin_edges[-1] = float('inf')

        train_data.loc[:, col] = pd.cut(
            normalized_train,
            bins=bin_edges,
            labels=False,
            include_lowest=True,
        )

        normalized_test = (test_data[col] - train_min) / train_range
        test_data.loc[:, col] = pd.cut(
            normalized_test,
            bins=bin_edges,
            labels=False,
            include_lowest=True,
        )

    # Drop columns that are constant in the training partition from both
    # training and test data.
    drop_columns.extend(
        col for col in train_data.columns
        if train_data[col].nunique() <= 1 and col not in drop_columns
    )
    train_data = train_data.drop(columns=drop_columns)
    test_data = test_data.drop(columns=drop_columns)

    # Divide the transformed training partition into the two subsets used by
    # Bayesian-network learning and anomaly-detector learning.
    train_1_data, train_2_data = train_test_split(
        train_data,
        test_size=train_2_split / (train_1_split + train_2_split),
        stratify=train_data['class'],
        random_state=random_state,
    )

    # create output directory if it doesn't exist
    os.makedirs('data/dataset/ton-iot_net', exist_ok=True)

    print(f'Columns to drop after discretization and normalization: {len(drop_columns)}')

    for data_name, data_split in zip(
        ['train_1', 'train_2', 'test'],
        [train_1_data, train_2_data, test_data],
    ):
        data_split.to_csv(
            f'data/dataset/ton-iot_net/{data_name}_data.csv',
            index=False,
        )
        print(f'{data_name} data: {len(data_split)} samples')

if __name__ == "__main__":
    ton_iot_data_path = '/home/vscaraggi/bayenesian/data/dataset/ton-iot_net/train_test_data.csv'
    data = pd.read_csv(ton_iot_data_path)
    preprocess(data)
