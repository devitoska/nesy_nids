import pandas as pd
from sklearn.model_selection import train_test_split
import os

train_1_split = 0.4
train_2_split = 0.4
test_split = 0.2

def preprocess(data) -> pd.DataFrame:

    '''
    Create the following dataset splits:
    - Train 1: known data used to create the Bayesian Network
    - Train 2: known data used to generate explanations
    - Test: known data used for testing
    '''

    # rename columns to lowercase
    data.columns = [col.lower() for col in data.columns]

    # rename 'type' column to 'class'
    data = data.rename(columns={'type': 'class'})

    # remove 'label', 'src_ip', 'dst_ip' columns
    data = data.drop(columns=['label', 'src_ip', 'dst_ip', 'dns_query'])

    # get discrete columns, i.e. those with dtype 'object'
    cat_columns = data.select_dtypes(include=['object']).columns.tolist() + ['dns_qclass', 'dns_qtype', 'dns_rcode', 'http_status_code']

    # get numeric columns to normalize and discretize
    numeric_columns = [col for col in data.columns if col not in cat_columns]
    
    # Convert categorical columns to numerical using alphabetical encoding
    for col in cat_columns:
        if col != 'class':
            data[col] = data[col].astype('category').cat.codes

    # Min-max normalize and Discretize numeric columns by creating equal-size bins, assuming 50 bins
    for col in numeric_columns:
       data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
       data[col] = pd.qcut(data[col], labels=False, q=50, duplicates='drop')

    # remove columns with only one unique value (not useful for classification)
    data = data.loc[:, data.nunique() > 1]

    # split data into train1, train2 and test with stratification
    train_1_data, temp_data = train_test_split(data, test_size=(1 - train_1_split), stratify=data['class'], random_state=42)
    train_2_data, test_data = train_test_split(temp_data, test_size=(test_split / (test_split + train_2_split)), stratify=temp_data['class'], random_state=42)

    # save datasets to csv
    os.makedirs('data/dataset/ton-iot_net', exist_ok=True)
    train_1_data.to_csv('data/dataset/ton-iot_net/train_1_data.csv', index=False)
    train_2_data.to_csv('data/dataset/ton-iot_net/train_2_data.csv', index=False)
    test_data.to_csv('data/dataset/ton-iot_net/test_data.csv', index=False)

    # print total count of samples in each split
    print(f"Train 1 samples: {len(train_1_data)}")
    print(f"Train 2 samples: {len(train_2_data)}")
    print(f"Test samples: {len(test_data)}")

if __name__ == "__main__":
    ton_iot_data_path = '/home/vscaraggi/bayenesian/data/dataset/ton-iot_net/train_test_data.csv'
    data = pd.read_csv(ton_iot_data_path)
    preprocess(data)