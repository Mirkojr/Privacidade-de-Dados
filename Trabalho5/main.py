import kagglehub
import pandas as pd
from LaplaceKnn import LaplaceKNN
import numpy as np
import os


def data_download():
    path = kagglehub.dataset_download("wenruliu/adult-income-dataset")
    csv_file = "adult.csv"
    data = pd.read_csv(os.path.join(path, csv_file))
    return data

def preprocess_data(data):
    cols_to_drop = ['fnlwgt', 'education', 'capital-gain', 'capital-loss', 'hours-per-week']
    data = data.drop(columns=cols_to_drop)
    
    data = data.replace('?', np.nan)
    data = data.dropna()
    return data

if __name__ == "__main__":

    print("Baixando o dataset Adult Income do Kaggle...")
    data = data_download()

    print("Pré-processando os dados...")
    data = preprocess_data(data)
    
    print(data.head())
    