import os
import pandas as pd

# Função de Ingestão de Dados
def ingest_data(new_data_path: str, feature_store_path: str) -> pd.DataFrame:
    """
    Faz a ingestão de dados da pasta 'new_data_path' e concatena com dados 
    existentes na feature store localizada em 'feature_store_path'. Caso 
    a feature store não exista, uma nova será criada.

    Args:
        new_data_path (str): caminho para os novos dados .csv
        feature_store_path (str): caminho para a feture store com dados .csv

    Returns:
        pd.DataFrame: dataset atualizado após nova ingestão de dados
    """
    # Verifica se a Feature Store existe
    if os.path.exists(feature_store_path):
        # Carrega Feature Store
        existing_df = pd.read_csv(feature_store_path)
    else:
        # Cria um DataFrame vazio se a Feature Store não existir
        existing_df = pd.DataFrame()

    # Carrega dados novos
    new_df = pd.read_csv(new_data_path)

    # Concatena novos dados com dados existentes
    updated_df = pd.concat([existing_df, new_df], ignore_index=True)

    # Remove possíveis duplicadas
    updated_df.drop_duplicates(inplace=True)

    # Grava dataframe concatenado na feature store
    updated_df.to_csv(feature_store_path, index=False)

    return updated_df

def build_training_dataset(feature_store_path: str, target_path: str, abt_path: str = "../data/abt.csv") -> pd.DataFrame:
    """
    Carrega a feature store e o arquivo target.csv, realiza a junção
    entre ambos utilizando 'worker_id' como chave primária,
    aplica limpeza e retorna o Dataframe final (ABT pronto para treinamento)

    Args:
        feature_store_path (str): caminho para a feature store (arquivo .csv)
        target_path (str): caminho para o arquivo target.csv (contendo worker_id e income)
        abt_path (str): caminho onde salvar a ABT final (padrão: "../data/abt.csv")
    
    Returns:
        pd.DataFrame: Dataframe ABT pronto para modelagem
    """
    # Carrega a Feature Store
    features_df = pd.read_csv(feature_store_path)
    features_df.drop_duplicates(inplace=True)
    features_df.dropna(inplace=True)

    # Carrega o arquivo Targuet
    target_df = pd.read_csv(target_path)
    target_df.drop_duplicates(inplace=True)
    target_df.dropna(inplace=True)

    # Faz o join usando o worker_id como chave primária
    abt_df = pd.merge(features_df, target_df, on='worker_id', how="left")

    # Salva a ABT
    abt_df.to_csv(abt_path, index=False)

    return abt_df

# O bloco abaixo roda apenas se 'python feature_store.py' seja chamado diretamente
if __name__ == "__main__":

    # Exemplo para testar e debugar:

    # 1. Ingestão de novos dados
    updated1 = ingest_data(
        new_data_path="../data/new_data/data1.csv",
        feature_store_path="../data/feature_store.csv"
        )
    print(f"Ingestão de data1.csv concluída. Shape: {updated1.shape}")

    updated2 = ingest_data(
        new_data_path="../data/new_data/data2.csv",
        feature_store_path="../data/feature_store.csv"
        )
    print(f"Ingestão de data2.csv concluída. Shape: {updated2.shape}")

    # 2. Construir ABT
    abt_df = build_training_dataset(
        feature_store_path="../data/feature_store.csv",
        target_path="../data/target.csv",
        abt_path="../data/abt.csv"
    )
    print(f"ABT criada: {abt_df.shape}")