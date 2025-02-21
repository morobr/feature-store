# Módulo para criar feature store

# Imports
import pandas as pd
import uuid 
  
# criação do dataframe à partir de base .csv
df = pd.read_csv(
    "../data/adult.data",
    sep=r",",
    header=None,
    engine="python",
    names=[
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education_num",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
        "native_country",
        "income",
    ],
)

# Embaralha as linhas para separar de forma randômica o dataset para separar a base para inferência
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Cria a coluna de ID dos colaboradores
df['worker_id'] = [str(uuid.uuid4()) for _ in range(len(df))]

# Separa as variáveis preditoras da target
df_target = df[["worker_id", "income"]].copy()
df_features = df.drop(columns=["income"]).copy()

# Separa features um 70% para modelar e 30% para inferência
split_idx = int(0.7 * len(df_features))
df_data = df_features.iloc[:split_idx].copy()
df_inference = df_features.iloc[split_idx:].copy()

# Separa df_data em dois para simular uma pipeline que receba dados novos em uma pasta e concatene
split_idx_data = len(df_data) // 2
df1 = df_data.iloc[:split_idx_data]
df2 = df.iloc[split_idx_data:]

# Salva daatsets

df1.to_csv("../data/new_data/data1.csv", index=False) # primeira metade dos 70% features + worker_id
df2.to_csv("../data/new_data/data2.csv", index=False) # sedunda metade dos 70% features + worker_id
df_inference.to_csv("../data/inference.csv", index=False) # 30% features + worker_id
df_target.to_csv("../data/target.csv", index=False) # worker_id + income