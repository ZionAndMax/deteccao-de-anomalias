import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import io

# ==========================================
# 1. CRIANDO A BASE DE DADOS SIMULADA (CSV)
# ==========================================
# Temos transações normais (valores baixos/médios de dia) 
# e anomalias (valores altíssimos de madrugada).
csv_data = """id_transacao,valor,hora_transacao,tipo_cartao
1,50.50,14:30,credito
2,25.00,09:15,debito
3,150.75,18:45,credito
4,15.20,08:00,debito
5,5000.00,03:15,credito
6,45.00,12:10,credito
7,80.90,20:00,debito
8,10.00,10:30,debito
9,12000.00,04:45,credito
10,65.20,15:20,credito
"""

# Carregando os dados para um DataFrame do Pandas
df = pd.read_csv(io.StringIO(csv_data))
print("Dados Originais:")
print(df[['id_transacao', 'valor', 'hora_transacao']].head(3), "\n")

# ==========================================
# 2. FEATURE ENGINEERING (Engenharia de Recursos)
# ==========================================
print("Aplicando Feature Engineering...")

# A. Extração de Tempo: Pegar apenas a hora da string (ex: '14:30' vira 14)
df['hora'] = pd.to_datetime(df['hora_transacao'], format='%H:%M').dt.hour

# B. Criação de Variável de Contexto: É madrugada? (1 para Sim, 0 para Não)
# Compras de madrugada têm maior risco de fraude.
df['is_madrugada'] = df['hora'].apply(lambda x: 1 if 0 <= x <= 6 else 0)

# C. Transformação Matemática: Logaritmo do Valor
# Valores financeiros costumam ser muito distorcidos (muitos valores baixos, poucos altíssimos).
# O logaritmo aproxima os dados de uma distribuição normal, ajudando o modelo.
df['valor_log'] = np.log1p(df['valor'])

# D. Codificação de Categoria (One-Hot Encoding)
# Transforma a coluna de texto 'tipo_cartao' em números que o modelo entende.
df = pd.get_dummies(df, columns=['tipo_cartao'], drop_first=True)

# Selecionando as features finais que vão alimentar o modelo
features = ['valor_log', 'is_madrugada', 'tipo_cartao_debito']
X = df[features]

# E. Normalização dos Dados
# Coloca todas as features na mesma escala, para que o 'valor' não ofusque a 'madrugada'.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# 3. TREINAMENTO DO MODELO
# ==========================================
# Isolation Forest isola anomalias cortando os dados aleatoriamente. 
# Anomalias são mais fáceis de isolar e precisam de menos "cortes".
# contamination = proporção esperada de anomalias (estamos estimando 20% para esta base pequena)
modelo = IsolationForest(contamination=0.2, random_state=42)
modelo.fit(X_scaled)

# ==========================================
# 4. PREDIÇÃO E RESULTADOS
# ==========================================
# O modelo retorna 1 para normal e -1 para anomalia
df['predicao'] = modelo.predict(X_scaled)
df['status'] = df['predicao'].map({1: '✅ Normal', -1: '🚨 ANOMALIA'})

print("\nResultado da Detecção:")
print(df[['id_transacao', 'valor', 'hora_transacao', 'status']])
