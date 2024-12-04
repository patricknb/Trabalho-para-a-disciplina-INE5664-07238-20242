import pandas as pd
from regressao import Regressao
import time

# Normalização com verificação de desvio padrão zero
def normalize_data(X):
    std_dev = X.std(axis=0)
    std_dev[std_dev == 0] = 1  # Substituir desvio padrão zero por 1
    return (X - X.mean(axis=0)) / std_dev

# 1. Carregar os dados de treinamento
caminho_arquivo_treino = './data/train_student_performance.csv'
dados = pd.read_csv(caminho_arquivo_treino)

# 2. Pré-processamento para os dados de treinamento
entradas = dados.iloc[:, :-1].values  # Todas as colunas, exceto a última
rotulos = dados.iloc[:, -1].values    # Última coluna (os rótulos numéricos)
entradas_normalizadas = normalize_data(entradas)

# 3. Configuração da rede neural
tamanho_entrada = entradas_normalizadas.shape[1]
#tamanho_saida = 1  # Problema de regressão com uma única saída
camadas_ocultas = [5] * 5
nn = Regressao(tamanho_entrada, camadas_ocultas, 1)

# 4. Carregar os dados de teste
#caminho_arquivo_teste = './data/train_student_performance.csv'
caminho_arquivo_teste = './data/test_student_performance.csv'
dados_teste = pd.read_csv(caminho_arquivo_teste)

# 5. Pré-processamento para os dados de teste
entradas_teste = dados_teste.iloc[:, :-1].values
rotulos_teste = dados_teste.iloc[:, -1].values
entradas_teste_normalizadas = normalize_data(entradas_teste)

# 6. Treinamento da rede neural
epocas = 1000
taxa_aprendizado = 0.01  # Taxa de aprendizado ajustada
nn.train(entradas_normalizadas, rotulos, epocas, taxa_aprendizado)

# 7. Salvar pesos e bias após o treinamento
seconds = time.time()
nn.save_pesos(f'modelo_regressao-{time.ctime(seconds)}.npz')

# 8. Avaliar o modelo com os dados de teste
print('==Teste==')
mse = nn.evaluate(entradas_teste_normalizadas, rotulos_teste)
print(f"RMSE: {mse:.4f}")
