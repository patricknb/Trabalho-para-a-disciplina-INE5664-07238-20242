import pandas as pd
from classificacao_multiclasse import Classificacao_multiclasse
import time


# 1. Carregar os dados de treinamento
#caminho_arquivo = 'esrb-rating.csv'
caminho_arquivo_treino = './data/train_esrb_rating.csv'
dados = pd.read_csv(caminho_arquivo_treino)

# 2. Pré-processamento para os dados de treinamento
entradas = dados.iloc[:, :-1].values  # Todas as colunas, exceto a última
rotulos = dados.iloc[:, -1].values    # Última coluna (os rótulos de classe)

# Normalização com verificação de desvio padrão zero
def normalize_data(X):
    # Evitar divisão por zero, adicionando uma constante para desvio padrão
    std_dev = X.std(axis=0)
    std_dev[std_dev == 0] = 1  # Substituir desvio padrão zero por 1
    return (X - X.mean(axis=0)) / std_dev

entradas_normalizadas = normalize_data(entradas)
rotulos_one_hot = pd.get_dummies(rotulos).values #preciso checar dnv, talvez não seja one-hot encoding

# 3. Inicializar a rede neural
tamanho_entrada = entradas_normalizadas.shape[1]
print(tamanho_entrada)
tamanho_saida = rotulos_one_hot.shape[1]
print(tamanho_saida)

qtd_camadas = 5
qtd_neuronio_por_camada = 10
camadas_ocultas = []
for n in range(qtd_camadas):
    camadas_ocultas.append(qtd_neuronio_por_camada)
#camadas_ocultas = [5, 5]  # Número de neurônios nas camadas ocultas
print(camadas_ocultas)
nn = Classificacao_multiclasse(tamanho_entrada, camadas_ocultas, tamanho_saida)

# 4. Carregar os dados de teste
caminho_arquivo_teste = './data/test_esrb_rating.csv'
dados_teste = pd.read_csv(caminho_arquivo_teste)

# 7. Pré-processamento para os dados de teste
entradas_teste = dados_teste.iloc[:, :-1].values  # Todas as colunas, exceto a última
rotulos_teste = dados_teste.iloc[:, -1].values    # Última coluna (os rótulos de classe)
entradas_teste_normalizadas = normalize_data(entradas_teste)
rotulos_teste_one_hot = pd.get_dummies(rotulos_teste).values

# 5. Treinamento da rede neural
epocas = 5000
batch_size = len(entradas_normalizadas)#32 funcionou como um min, preciso checar se funciona com outro data base :\
taxa_aprendizado = 0.01
nn.train(entradas_normalizadas, rotulos_one_hot, epocas, batch_size, taxa_aprendizado)

# 6. Salvar pesos e bias após o treinamento
seconds = time.time()
nn.save_pesos('modelo_classificacao_multiclasse-{}.npz'.format(time.ctime(seconds)))

# 7. Avaliar o modelo com os dados de teste após o treinamento
#precisao_treino = nn.evaluate(entradas_normalizadas, rotulos_one_hot)
#print(f'Precisão nos dados de treino: {precisao_treino:.2%}')
print('==Teste==')
'''precisao_teste = nn.evaluate(entradas_teste_normalizadas, rotulos_teste_one_hot)
print(f'Precisão nos dados de teste: {precisao_teste:.2%}')'''

metrics = nn.evaluate(entradas_teste_normalizadas, rotulos_teste_one_hot)

print(f"Acurácia: {metrics['Accuracy']:.2%}")
print(f"Precisão: {metrics['Precision']:.2%}")
print(f"Recall: {metrics['Recall']:.2%}")
print(f"F1-Score: {metrics['F1-Score']:.2%}")

