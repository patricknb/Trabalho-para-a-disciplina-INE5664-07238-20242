import pandas as pd
import numpy as np

# Carregar o arquivo
file_path = 'esrb-rating.csv'
data = pd.read_csv(file_path)

# Separar os dados: entradas (todas as colunas, exceto a última) e rótulos (última coluna)
inputs = data.iloc[:, :-1]  # Todas as colunas, exceto a última
labels = data.iloc[:, -1]  # Última coluna (as classes)

# Configurações da rede
nro_entradas = inputs.shape[1]  # Número de colunas de entrada
nro_neuronios = 4  # Número de neurônios por camada oculta
nro_camadas = 3  # Número de camadas ocultas
nro_classes = 4  # Número de classes para a saída

# Inicializar pesos e bias para cada camada oculta
camadas = []
for i in range(nro_camadas):
    if i == 0:
        # Primeira camada: pesos conectando entradas aos neurônios
        pesos = np.random.rand(nro_entradas, nro_neuronios)
    else:
        # Camadas ocultas: pesos conectando neurônios da camada anterior
        pesos = np.random.rand(nro_neuronios, nro_neuronios)
    bias = np.random.rand(nro_neuronios)  # Bias para os neurônios da camada
    camadas.append((pesos, bias))

# Adicionar a camada de saída
pesos_saida = np.random.rand(nro_neuronios, nro_classes)
bias_saida = np.random.rand(nro_classes)

# Definir as funções de ativação
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Estabilização numérica
    return exp_x / np.sum(exp_x)

def softmax_probabilidades(logits):
    logits = np.array(logits)  # Garantir que os logits são um array numpy
    max_logit = np.max(logits)
    exp_logits = np.exp(logits - max_logit)  # Exponenciação dos logits
    probabilities = exp_logits / np.sum(exp_logits)  # Normalização para somar 1
    return probabilities

# Exibir configurações
print(f"Número de entradas: {nro_entradas}")
print(f"Número de camadas ocultas: {nro_camadas}")
print(f"Número de neurônios por camada oculta: {nro_neuronios}")
print(f"Número de classes (saída): {nro_classes}")

# Forward propagation
for epoca in range(1):  # Apenas uma época para simplificar
    for index, row in inputs.iterrows():  # Iterar pelas linhas das entradas
        x = row.to_numpy()  # Converter a linha para um vetor NumPy
        
        # Passar pelas camadas ocultas
        for camada_idx, (pesos, bias) in enumerate(camadas):
            v = np.dot(x, pesos) + bias  # Cálculo do valor da camada
            x = relu(v)  # Aplicar ReLU e usar a saída como entrada para a próxima camada
            print(f"Camada {camada_idx + 1}, saída: {x}")
        
        # Camada de saída (após as camadas ocultas)
        v_saida = np.dot(x, pesos_saida) + bias_saida
        logits = softmax(v_saida)  # Aplicar softmax para obter logits
        y_prob = softmax_probabilidades(logits)
        print(f"Camada de saída (softmax): {y_prob}")

         # Mostrar a classe prevista (índice com maior probabilidade)
        classe_prevista = np.argmax(y_prob) + 1  # Adiciona 1 porque as classes começam de 1
        classe_corretiva = labels.iloc[index]  # A classe correta para esta linha de entrada
        
        print(f"Classe correta: {classe_corretiva}, Classe prevista: {classe_prevista}\n")


# Apenas para verificação, podemos mostrar os rótulos correspondentes
print("\nRótulos das amostras:")
print(labels.to_numpy())
