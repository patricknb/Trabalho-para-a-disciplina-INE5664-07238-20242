'''versão com treino

import numpy as np
import pandas as pd

# Classe para a Rede Neural
class RedeNeural:
    def __init__(self, tamanho_entrada, camadas_ocultas, tamanho_saida):
        self.tamanho_entrada = tamanho_entrada
        self.camadas_ocultas = camadas_ocultas
        self.tamanho_saida = tamanho_saida
        
        # Inicializando os pesos e bias para cada camada com inicialização de He
        self.pesos = []
        self.biases = []
        
        # Primeira camada oculta (inicialização de He)
        self.pesos.append(np.random.randn(tamanho_entrada, camadas_ocultas[0]) * np.sqrt(2. / tamanho_entrada))
        self.biases.append(np.zeros((1, camadas_ocultas[0])))
        
        # Camadas ocultas intermediárias (inicialização de He)
        for i in range(1, len(camadas_ocultas)):
            self.pesos.append(np.random.randn(camadas_ocultas[i-1], camadas_ocultas[i]) * np.sqrt(2. / camadas_ocultas[i-1]))
            self.biases.append(np.zeros((1, camadas_ocultas[i])))
        
        # Camada de saída (inicialização de He)
        self.pesos.append(np.random.randn(camadas_ocultas[-1], tamanho_saida) * np.sqrt(2. / camadas_ocultas[-1]))
        self.biases.append(np.zeros((1, tamanho_saida)))
    
    # Função de ativação ReLU
    def relu(self, x):
        return np.maximum(0, x)

    # Derivada da ReLU
    def relu_derivada(self, x):
        return (x > 0).astype(int)

    # Função Softmax
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Para estabilidade numérica
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    # Função de custo (entropia cruzada)
    def cross_entropy(self, y_pred, y_true):
        m = y_true.shape[0]
        epsilon = 1e-8  # Para evitar log(0)
        return -np.sum(y_true * np.log(y_pred + epsilon)) / m

    # Forward pass
    def forward(self, X):
        self.activations = []
        self.z_values = []
        
        # Passando pela primeira camada oculta
        z = np.dot(X, self.pesos[0]) + self.biases[0]
        a = self.relu(z)
        self.activations.append(a)
        self.z_values.append(z)
        
        # Passando pelas camadas ocultas intermediárias
        for i in range(1, len(self.camadas_ocultas)):
            z = np.dot(self.activations[-1], self.pesos[i]) + self.biases[i]
            a = self.relu(z)
            self.activations.append(a)
            self.z_values.append(z)
        
        # Camada de saída (com softmax)
        z_output = np.dot(self.activations[-1], self.pesos[-1]) + self.biases[-1]
        y_pred = self.softmax(z_output)
        self.activations.append(y_pred)
        
        return y_pred

    # Backpropagation
    def backpropagate(self, X, y, taxa_aprendizado):
        m = X.shape[0]
        
        # Gradiente da camada de saída
        gradiente_saida = self.activations[-1] - y
        
        # Atualizar pesos e bias da camada de saída
        self.pesos[-1] -= np.dot(self.activations[-2].T, gradiente_saida) * taxa_aprendizado / m
        self.biases[-1] -= np.sum(gradiente_saida, axis=0, keepdims=True) * taxa_aprendizado / m
        
        # Propagar os gradientes pelas camadas ocultas
        gradiente = gradiente_saida
        for i in range(len(self.camadas_ocultas) - 1, -1, -1):
            gradiente = np.dot(gradiente, self.pesos[i+1].T) * self.relu_derivada(self.z_values[i])
            if i > 0:
                self.pesos[i] -= np.dot(self.activations[i-1].T, gradiente) * taxa_aprendizado / m
                self.biases[i] -= np.sum(gradiente, axis=0, keepdims=True) * taxa_aprendizado / m
            else:
                self.pesos[i] -= np.dot(X.T, gradiente) * taxa_aprendizado / m
                self.biases[i] -= np.sum(gradiente, axis=0, keepdims=True) * taxa_aprendizado / m
        
        # Verificando o valor do gradiente para debugging
        if np.any(np.isnan(self.pesos[0])) or np.any(np.isnan(self.biases[0])):
            print("NaN encontrado nos pesos ou bias")

    # Treinamento da rede neural
    def train(self, X, y, epocas, batch_size, taxa_aprendizado):
        for epoca in range(epocas):
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                
                # Forward pass
                y_pred = self.forward(X_batch)
                
                # Cálculo do custo
                custo = self.cross_entropy(y_pred, y_batch)
                
                # Backpropagation
                self.backpropagate(X_batch, y_batch, taxa_aprendizado)
            
            if epoca % 10 == 0:
                print(f'Época {epoca}, Custo: {custo:.4f}')
    
    # Avaliação do modelo
    def evaluate(self, X, y):
        y_pred = self.forward(X)
        predictions = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == true_labels)
        return accuracy

# 1. Carregar os dados
caminho_arquivo = 'esrb-rating.csv'
dados = pd.read_csv(caminho_arquivo)

# 2. Pré-processamento
entradas = dados.iloc[:, :-1].values  # Todas as colunas, exceto a última
rotulos = dados.iloc[:, -1].values    # Última coluna (os rótulos de classe)

# 3. Normalizar os dados de entrada
entradas_normalizadas = (entradas - entradas.mean(axis=0)) / entradas.std(axis=0)

# 4. One-Hot Encoding dos rótulos
rotulos_one_hot = pd.get_dummies(rotulos).values

# 5. Inicializar a rede neural
tamanho_entrada = entradas_normalizadas.shape[1]
tamanho_saida = rotulos_one_hot.shape[1]
camadas_ocultas = [5, 5]  # Número de neurônios nas camadas ocultas
nn = RedeNeural(tamanho_entrada, camadas_ocultas, tamanho_saida)

# 6. Treinamento da rede neural
epocas = 100
batch_size = 32
taxa_aprendizado = 0.01
nn.train(entradas_normalizadas, rotulos_one_hot, epocas, batch_size, taxa_aprendizado)

# 7. Avaliação
precisao = nn.evaluate(entradas_normalizadas, rotulos_one_hot)
print(f'Precisão final: {precisao:.2%}')
'''

#versão com treino e teste
import numpy as np
import pandas as pd

class RedeNeural:
    def __init__(self, tamanho_entrada, camadas_ocultas, tamanho_saida):
        self.tamanho_entrada = tamanho_entrada
        self.camadas_ocultas = camadas_ocultas
        self.tamanho_saida = tamanho_saida
        
        self.pesos = []
        self.biases = []
        
        # Inicialização das camadas
        self.pesos.append(np.random.randn(tamanho_entrada, camadas_ocultas[0]) * np.sqrt(2. / tamanho_entrada))
        self.biases.append(np.zeros((1, camadas_ocultas[0])))
        
        for i in range(1, len(camadas_ocultas)):
            self.pesos.append(np.random.randn(camadas_ocultas[i-1], camadas_ocultas[i]) * np.sqrt(2. / camadas_ocultas[i-1]))
            self.biases.append(np.zeros((1, camadas_ocultas[i])))
        
        self.pesos.append(np.random.randn(camadas_ocultas[-1], tamanho_saida) * np.sqrt(2. / camadas_ocultas[-1]))
        self.biases.append(np.zeros((1, tamanho_saida)))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivada(self, x):
        return (x > 0).astype(int)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def cross_entropy(self, y_pred, y_true):
        m = y_true.shape[0]
        epsilon = 1e-8  # Para evitar log(0)
        return -np.sum(y_true * np.log(y_pred + epsilon)) / m

    def forward(self, X):
        self.activations = []
        self.z_values = []
        
        # Propagação para frente
        z = np.dot(X, self.pesos[0]) + self.biases[0]
        a = self.relu(z)
        self.activations.append(a)
        self.z_values.append(z)
        
        for i in range(1, len(self.camadas_ocultas)):
            z = np.dot(self.activations[-1], self.pesos[i]) + self.biases[i]
            a = self.relu(z)
            self.activations.append(a)
            self.z_values.append(z)
        
        z_output = np.dot(self.activations[-1], self.pesos[-1]) + self.biases[-1]
        y_pred = self.softmax(z_output)
        self.activations.append(y_pred)
        
        return y_pred

    def backpropagate(self, X, y, taxa_aprendizado):
        m = X.shape[0]
        
        gradiente_saida = self.activations[-1] - y
        
        self.pesos[-1] -= np.dot(self.activations[-2].T, gradiente_saida) * taxa_aprendizado / m
        self.biases[-1] -= np.sum(gradiente_saida, axis=0, keepdims=True) * taxa_aprendizado / m
        
        gradiente = gradiente_saida
        for i in range(len(self.camadas_ocultas) - 1, -1, -1):
            gradiente = np.dot(gradiente, self.pesos[i+1].T) * self.relu_derivada(self.z_values[i])
            if i > 0:
                self.pesos[i] -= np.dot(self.activations[i-1].T, gradiente) * taxa_aprendizado / m
                self.biases[i] -= np.sum(gradiente, axis=0, keepdims=True) * taxa_aprendizado / m
            else:
                self.pesos[i] -= np.dot(X.T, gradiente) * taxa_aprendizado / m
                self.biases[i] -= np.sum(gradiente, axis=0, keepdims=True) * taxa_aprendizado / m

    def train(self, X, y, epocas, batch_size, taxa_aprendizado):
        for epoca in range(epocas):
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                
                y_pred = self.forward(X_batch)
                custo = self.cross_entropy(y_pred, y_batch)
                self.backpropagate(X_batch, y_batch, taxa_aprendizado)
            
            if epoca % 10 == 0 or epoca == epocas-1:
                # Calcular a precisão nos dados de treino durante o treinamento
                treino_precisao = self.evaluate(X, y)
                print(f'Época {epoca}, Custo: {custo:.4f}, Precisão no treino: {treino_precisao:.2%}')
    
    def evaluate(self, X, y):
        y_pred = self.forward(X)
        predictions = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == true_labels)
        return accuracy
    
    # Salvar pesos e bias
    def save_pesos(self, file_path):
        # Salva pesos e bias com nomes claros para cada camada
        np.savez(file_path, 
                pesos_0=self.pesos[0], biases_0=self.biases[0],
                pesos_1=self.pesos[1], biases_1=self.biases[1],
                pesos_2=self.pesos[2], biases_2=self.biases[2])

    # Carregar pesos e bias
    # talvez delete, não estou usando :x
    def load_pesos(self, file_path):
        npzfile = np.load(file_path)
        # Carregar pesos e bias para cada camada com nomes explícitos
        self.pesos = [npzfile[f'pesos_{i}'] for i in range(len(self.pesos))]
        self.biases = [npzfile[f'biases_{i}'] for i in range(len(self.biases))]

