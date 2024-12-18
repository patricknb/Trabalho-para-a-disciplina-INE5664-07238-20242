import numpy as np
import pandas as pd
import time

# ==================== Classe Classificação Multiclasse ====================
class ClassificacaoMulticlasse:
    def __init__(self, tamanho_entrada, camadas_ocultas, tamanho_saida):
        # Inicializa as camadas da rede neural
        self.tamanho_entrada = tamanho_entrada
        self.camadas_ocultas = camadas_ocultas
        self.tamanho_saida = tamanho_saida
        
        self.pesos = []
        self.biases = []
        
        # Inicialização das camadas com valores aleatórios
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
        # Função softmax para ativação da camada de saída
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def cross_entropy(self, y_pred, y_true):
        # Função de perda: Entropia cruzada
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
        # Atualização dos pesos e vieses na camada de saída
        self.pesos[-1] -= np.dot(self.activations[-2].T, gradiente_saida) * taxa_aprendizado / m
        self.biases[-1] -= np.sum(gradiente_saida, axis=0, keepdims=True) * taxa_aprendizado / m
        
        gradiente = gradiente_saida
        # Retropropagação para as camadas ocultas
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
            
            if epoca % 100 == 0 or epoca == epocas-1:
                # Calcular a precisão nos dados de treino durante o treinamento
                metrics = self.evaluate(X, y)
                print(f'Época {epoca}, Custo: {custo:.4f}, Acurácia: {metrics["Accuracy"]:.2%} Precisão no treino: {metrics["Precision"]:.2%}, Recall: {metrics["Recall"]:.2%}, F1-score: {metrics["F1-Score"]:.2%}')
    
    def evaluate(self, X, y):
        # Avalia a precisão do modelo com métricas adicionais
        y_pred = self.forward(X)  # Probabilidades previstas
        predictions = np.argmax(y_pred, axis=1)  # Classes previstas
        true_labels = np.argmax(y, axis=1)  # Classes reais

        # Inicializar variáveis para métricas
        classes = np.unique(true_labels)
        precisions, recalls, f1_scores = [], [], []

        for c in classes:
            TP = np.sum((predictions == c) & (true_labels == c))  # Verdadeiros Positivos para a classe c
            FP = np.sum((predictions == c) & (true_labels != c))  # Falsos Positivos para a classe c
            FN = np.sum((predictions != c) & (true_labels == c))  # Falsos Negativos para a classe c

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1_score)

        # Cálculo das médias (média macro)
        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)
        avg_f1_score = np.mean(f1_scores)
        accuracy = np.mean(predictions == true_labels)

        # Retorno como dicionário
        metrics = {
            'Accuracy': accuracy,
            'Precision': avg_precision,
            'Recall': avg_recall,
            'F1-Score': avg_f1_score
        }
        return metrics
    
    # Salvar pesos e bias
    def save_pesos(self, file_path):
        # Salva pesos e bias com nomes claros para cada camada
        np.savez(file_path, 
                pesos_0=self.pesos[0], biases_0=self.biases[0],
                pesos_1=self.pesos[1], biases_1=self.biases[1],
                pesos_2=self.pesos[2], biases_2=self.biases[2])

# ==================== Funções de Carregamento e Pré-processamento ====================
def normalizar_dados(entradas):
    # Normalização: traz os dados para a mesma escala
    desvio_padrao = entradas.std(axis=0)
    desvio_padrao[desvio_padrao == 0] = 1  # Evita divisão por zero
    return (entradas - entradas.mean(axis=0)) / desvio_padrao

# ==================== Execução do Modelo ====================
# 1. Carregar os dados de treinamento
caminho_arquivo_treino = './data/train_esrb_rating.csv'
dados = pd.read_csv(caminho_arquivo_treino)

# Separar entradas e rótulos
entradas = dados.iloc[:, :-1].values  # Todas as colunas, exceto a última
rotulos = dados.iloc[:, -1].values    # Última coluna (os rótulos de classe)

# 2. Pré-processamento para os dados de treinamento
entradas_normalizadas = normalizar_dados(entradas)
rotulos_one_hot = pd.get_dummies(rotulos).values  # One-hot encoding para as classes

# 3. Definir a arquitetura da rede neural
tamanho_entrada = entradas_normalizadas.shape[1]
tamanho_saida = rotulos_one_hot.shape[1]
camadas_ocultas = [10, 10, 10]  # Número de neurônios nas camadas ocultas

# Inicializar a rede neural
nn = ClassificacaoMulticlasse(tamanho_entrada, camadas_ocultas, tamanho_saida)

# 4. Carregar os dados de teste
caminho_arquivo_teste = './data/test_esrb_rating.csv'
dados_teste = pd.read_csv(caminho_arquivo_teste)

# Separar entradas e rótulos para o teste
entradas_teste = dados_teste.iloc[:, :-1].values
rotulos_teste = dados_teste.iloc[:, -1].values

# 5. Pré-processamento para os dados de teste
entradas_teste_normalizadas = normalizar_dados(entradas_teste)
rotulos_teste_one_hot = pd.get_dummies(rotulos_teste).values

# 6. Treinamento do modelo
epocas = 5000
batch_size = len(entradas_normalizadas)  # Usando todos os dados para o treinamento
taxa_aprendizado = 0.01
nn.train(entradas_normalizadas, rotulos_one_hot, epocas, batch_size, taxa_aprendizado)

# 7. Salvar os pesos após o treinamento
seconds = time.time()
nn.save_pesos(f'modelo_classificacao_multiclasse-{time.ctime(seconds)}.npz')

# 8. Avaliar o modelo com os dados de teste
metrics = nn.evaluate(entradas_teste_normalizadas, rotulos_teste_one_hot)
print('==Teste==')
print(f"Acurácia: {metrics['Accuracy']:.2%}")
print(f"Precisão: {metrics['Precision']:.2%}")
print(f"Recall: {metrics['Recall']:.2%}")
print(f"F1-Score: {metrics['F1-Score']:.2%}")
