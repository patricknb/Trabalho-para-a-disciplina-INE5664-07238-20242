import numpy as np
import pandas as pd
import time

# ==================== Classe Regressão Neural ====================
class Regressao:
    def __init__(self, entradas_tamanho, camadas_ocultas, saidas_tamanho):
        # Define as dimensões das camadas da rede neural
        self.tamanhos_camadas = [entradas_tamanho] + camadas_ocultas + [saidas_tamanho]
        self.num_camadas = len(self.tamanhos_camadas) - 1

        # Inicializa pesos e vieses aleatoriamente
        self.pesos = [
            np.random.randn(self.tamanhos_camadas[i], self.tamanhos_camadas[i + 1]) * np.sqrt(2. / self.tamanhos_camadas[i])
            for i in range(self.num_camadas)
        ]
        self.vieses = [
            np.zeros((1, self.tamanhos_camadas[i + 1]))
            for i in range(self.num_camadas)
        ]
    
    # Função de ativação ReLU (Retifica valores negativos para zero)
    def relu(self, x):
        return np.maximum(0, x)

    # Derivada da função ReLU
    def relu_derivada(self, x):
        return (x > 0).astype(int)

    # Propagação para frente (cálculo das saídas)
    def propagacao_frente(self, entradas):
        self.ativacoes = [entradas]
        self.valores_z = []

        for i in range(self.num_camadas):
            z = np.dot(self.ativacoes[-1], self.pesos[i]) + self.vieses[i]
            self.valores_z.append(z)

            if i == self.num_camadas - 1:  # Camada de saída
                a = z  # Função identidade
            else:
                a = self.relu(z)
            self.ativacoes.append(a)
        return self.ativacoes[-1]

    # Função de perda: Erro Quadrático Médio (MSE)
    def calcular_mse(self, previsoes, verdadeiros):
        return np.mean((previsoes - verdadeiros) ** 2)

    # Cálculo do RMSE (raiz do MSE)
    def calcular_rmse(self, previsoes, verdadeiros):
        mse = self.calcular_mse(previsoes, verdadeiros)
        return np.sqrt(mse)

    # Retropropagação para ajustar pesos e vieses
    def retropropagacao(self, verdadeiros):
        m = verdadeiros.shape[0]
        verdadeiros = verdadeiros.reshape(-1, 1)

        gradientes_ativacoes = [(self.ativacoes[-1] - verdadeiros) / m]

        for i in reversed(range(self.num_camadas)):
            dz = (
                gradientes_ativacoes[0]
                if i == self.num_camadas - 1
                else gradientes_ativacoes[0] * self.relu_derivada(self.valores_z[i])
            )
            dw = np.dot(self.ativacoes[i].T, dz)
            db = np.sum(dz, axis=0, keepdims=True)

            dw += 2 * 0.01 * self.pesos[i]  # Regularização L2

            if i > 0:
                gradientes_ativacoes.insert(0, np.dot(dz, self.pesos[i].T))

            self.pesos[i] -= self.taxa_aprendizado * dw
            self.vieses[i] -= self.taxa_aprendizado * db

    # Treinamento do modelo
    def treinar(self, entradas, verdadeiros, epocas, taxa_aprendizado):
        self.taxa_aprendizado = taxa_aprendizado
        for epoca in range(epocas):
            previsoes = self.propagacao_frente(entradas)
            rmse = self.calcular_rmse(previsoes, verdadeiros)
            erro_percentual_medio = np.mean(np.abs(previsoes - verdadeiros) / np.maximum(np.abs(verdadeiros), 1e-10)) * 100

            self.retropropagacao(verdadeiros)

            if epoca % 10 == 0 or epoca == epocas - 1:
                print(f"Época {epoca} - RMSE: {rmse:.4f}, Erro percentual médio: {erro_percentual_medio:.2f}%")

    # Avaliação do modelo
    def avaliar(self, entradas, verdadeiros):
        previsoes = self.propagacao_frente(entradas)
        rmse = self.calcular_rmse(previsoes, verdadeiros)
        erro_percentual_medio = np.mean(np.abs(previsoes - verdadeiros) / np.maximum(np.abs(verdadeiros), 1e-10)) * 100
        print(f"RMSE: {rmse:.4f}")
        print(f"Erro percentual médio: {erro_percentual_medio:.2f}%")
        return rmse, erro_percentual_medio

# ==================== Normalização de Dados ====================
def normalizar_dados(entradas):
    desvio_padrao = entradas.std(axis=0)
    desvio_padrao[desvio_padrao == 0] = 1
    return (entradas - entradas.mean(axis=0)) / desvio_padrao

# ==================== Carregamento e Execução ====================
caminho_treino = './data/train_house_price.csv'
caminho_teste = './data/test_house_price.csv'

# Carregar dados
dados_treino = pd.read_csv(caminho_treino)
X_treino = normalizar_dados(dados_treino.iloc[:, :-1].values)
y_treino = dados_treino.iloc[:, -1].values

dados_teste = pd.read_csv(caminho_teste)
X_teste = normalizar_dados(dados_teste.iloc[:, :-1].values)
y_teste = dados_teste.iloc[:, -1].values

# Configuração da rede
rede = Regressao(entradas_tamanho=X_treino.shape[1], camadas_ocultas=[3, 6, 9, 3, 6, 9], saidas_tamanho=1)

# Treinamento
rede.treinar(X_treino, y_treino, epocas=10000, taxa_aprendizado=0.0001)

# Avaliação
rede.avaliar(X_teste, y_teste)
