import numpy as np


# Função de ativação: Sinal
def sinal(x):
    u = x.item(0)
    if x > 0:
        return 1;
    return -1;


# Função de ativação: Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivada da função Sigmoid
def derivada_sigmoid(x):
    return x * (1 - x)


# Função de ativação: ReLU
def relu(x):
    return np.maximum(0, x)


# Derivada de ReLU
def derivada_relu(x):
    return np.where(x > 0, 1, 0)


# Função de perda: Entropia Cruzada Binária: calcula a diferença entre os valores reais e previstos no contexto de classificação binária.
def entropia_cruzada_binaria(y_verdadeiro, y_predito):
    
    y_predito = np.clip(y_predito, 1e-15, 1 - 1e-15)  # Evitar valores fora da faixa (0, 1)
    return -np.mean(y_verdadeiro * np.log(y_predito) + (1 - y_verdadeiro) * np.log(1 - y_predito))


# Função de perda: Erro Quadrático Médio (EQM)
def eqm(y_verdadeiro, y_predito):
    return np.mean((y_verdadeiro - y_predito) ** 2)