# Importar funções do arquivo funcoes.py
from funcoes_ativacao_perda import sigmoid, derivada_sigmoid, entropia_cruzada_binaria, eqm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# Estrutura da Rede Neural
class ClassificacaoBinaria:

    def __init__(self, entrada_tamanho, oculto_tamanho, saida_tamanho, taxa_aprendizado):
        
        # Número de atributos (colunas de entrada).
        self.entrada_tamanho = entrada_tamanho
        # Número de neurônios na camada oculta.
        self.oculto_tamanho = oculto_tamanho
        # Número de classes ou saídas (1 neste caso para classificação binária).
        self.saida_tamanho = saida_tamanho
        # Taxa para o gradiente descendente.
        self.taxa_aprendizado = taxa_aprendizado

        # Inicialização dos pesos e vieses com valores aleatórios pequenos
        self.pesos_entrada_oculto = np.random.randn(entrada_tamanho, oculto_tamanho) * 0.01
        self.vies_oculto = np.zeros((1, oculto_tamanho))
        self.pesos_oculto_saida = np.random.randn(oculto_tamanho, saida_tamanho) * 0.01
        self.vies_saida = np.zeros((1, saida_tamanho))


    # Calcula a saída da rede. Recebe como parâmetro o conjunto de dados de entrada.
    def propagacao_frente(self, entrada):

        # CAMADA 1: Propagação da camada de entrada para a oculta
        self.entrada_oculto = np.dot( entrada, self.pesos_entrada_oculto ) + self.vies_oculto
        self.saida_oculto = sigmoid( self.entrada_oculto )

        # CAMADA 2: Propagação da camada oculta para a camada de saída
        self.entrada_saida = np.dot( self.saida_oculto, self.pesos_oculto_saida ) + self.vies_saida
        self.saida_final = sigmoid( self.entrada_saida )
        
        return self.saida_final


    # Executa retropropagação (backpropagation) para atualizar os pesos.
    # - entrada: dados de entrada.
    # - y_verdadeiro: valores reais
    # - saida: saída gerada pela rede.
    def retropropagacao(self, entrada, y_verdadeiro, saida):

        # Calcula o erro da saída
        erro = y_verdadeiro - saida

        # Calcula o gradiente na camada de saída
        gradiente_saida = erro * derivada_sigmoid(saida)

        # Calcula o erro e o gradiente na camada oculta
        erro_oculto = np.dot( gradiente_saida, self.pesos_oculto_saida.T )
        gradiente_oculto = erro_oculto * derivada_sigmoid(self.saida_oculto)

        # Atualização dos pesos e vieses entre a camada oculta e a saída
        gradiente_pesos_saida = np.dot( self.saida_oculto.T, gradiente_saida )  # Matriz de ajustes para pesos
        gradiente_vies_saida = np.sum( gradiente_saida, axis=0, keepdims=True )  # Vetor de ajustes para vieses
        self.pesos_oculto_saida += gradiente_pesos_saida * self.taxa_aprendizado
        self.vies_saida += gradiente_vies_saida * self.taxa_aprendizado

        # Atualização dos pesos e vieses entre a camada de entrada e a oculta
        gradiente_pesos_oculto = np.dot (entrada.T, gradiente_oculto )  # Matriz de ajustes para pesos
        gradiente_vies_oculto = np.sum( gradiente_oculto, axis=0, keepdims=True )  # Vetor de ajustes para vieses
        self.pesos_entrada_oculto += gradiente_pesos_oculto * self.taxa_aprendizado
        self.vies_oculto += gradiente_vies_oculto * self.taxa_aprendizado


    # Treina a rede neural por um número determinado de épocas.
    def treinar(self, entrada, y_verdadeiro, epocas):
        print()
        perdas = []
        for epoca in range(epocas):
            saida = self.propagacao_frente(entrada)
            # Calcula a perda (erro) da rede usando a função de perda de entropia cruzada binária
            perda = entropia_cruzada_binaria(y_verdadeiro, saida)
            # Calcula a perda (erro) da rede usando a função de perda de erro quadrático médio
            # perda = eqm(y_verdadeiro, saida)
            perdas.append(perda)
            self.retropropagacao(entrada, y_verdadeiro, saida)
            if epoca % 200 == 0:
                print("Época " + str(epoca) + ", Perda: " + "{:.4f}".format(perda))
        return perdas


    # Realiza previsões para o conjunto de entrada.
    def prever(self, entrada):
        saida = self.propagacao_frente(entrada)
        return np.round(saida)

# Carregamento e preparação do conjunto de dados
caminho_arquivo = r".\dados\water_potability.csv"
dados = pd.read_csv(caminho_arquivo)

# Fonte: https://www.kaggle.com/datasets/uom190346a/water-quality-and-potability
#:: Variáveis independentes:
#   1. pH: The pH level of the water.
#   2. Hardness: Water hardness, a measure of mineral content.
#   3. Solids: Total dissolved solids in the water.
#   4. Chloramines: Chloramines concentration in the water.
#   5. Sulfate: Sulfate concentration in the water.
#   6. Conductivity: Electrical conductivity of the water.
#   7. Organic carbon: Organic carbon content in the water.
#   8. Trihalomethanes: Trihalomethanes concentration in the water.
#   9. Turbidity: Turbidity level, a measure of water clarity.
#   10. Potability: Target variable; indicates water
#:: Variável dependente:
#   Potability: Variável alvo; indica potabilidade da água com valores 1 (potável) e 0 (não potável).

print("\nTamanho original do conjunto de dados (linhas, colunas): ", dados.shape)

# Remove linhas com valores nulos
dados.dropna(inplace=True)

print("\nTamanho após remoção de linhas com valores nulos (linhas, colunas): ", dados.shape)

# 10 primeiras linhas do conjunto de dados
print("\n", dados.head(10))

# Separa o conjunto entre variáveis de entrada/independentes (X) e variável alvo/dependente (y)
X = dados.drop(columns="Potability").values
y = dados["Potability"].values.reshape(-1, 1)

# Normalização dos dados para trazer todas as variáveis para a mesma escala
normalizador = StandardScaler()
X = normalizador.fit_transform(X)

# Divisão dos dados entre treino e teste na proporção de 90/10
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.10, random_state=777)

# Configura o NumPy para evitar quebras de linha ao exibir arrays
np.set_printoptions(linewidth=np.inf)

print("\nPrimeiras 10 linhas de X_treino (conjunto de TREINO), de um total de ", X_treino.shape[0], " linhas, seguidas pelo vetor de variáveis dependentes (y_treino):")
print(X_treino[:10])
print(y_treino[:10])

print("\nPrimeiras 10 linhas de X_teste (conjunto de TESTE), de um total de ", X_teste.shape[0], " linhas, seguidas pelo vetor de variáveis dependentes (y_teste):")
print(X_teste[:10])
print(y_teste[:10])

# Criação da Rede Neural de Classificação Binária: entrada_tamanho, oculto_tamanho, saida_tamanho, taxa_aprendizado=0.01:
rede_Class_Binaria = ClassificacaoBinaria(X_treino.shape[1], 10, 1, 0.01)

# Treinamento da Rede Neural
perdas = rede_Class_Binaria.treinar(X_treino, y_treino, epocas=4001)

# Avaliação da Rede Neural
y_predito = rede_Class_Binaria.prever(X_teste)

# Calcula o total de previsões corretas
previsoes_corretas = 0
for yreal, ypred in zip(y_teste, y_predito):
    if yreal == ypred: 
        previsoes_corretas += 1

# Calcula a acurácia do modelo
acuracia = previsoes_corretas / len(y_teste)
print("\n:: Acurácia do modelo: {:.4f}".format(acuracia))

# Exibe a curva de perda
plt.plot(perdas)
plt.title("Perda ao longo das épocas")
plt.xlabel("Épocas")
plt.ylabel("Perda")
plt.show()

print()