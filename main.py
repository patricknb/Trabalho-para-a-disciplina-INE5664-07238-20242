import pandas as pd
from rede_neural import RedeNeural


def main():

    # 1. Carregar os dados de treinamento
    #caminho_arquivo = 'esrb-rating.csv'
    caminho_arquivo_treino = input('Digite o caminho para o arquivo de treino(ex: esrb-rating.csv)\n>')
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

    qtd_camadas = int(input('Digite a quantidade de camadas desejadas.\n>'))
    camadas_ocultas = []
    for n in range(qtd_camadas):
        temp_neuronios = int(input('Digite a quantidade desejada de neuronios na camada {}.\n>'.format(n+1)))
        camadas_ocultas.append(temp_neuronios)
    #camadas_ocultas = [5, 5]  # Número de neurônios nas camadas ocultas
    print(camadas_ocultas)
    nn = RedeNeural(tamanho_entrada, camadas_ocultas, tamanho_saida)

    # 4. Carregar os dados de teste
    caminho_arquivo_teste = input('Digite o caminho para o arquivo de teste(ex: test_esrb.csv)\n>')
    dados_teste = pd.read_csv(caminho_arquivo_teste)

    # 7. Pré-processamento para os dados de teste
    entradas_teste = dados_teste.iloc[:, :-1].values  # Todas as colunas, exceto a última
    rotulos_teste = dados_teste.iloc[:, -1].values    # Última coluna (os rótulos de classe)
    entradas_teste_normalizadas = normalize_data(entradas_teste)
    rotulos_teste_one_hot = pd.get_dummies(rotulos_teste).values

    # 5. Treinamento da rede neural
    epocas = int(input('Digite a quantidade de epocas desejada.\n>'))
    batch_size = 32 #32 funcionou, preciso checar se funciona com outro data base :\
    taxa_aprendizado = float(input('Digite a taxa de aprendizado desejada(ex: 0.01).\n>'))
    nn.train(entradas_normalizadas, rotulos_one_hot, epocas, batch_size, taxa_aprendizado)

    # 6. Salvar pesos e bias após o treinamento
    nn.save_pesos('modelo_treinado.npz')

    # 7. Avaliar o modelo com os dados de teste após o treinamento
    #precisao_treino = nn.evaluate(entradas_normalizadas, rotulos_one_hot)
    #print(f'Precisão nos dados de treino: {precisao_treino:.2%}')
    print('==Teste==')
    precisao_teste = nn.evaluate(entradas_teste_normalizadas, rotulos_teste_one_hot)
    print(f'Precisão nos dados de teste: {precisao_teste:.2%}')


if __name__ == "__main__":
    main()
