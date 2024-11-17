from neuronio import Neuronio
import pandas as pd

class RedeNeural:
    
    def __init__(self) -> None:
        self.__camadas = []
        self.__saidas = []
        self.executar()

    def executar(self):
        qtd_epocas = int(input('quantas epocas no maximo: '))
        qtd_camadas = int(input('quantas camadas: '))
        qtd_saidas = int(input('quantas saidas: '))
        atributos = self.carregar_dados()
        qtd_atributos = len(atributos.columns) - 1

        self.gerar_camadas(qtd_camadas, qtd_atributos)
        self.gerar_saidas(qtd_saidas, atributos)

        for epoca in range(qtd_epocas):
            for index, row in atributos.iterrows():
                #print('index: {} row: {}'.format(index, row.tolist()))
                pass

    def carregar_dados(self):
        file_path = input('Arquivo: ')
        return pd.read_csv(file_path)

    def gerar_camadas(self, qtd_camadas, qtd_atributos):
        for n in range(qtd_camadas):
            temp_neu = []
            for m in range(qtd_atributos):
                temp_neu.append(Neuronio())
                print('camada {} neuronio {} criado!'.format(n, m))

            self.__camadas.append(temp_neu)

r = RedeNeural()