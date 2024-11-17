import random


class Neuronio:

    def __init__(self) -> None:
        self.__atributos = []
        self.__pesos_sinapticos = []
        self.__vies = random.randint(1, 100)
        self.__v = 0
        self.__y = 0

    def set_atributos(self, atributos:list):
        self.__atributos = atributos

    def gerar_pesos_aleatorios(self, nro_atributos:int):
        for n in range(nro_atributos):
            self.__pesos_sinapticos.append((random.randint(1,50)/100))

    def combinador_linear(self):
        for i in range(len(self.__atributos)):
            self.__v = self.__atributos[i] * self.__pesos_sinapticos[i]
        self.__v = self.__v + self.__vies

    def funcao_de_ativacao(self):
        self.__y = self.__v * (self.__v > 0)
        
    