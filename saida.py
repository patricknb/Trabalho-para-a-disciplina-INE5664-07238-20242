import numpy as np


class Saida:

    def __init__(self, classe) -> None:
        self.__classe = classe
        
    def get_classe(self):
        return self.__classe

    def softmax(y):
        """Compute softmax values for each sets of scores in x."""
        e_y = np.exp(y - np.max(y))
        return e_y / e_y.sum()
    