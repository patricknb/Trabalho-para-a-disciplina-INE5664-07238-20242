import numpy as np


class CamadaSaida:

    def __init__(self) -> None:
        pass
        
    def softmax(y):
        """Compute softmax values for each sets of scores in x."""
        e_y = np.exp(y - np.max(y))
        return e_y / e_y.sum()
    