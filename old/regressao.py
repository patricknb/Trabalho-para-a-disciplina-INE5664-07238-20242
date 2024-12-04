import numpy as np

class Regressao:
    def __init__(self, input_size, hidden_layers, output_size):
        self.layer_sizes = [input_size] + hidden_layers + [output_size]
        self.num_layers = len(self.layer_sizes) - 1

        self.weights = [
            np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * np.sqrt(2. / self.layer_sizes[i])
            for i in range(self.num_layers)
        ]
        self.biases = [
            np.zeros((1, self.layer_sizes[i + 1]))
            for i in range(self.num_layers)
        ]
    
    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivada(self, x):
        return (x > 0).astype(int)

    def forward(self, X):
        self.activations = [X]
        self.z_values = []

        for i in range(self.num_layers):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)

            if i == self.num_layers - 1:
                a = z  # Sem ativação na camada de saída para regressão/identidade
            else:
                a = self.relu(z)
            self.activations.append(a)
        return self.activations[-1]

    def compute_loss(self, y_pred, y_true):
        """
        Calcula o erro quadrático médio (MSE).
        """
        return np.mean((y_pred - y_true) ** 2)

    def compute_rmse(self, y_pred, y_true):
        """
        Calcula o erro quadrático médio (RMSE).
        """
        mse = self.compute_loss(y_pred, y_true)
        return np.sqrt(mse)

    def compute_error_percentage(self, rmse, y_true):
        """
        Calcula o erro percentual médio do RMSE em relação aos valores reais.
        """
        proporcoes = rmse / y_true
        return np.mean(proporcoes) * 100  # Convertido para porcentagem

    def backward(self, y_true):
        m = y_true.shape[0]
        y_true = y_true.reshape(-1, 1)

        d_activations = [(self.activations[-1] - y_true) / m]

        for i in reversed(range(self.num_layers)):
            dz = (
                d_activations[0]
                if i == self.num_layers - 1
                else d_activations[0] * self.relu_derivada(self.z_values[i])
            )
            dw = np.dot(self.activations[i].T, dz)
            db = np.sum(dz, axis=0, keepdims=True)

            dw += 2 * 0.01 * self.weights[i]  # Regularização L2

            if i > 0:
                d_activations.insert(0, np.dot(dz, self.weights[i].T))

            self.weights[i] -= self.learning_rate * dw
            self.biases[i] -= self.learning_rate * db

    def train(self, X, y, epochs, learning_rate):
        self.learning_rate = learning_rate
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y_pred, y)
            self.backward(y)

            if epoch % 100 == 0 or epoch == epochs - 1:
                rmse = self.compute_rmse(y_pred, y)
                print(f"Epoca {epoch} RMSE = {rmse:.4f}")

    def evaluate(self, X, y):
        """
        Avalia o modelo e retorna RMSE e erro percentual médio.
        """
        y_pred = self.forward(X)
        rmse = self.compute_rmse(y_pred, y)

        error_percentage = self.compute_error_percentage(rmse, y_pred)
        print(f"RMSE: {rmse:.4f}")
        print(f"Erro percentual médio: {error_percentage:.2f}%")
        return error_percentage

    def save_pesos(self, filename):
        weights_dict = {f"weight_{i}": w for i, w in enumerate(self.weights)}
        biases_dict = {f"bias_{i}": b for i, b in enumerate(self.biases)}
        np.savez(filename, **weights_dict, **biases_dict)

    def load_pesos(self, filename):
        data = np.load(filename)
        self.weights = [data[f"weight_{i}"] for i in range(len(self.weights))]
        self.biases = [data[f"bias_{i}"] for i in range(len(self.biases))]
