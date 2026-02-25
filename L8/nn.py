import numpy as np

class Perceptron:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # initialize weights and bias
        self.w = np.zeros(n_features)
        self.b = 0

        # convert label to {-1, +1}
        y = np.where(y<=0, -1, 1)

        # training loop
        for _ in range(self.epochs):
            for i in range(n_samples):
                linear_output = np.dot(X[i], self.w) + self.b
                y_pred = 1 if linear_output >= 0 else -1

                # perceptron update rule
                update = self.lr * (y[i] - y_pred)
                self.w += update * X[i]
                self.b += update
        
    def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b
        return np.where(linear_output >= 0, 1, 0)
    

X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

y = np.array([0,0,0,1])

p = Perceptron(lr=0.1, epochs=20)

if __name__ == "__main__":
    p.fit(X, y)
    print("Predictions:", p.predict(X))
