# Importer les bibliothèques nécessaires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2)

# Étape 2 : Lire les données
df = pd.read_csv("dataset.csv")

# Étape 3 : Explorer les données
print("Dimensions du DataFrame:", df.shape)
print("Les premières lignes des données:")
print(df.head())
print("Distribution de la variable cible (0 et 1):")
print(df['target'].value_counts())

# Étape 4 : Préparer les données
X = df[['x1', 'x2']].values.T
Y = df['target'].values.reshape(1, -1)

# Étape 5 : Définir les hyperparamètres
m = X.shape[1]
n_x = 2
n_h = 10
n_y = 1
num_of_iters = 1000
learning_rate = 0.3


# Étape 6 : Définir la fonction d'activation sigmoïde
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Étape 7 : Initialiser les paramètres du modèle
def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x)
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)
    b2 = np.zeros((n_y, 1))

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }
    return parameters


# Étape 8 : Effectuer la propagation avant (Forward Propagation)
def forward_prop(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {
        "A1": A1,
        "A2": A2
    }
    return A2, cache


# Étape 9 : Calculer la perte (Loss)
def calculate_cost(A2, Y):
    cost = -np.sum(np.multiply(Y, np.log(A2)) + np.multiply(1 - Y, np.log(1 - A2))) / m
    cost = np.squeeze(cost)
    return cost


# Étape 10 : Effectuer la rétropropagation (Backward Propagation)
def backward_prop(X, Y, cache, parameters):
    A1 = cache["A1"]
    A2 = cache["A2"]

    W2 = parameters["W2"]

    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    grads = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }

    return grads


# Étape 11 : Mettre à jour les paramètres
def update_parameters(parameters, grads, learning_rate):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    new_parameters = {
        "W1": W1,
        "W2": W2,
        "b1": b1,
        "b2": b2
    }

    return new_parameters


# Étape 12 : Définir le modèle complet
def model(X, Y, n_x, n_h, n_y, num_of_iters, learning_rate, display_loss=False):
    parameters = initialize_parameters(n_x, n_h, n_y)

    for i in range(0, num_of_iters + 1):
        a2, cache = forward_prop(X, parameters)

        cost = calculate_cost(a2, Y)

        grads = backward_prop(X, Y, cache, parameters)

        parameters = update_parameters(parameters, grads, learning_rate)

        if display_loss:
            if (i % 100 == 0):
                print('Coût après l\'itération# {:d}: {:f}'.format(i, cost))

    return parameters


# Étape 13 : Entraîner le modèle
trained_parameters = model(X, Y, n_x, n_h, n_y, num_of_iters, learning_rate, display_loss=True)


# Étape 14 : Faire des prédictions
def predict(parameters, X):
    A2, cache = forward_prop(X, parameters)
    predictions = A2 > 0.5

    return predictions


# Étape 15 : Visualiser la frontière de décision
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y.reshape(200, ), cmap=plt.cm.Spectral)


plot_decision_boundary(lambda x: predict(trained_parameters, x.T), X, Y)
plt.savefig(r'C:\Users\achraf\PycharmProjects\TPsoft\trained_parameters.png', format='png')

# Étape 16 : Explorer différents paramètres
plt.figure(figsize=(15, 10))
hidden_layer_sizes = [1, 2, 3, 5, 10, 20]
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(2, 3, i + 1)
    plt.title('Hidden Layer of size %d' % n_h)

    parameters = model(X, Y, n_x, n_h, n_y, num_of_iters, learning_rate)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.savefig(r'C:\Users\achraf\PycharmProjects\TPsoft\parameter_exploration.png', format='png')

plt.show()
