from perceptron import *

features = np.array([[0, 1],
                     [1, 0],
                     [1, 1],
                     [0, 0]])

targets = np.array([[1],
                    [1],
                    [0],
                    [0]])

perceptron = Perceptron([2, 2, 1],
                        features, targets, learning_rate=0.5)

perceptron.train(epochs=10000)
perceptron.show_result()
