import numpy as np

class NN:
    def __init__(self, structure):
        #structure
        self.structure = structure
        #print('Structure:', self.structure)

        #values
        self.values = [[0 for node in range(layer)]
                       for layer in self.structure]
        #print('Values:', self.values)

        #Weights
        self.weights = []
        for i in range(1, len(self.structure)):
            self.weights.append([[
                round(np.random.uniform(0, 1), 3)
                for j in range(self.structure[i - 1])
            ] for _ in range(self.structure[i])])
        #print('Weights:', self.weights)

        #Biases
        self.biases = [[
            round(np.random.uniform(0, 1), 3) for node in range(layer)
        ] for layer in self.structure[1:]]
        #print('Biases:', self.biases)

        #Activation Function
        self.activation_function = self.linear()

    def set_wb(self, genotype):
        idx = 0
        for layer in range(len(self.weights)):
            for node in range(len(self.weights[layer])):
                self.biases[layer][node] = genotype[idx]
                idx += 1
                #print('Bias:', self.biases[layer][node])
                for connection in range(len(self.weights[layer][node])):
                    self.weights[layer][node][connection] = genotype[idx]
                    idx += 1
                    #print('Weight:', self.weights[layer][node][connection])
        #print('Weights:', self.weights)
        #print('Biases:', self.biases)

    def sigmoid(self, point=9):
        def f(x):
          if abs(x) >= 700:
            return 0
          return round(1 / (1 + np.exp(-x)), point)
        return f

    def linear(self, point=9):
      def f(x):
        return round(x, point)
      return f

    def calculate(self, inp):
        if len(self.values[0]) == len(inp):
            self.values[0] = inp
        else:
            raise IndexError('Input and first layer length should be the same')
        #print('Before calculation:', self.values)
        for layer in range(len(self.values) - 1):
            for node in range(len(self.values[layer + 1])):
                self.values[layer + 1][node] = self.activation_function(
                    self.biases[layer][node] + sum([
                        x * y for x, y in zip(self.values[layer],
                                              self.weights[layer][node])
                    ]))
        #print('After calculation:', self.values)
        #print('Output:', self.values[-1])
        #print('Outcome:', self.values[-1].index(max(self.values[-1])))
        return self.values[-1].index(max(self.values[-1]))


def main():
    pass