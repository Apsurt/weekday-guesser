import numpy as np
from NeuralNetwork import NN
from time import time


class Population:
    def __init__(self,
                 structure,
                 size,
                 param_range,
                 test_func,
                 test_iter=100,
                 elite_percentage=0.25,
                 peasant_percentage=0,
                 mutation_rate=0.05,
                 natural_selection_rate=1.85):
        self.size = size
        self.structure = structure
        self.length = 0
        for layer in range(1, len(structure)):
            self.length += (structure[layer - 1] * structure[layer] +
                            structure[layer])
        self.range = param_range
        self.test_func = test_func
        self.test_iter = test_iter
        self.elite_percentage = elite_percentage
        self.elite_number = int(np.ceil(self.size * self.elite_percentage))
        self.peasant_percentage = peasant_percentage
        self.peasant_number = int(np.ceil(self.size * self.peasant_percentage))
        self.mutation_rate = mutation_rate
        self.natural_selection_rate = natural_selection_rate
        self.generations = [self.random_generation()]
        self.fitted = [self.fitness_generation()]

    def random_generation(self):
        return [[np.random.uniform(*self.range) for _ in range(self.length)]
                for _ in range(self.size)]

    def fitness_generation(self):
        inp_list, expected_list = self.test_func(self.test_iter)
        if len(inp_list) != len(expected_list):
            raise IndexError('Input list and expected list have to have same lenght')
        if len(inp_list) != self.test_iter:
            raise IndexError('Length of data to check has to be equal with number of iterations')
        fitted = []
        nn = NN(self.structure)
        for idx, specimen in enumerate(self.generations[-1]):
          #print('Fitness specimen num:', idx+1,'|',len(self.generations[-1]))
          nn.set_wb(specimen)
          correct = 0
          for i in range(0, self.test_iter):
            nn_out = nn.calculate(inp_list[i])
            if expected_list[i] == nn_out:
              correct += 1
          fitted.append(correct / self.test_iter)
        return fitted

    def generation_avg_score(self, n, point = 5):
        return round(sum(self.fitted[n])/len(self.fitted[n]), point)
    
    def sort_scores(self, fit_scores):
        temp_sorted = fit_scores.copy()
        template = fit_scores.copy()
        temp_sorted.sort(reverse=True)
        idx_list = []
        for i in temp_sorted:
            idx_list.append(template.index(i))
            template[template.index(i)] = -1
        return idx_list

    def mutate(self, specimen):
        for i in range(self.length):
            if np.random.randint(0, 100) <= self.mutation_rate * 100:
                specimen[i] = abs(specimen[i] - 1)
                return specimen
        return specimen

    def crossover(self, parent1, parent2):
        cut = round(self.length / 2)
        child = (parent1[:cut])
        for i in parent2[cut:]:
            child.append(i)
        return child

    def selection(self):
        new_generation = []
        fit_scores = self.sort_scores(self.fitted[-1])
        #passing best
        for i in range(self.elite_number):
            new_generation.append(self.generations[-1][fit_scores[i]])
        #flipping worst
        for i in range(self.peasant_number):
            worst_specimen = self.generations[-1][fit_scores[-i - 1]]
            for i in range(len(worst_specimen)):
                if worst_specimen[i] == 1:
                    worst_specimen[i] = 0
                else:
                    worst_specimen[i] = 1
            new_generation.append(worst_specimen)
        #breeding rest
        while len(new_generation) < self.size:
            parents = []
            for _ in range(2):
                target = np.random.uniform(0, sum(self.fitted[-1]))
                s = 0
                current_num = 0
                while s < target:
                    s += self.fitted[-1][fit_scores[current_num]]
                    current_num += 1
                parents.append(self.generations[-1][fit_scores[current_num -
                                                               1]])
            child = self.crossover(*parents)
            #print('Parents:', parents)
            child = self.mutate(child)
            #print('Child:', child)
            new_generation.append(child)
        return new_generation

    def run(self, epochs, interval = 10):
        start = time()
        for i in range(epochs+1):
            self.generations.append(self.selection())
            self.fitted.append(self.fitness_generation())
            if interval != None:
                try:
                    if i % interval == 0 or i==epochs:
                        print('Generation:', i, 'Average score:', self.generation_avg_score(i), 'Time remaining:', round((((time()-start)*epochs)/interval)-(((time()-start)*i)/interval), 2), 's')
                        start = time()
                except:
                    if i==epochs:
                        print('Generation:', i, 'Average score:', self.generation_avg_score(i))
                        start = time()

def main():
    pop = Population([1, 2], 2, 5)
    print(pop.length)
    print(pop.generations)
    #print(pop.read_genes(pop.generations[0][0]))


#main()
