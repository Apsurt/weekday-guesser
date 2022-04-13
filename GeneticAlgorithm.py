import numpy as np

class Population:
    def __init__(self, structure, size, accuracy, elite_percentage = 0.25, peasant_percentage = 0.1, mutation_rate = 0.05, natural_selection_rate = 1.85):
      self.size = size
      self.structure = structure
      self.length = 0
      for layer in range(1, len(structure)):
        self.length += (structure[layer-1]*structure[layer] + structure[layer]) * accuracy
      self.accuracy = accuracy
      self.elite_percentage = elite_percentage
      self.elite_number = int(np.ceil(self.size * self.elite_percentage))
      self.peasant_percentage = peasant_percentage
      self.peasant_number = int(np.ceil(self.size * self.peasant_percentage))
      self.mutation_rate = mutation_rate
      self.natural_selection_rate = natural_selection_rate
      self.generations = [self.random_generation()]
      self.fitted = []

    def random_generation(self):
      return [[np.random.randint(0,2) for _ in range(self.length)] for _ in range(self.size)]

    def read_genes(self, specimen):
      genes = [specimen[x: self.accuracy+x] for x in range(0, len(specimen), self.accuracy)]
      for indx, gene in enumerate(genes):
        dec = 0
        for idx, bit in enumerate(range(len(gene)-1, -1, -1)):
          dec += gene[idx] * pow(2, bit)
        genes[indx] = dec
      highest = max(genes)
      power = 0
      while highest > 1:
        highest = highest/10
        power += 1
      genes = [gene/pow(10,power) for gene in genes]
      return genes

def main():
  pop = Population([1,2], 1, 10)
  print(pop.length)
  print(pop.generations)
  print(pop.read_genes(pop.generations[0][0]))

#main()