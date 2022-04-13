import datetime
from NeuralNetwork import NN
import numpy as np
from GeneticAlgorithm import Population

def generate_random_date(low_year=1, high_year=2101):
  year = np.random.randint(low_year,high_year)
  month = np.random.randint(1,13)
  leap = False
  if((year % 400 == 0) or  
     (year % 100 != 0) and  
     (year % 4 == 0)):
       leap = True
  days = [31,28,31,30,31,30,31,31,30,31,30,31]
  if leap:
    days[1] = 29
  day = np.random.randint(1,days[month-1]+1)
  return [year,month,day]

def score(neural_network, iterations):
  correct = 0
  for i in range(0,iterations):
    rnd_date = generate_random_date(2000,2023)
    #print(rnd_date)
    algo_out = datetime.datetime(*rnd_date,0,0,0,0).weekday()
    #print(weekdays[algo_out])
    nn_out = neural_network.calculate(rnd_date, neural_network.sigmoid(4))
    #print(weekdays[nn_out])
    if algo_out == nn_out:
      correct += 1
  print(correct/iterations)
  
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

def main():
  iter = 100
  nn = NN([3,5,7])
  score(nn,iter)
  pop = Population([3,5,7], 1, 4)
  genes = pop.read_genes(pop.generations[0][0])
  nn.set_wb(genes)
  score(nn,100)

main()