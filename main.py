import datetime
from NeuralNetwork import NN
import numpy as np
from GeneticAlgorithm import Population


def generate_random_date(low_year=1, high_year=2101):
    year = np.random.randint(low_year, high_year)
    month = np.random.randint(1, 13)
    leap = False
    if ((year % 400 == 0) or (year % 100 != 0) and (year % 4 == 0)):
        leap = True
    days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if leap:
        days[1] = 29
    day = np.random.randint(1, days[month - 1] + 1)
    return [year, month, day]


def score_data(iterations):
    inp_list = []
    expected_list = []
    for i in range(0, iterations):
        rnd_date = generate_random_date(2000, 2023)
        inp_list.append(rnd_date)
        expected_list.append(
            datetime.datetime(*rnd_date, 0, 0, 0, 0).weekday())
    return [inp_list, expected_list]


weekdays = [
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday',
    'Sunday'
]


def main():
    pop = Population([3, 7], 100, 15, score_data)
    print('Length:', pop.length)
    #print('Generations:', pop.generations)
    #print('Genes:', pop.read_genes(pop.generations[0][0]))
    #print('Genes:', pop.read_genes(pop.generations[0][1]))
    print('Fitted:', pop.fitted)
    print('Sorted:', pop.sort_scores(pop.fitted[-1]))
    pop.run(10,1)

main()