from datetime import time
import numpy
import pandas
from random import randint, random, choice

city_coordinates = pandas.read_csv(
    'city_position.csv', index_col='name of city')
print(city_coordinates)

distances = []
city_arr = []

for i in range(len(city_coordinates)):
    city = city_coordinates.loc[i + 1]
    city_arr.append(numpy.array(city[0], city[1]))

for city in city_arr:
    del city_arr[0]
    for next_city in city_arr:
        distances.append(numpy.linalg.norm(next_city - city))

for i in range(len(city_coordinates)):
    city = city_coordinates.loc[i + 1]
    city_arr.append(numpy.array(city[0], city[1]))
max_dist = max(distances)
best_before_chromosome = []

# Procedure tsp - generate - initial - population, start population with population size pop_size


def initialize_population(pop_size):
    population = []
    for i in range(pop_size):
        chromosome = []
        chromosome.append(1)
        print("no of cities = " + str(len(city_coordinates.index)))
        while len(chromosome) is not len(city_coordinates.index):
            index = randint(2, len(city_coordinates.index))
            if index not in chromosome:
                chromosome.append(index)
        print(chromosome)
        population.append(chromosome)
    return population


# TSP fitness function for GA
def fitness(chromosome):
    sum = 0
    for i in range(len(chromosome) - 1):
        gene = chromosome[i]
        next_gene = chromosome[i + 1]
        sum += max_dist - \
            numpy.linalg.norm(city_arr[next_gene] - city_arr[gene])
    sum += max_dist - numpy.linalg.norm(city_arr[next_gene] - city_arr[0])
    return sum


def roulette_wheel_selection(population):
    fitness_sum = 0
    mating_pool = []
    for chromosome in population:
        fitness_sum += fitness(chromosome)

    for chromosome in population:
        r = random()
        j = 1
        sum = 0
        while sum < r:
            sum += fitness(population[j])
            j += 1
        mating_pool.append(population[j - 1])
    return mating_pool


def survival_of_the_fittest(population):
    j = 0
    for i in range(len(population)):
        if fitness(population[i]) > fitness(population[j]):
            j = i
    print("fittest = " + str(fitness(population[j])))
    return j

# def tournament_selection(population):
# DO LATER MAYBE


def circular_operator_encode(chromosome):
    unvisited_cities = list(range(1, len(chromosome) + 1))
    # print(unvisited_cities)
    # print(chromosome)
    for i in range(len(chromosome)):
        j = 0
        while chromosome[i] != unvisited_cities[j]:
            j += 1
        del unvisited_cities[j]
        chromosome[i] = j
    return chromosome


def circular_operator_decode(chromosome):
    unvisited_cities = list(range(1, len(chromosome) + 1))
    for i in range(len(chromosome)):
        j = chromosome[i]
        chromosome[i] = unvisited_cities[j]
        del unvisited_cities[j]
    return chromosome


def crossover(father, mother, p_c):

    if random() > p_c:
        return father, mother
    else:
        cut_point = randint(1, len(father) - 1)
        father_enc = circular_operator_encode(father.copy())
        mother_enc = circular_operator_encode(mother.copy())
        temp = mother_enc[cut_point]
        mother_enc[cut_point] = father_enc[cut_point]
        father_enc[cut_point] = temp
        return circular_operator_decode(father_enc), circular_operator_decode(mother_enc)


def mutate(chromosome, p_mu):
    if random() > p_mu:
        return chromosome
    else:
        mutation_point1 = randint(1, len(chromosome) - 1)
        mutation_point2 = randint(1, len(chromosome) - 1)
        temp = chromosome[mutation_point1]
        chromosome[mutation_point1] = chromosome[mutation_point2]
        chromosome[mutation_point2] = temp
        return chromosome


def reproduce(population, mating_pool, p_c, p_mu):
    new_generation = []
    fittest_index = survival_of_the_fittest(population)
    new_generation.append(population[fittest_index])
    try:
        mating_pool.remove(population[fittest_index])
    except:
        pass
    while len(mating_pool) > 1:
        father_index = choice(range(len(mating_pool)))
        father = mating_pool[father_index]
        del mating_pool[father_index]
        mother_index = choice(range(len(mating_pool)))
        mother = mating_pool[mother_index]
        del mating_pool[mother_index]
        son, daughter = crossover(father, mother, p_c)
        son = mutate(son, p_mu)
        daughter = mutate(daughter, p_mu)
        new_generation.append(son)
        new_generation.append(daughter)
    while mating_pool:
        new_generation.append(mating_pool[0])
        del mating_pool[0]
    return new_generation


def genetic_algorithm(pop_size, N_g, p_c, p_u):
    population = initialize_population(pop_size)
    for i in range(N_g):
        mating_pool = roulette_wheel_selection(population)
        population = reproduce(population, mating_pool, p_c, p_u)
    fittest_index = survival_of_the_fittest(population)
    return population[fittest_index]


result = genetic_algorithm(50, 100, 0.6, 0.001)
print("result = " + str(result))
