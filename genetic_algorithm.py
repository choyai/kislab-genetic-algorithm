import time
import numpy
import pandas
from random import randint, random, choice
import copy
from math import sqrt

# GLOBALS
# setup code: import .csv data using pandas, convert to array, find the maximum distance between cities
city_coordinates = pandas.read_csv(
    'world_position.csv', index_col='name of city')
print(city_coordinates)

distances = []
# array of all cities, contains array of x, y coordinates
city_arr = []

for i in range(len(city_coordinates)):
    city = city_coordinates.loc[i + 1]
    city_arr.append([city[0], city[1]])

# for city in city_arr:
#     del city_arr[0]
#     for next_city in city_arr:
#         distances.append(numpy.linalg.norm(next_city - city))
#
# for i in range(len(city_coordinates)):
#     city = city_coordinates.loc[i + 1]
#     city_arr.append(numpy.array(city[0], city[1]))

# max distance between cities, GA is a maximization algorithm
# max_dist = max(distances)


def euclidean_dist(current, next):
    return sqrt((next[0] - current[0])**2 + (next[1] - current[1])**2)
# TSP fitness function for GA
# fitness is the sum of distances between each city on the path subtracted from the maximum distance


def fitness(chromosome):
    sum = 0
    for i in range(len(chromosome) - 1):
        gene = chromosome[i]
        next_gene = chromosome[i + 1]
        sum += euclidean_dist(city_arr[gene - 1], city_arr[next_gene - 1])
    sum += euclidean_dist(city_arr[0], city_arr[next_gene - 1])
    sum = 1 / sum
    return sum

# cost is total distance(would be the fitness fn for the minimization problem)


def cost(chromosome):
    sum = 0
    for i in range(len(chromosome) - 1):
        gene = chromosome[i]
        next_gene = chromosome[i + 1]
        sum += euclidean_dist(city_arr[gene - 1], city_arr[next_gene - 1])
    sum += euclidean_dist(city_arr[0], city_arr[next_gene - 1])
    return sum

# chromosome class


class Chromosome:
    def __init__(self, city_dataframe, array=[]):
        self.length = len(city_dataframe.index)
        if array == []:
            self.array = [1]
            while len(self.array) is not self.length:
                index = randint(2, self.length)
                if index not in self.array:
                    self.array.append(index)
        else:
            self.array = array
        self.fitness = fitness(self.array)
        self.cost = cost(self.array)

    def __str__(self):
        return str(self.array)

    def __copy__(self):
        b = Chromosome(city_coordinates)
        b.array = self.array
        return b

    def __deepcopy__(self, memo):
        return Chromosome(copy.deepcopy(city_coordinates, self.array, memo))


# Procedure tsp-generate-initial-population, start population with population size pop_size


def initialize_population(pop_size):
    print("initialzing population..")
    population = []
    for i in range(pop_size):
        chromosome = Chromosome(city_coordinates)
        population.append(chromosome)
    return population
# roulette_wheel_selection assigns a probabilty for each chromosome based on it's fitness value
# returns execution time


def roulette_wheel_selection(population):
    start = time.time()
    fitness_sum = 0
    cost_sum = 0
    mating_pool = []
    for chromosome in population:
        fitness_sum += chromosome.fitness
        cost_sum += chromosome.cost

    for chromosome in population:
        r = random()
        j = 0
        sum = 0
        while sum < r:
            sum += population[j].fitness / fitness_sum
            j += 1
        mating_pool.append(population[j - 1])
    end = time.time()
    return mating_pool, end - start, cost_sum


def survival_of_the_fittest(population, fitness_sum, dict):
    j = 0
    k = 0
    for i in range(len(population)):
        if population[i].fitness > population[j].fitness:
            j = i
        elif population[i].fitness < population[k].fitness:
            k = i
    print("fittest = " + str(population[j].fitness))
    print("cost = " + str(population[j].cost))

    dict["best"].append(population[j].cost)
    dict["avg"].append(fitness_sum / (len(population)))
    dict["worst"].append(population[k].cost)

    return j

# def tournament_selection(population):
# DO LATER MAYBE


def circular_operator_encode(chromosome):
    unvisited_cities = list(range(1, chromosome.length + 1))
    # print(unvisited_cities)
    # print(chromosome.array)
    new_array = list(range(1, chromosome.length + 1))
    for i in range(chromosome.length):
        j = 0
        while chromosome.array[i] != unvisited_cities[j]:
            j += 1
        del unvisited_cities[j]
        new_array[i] = j
    return Chromosome(city_coordinates, new_array)


def circular_operator_decode(chromosome):
    unvisited_cities = list(range(1, chromosome.length + 1))
    new_array = list(range(1, chromosome.length + 1))
    for i in range(chromosome.length):
        j = chromosome.array[i]
        new_array[i] = unvisited_cities[j]
        del unvisited_cities[j]
    return Chromosome(city_coordinates, new_array)


def crossover(father, mother, p_c):

    if random() > p_c:
        return father, mother
    else:
        cut_point = randint(1, len(father.array) - 1)
        father_enc = circular_operator_encode(father)
        mother_enc = circular_operator_encode(mother)
        temp = mother_enc.array[cut_point]
        mother_enc.array[cut_point] = father_enc.array[cut_point]
        father_enc.array[cut_point] = temp
        return circular_operator_decode(father_enc), circular_operator_decode(mother_enc)


def mutate(chromosome, p_mu):
    if random() > p_mu:
        return chromosome
    else:
        mutation_point1 = randint(1, chromosome.length - 1)
        mutation_point2 = randint(1, chromosome.length - 1)
        temp = chromosome.array[mutation_point1]
        chromosome.array[mutation_point1] = chromosome.array[mutation_point2]
        chromosome.array[mutation_point2] = temp
        return chromosome

# reproduce using mating pool
# returns exec_time


def reproduce(population, mating_pool, p_c, p_mu, fitness_sum, dict):
    start = time.time()
    new_generation = []
    fittest_index = survival_of_the_fittest(population, fitness_sum, dict)
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
    while len(new_generation) < len(population):
        new_generation.append(mating_pool[0])
        del mating_pool[0]
    end = time.time()
    return new_generation, end - start


def genetic_algorithm(pop_size, N_g, p_c, p_u, iteration):
    start = time.time()
    dict = {"generation": [], "best": [], "avg": [], "worst": [], "time": []}
    dict["generation"].append(1)
    dict["time"].append(0)
    population = initialize_population(pop_size)
    print("initial pop: ")
    for i in population:
        print(i)
    for i in range(N_g - 1):
        time_per_gen = time.time()
        mating_pool, selection_time, fitness_sum = roulette_wheel_selection(
            population)
        population, mating_time = reproduce(
            population, mating_pool, p_c, p_u, fitness_sum, dict)
        print("GEN" + str(i + 1) + " of iteration " + str(iteration + 1))
        # print("GEN" + str(i) + " took " + str(selection_time) +
        # " seconds to select and " + str(mating_time) + " to reproduce")
        rem_time = (time.time() - time_per_gen) * \
            ((N_g - i) + N_g * (100 - iteration))
        print("remaining time = " + str(rem_time) +
              "seconds + (" + str(rem_time / 3600) + " hours)")
        gen_time = time.time() - start
        dict["generation"].append(i + 2)
        dict["time"].append(gen_time)
    fittest_index = survival_of_the_fittest(population, fitness_sum, dict)
    print("final pop: ")
    for i in population:
        print(i)
    return population[fittest_index], dict


best_dict = {}
best_result = 0
for i in range(100):
    result, dict = genetic_algorithm(100, 10000, 1.0, 0.05, i)
    if result.fitness > best_result:
        best_dict = dict.copy()
        best_result = result.fitness
    print("result = " + str(result))
    print("cost = " + str(result.cost))
    gen_dataf = pandas.DataFrame(best_dict)
    gen_dataf.to_csv("result.csv", index=False)
print(gen_dataf)
