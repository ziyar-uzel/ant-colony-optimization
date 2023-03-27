import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import random
import numpy as np
from src.TSPData import TSPData


# TSP problem solver using genetic algorithms.
class GeneticAlgorithm:

    # Constructs a new 'genetic algorithm' object.
    # @param generations the amount of generations.
    # @param popSize the population size.
    def __init__(self, generations, pop_size, elite_size, mutation_rate):
        self.generations = generations
        self.pop_size = pop_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate


    # Knuth-Yates shuffle, reordering a array randomly
    # @param chromosome array to shuffle.
    def shuffle(self, chromosome):
        n = len(chromosome)
        for i in range(n):
            r = i + int(random.uniform(0, 1) * (n - i))
            swap = chromosome[r]
            chromosome[r] = chromosome[i]
            chromosome[i] = swap
        return chromosome


    # creates a population
    # returns a 2D array containing the chromosomes
    def generate_initial_population(self, list):
        n = len(list)
        # create population array
        res = np.zeros((self.pop_size, n))
        for i in range(self.pop_size):
            temp = self.shuffle(list)
            res[i] = temp
        return res


    # returns the fitness value of a chromosome
    def calculate_fitness_of_chromosome(self, chromosome, product_to_product, product_start_distances, product_end_distances):

        start_distance = product_start_distances[int(chromosome[0])]
        end_distance = product_end_distances[int(chromosome[len(chromosome) - 1])]
        total_distance = 0

        n = len(chromosome)
        for i in range(n - 1):
            from_product = chromosome[i]
            to_product = chromosome[i + 1]
            distance_between_products = product_to_product[int(from_product)][int(to_product)]
            total_distance += distance_between_products

        total_distance += (start_distance + end_distance)
        fitness = 1 / total_distance
        return fitness


    # calculates the fitness ratio of each chromosome
    # returns an array of [[index_chromosome, fitness ratio] ... []]
    def fitness_ratio_of_chromosomes(self, population, product_to_product, product_start_distances, product_end_distances):
        row, col = population.shape
        total_fitness = 0
        for i in range(row):
            fitness = self.calculate_fitness_of_chromosome(population[i], product_to_product, product_start_distances, product_end_distances)
            total_fitness += fitness

        normalization = 100 / total_fitness

        fitness_ratios_with_chromosome_index = np.zeros((row, 2))
        for i in range(row):
            fitness = self.calculate_fitness_of_chromosome(population[i], product_to_product, product_start_distances, product_end_distances)
            fitness_ratios_with_chromosome_index[i] = [i, fitness * normalization]

        # get second column
        print("fitness ratios :", "\n", fitness_ratios_with_chromosome_index)
        return fitness_ratios_with_chromosome_index


    # rank the routes according to the fitness ratio
    def ranked_routes(self, population, fitness_ratios):
        res = fitness_ratios[fitness_ratios[:, 1].argsort()[::-1]]
        print("ranked population :", "\n", res)
        return res


    # calculates the cumulative sum for each chromosome
    # returns a numpy array of [[chromosome_index, cumulative sum] ... []]
    def generate_cumulative_sum_array(self, population, fr):
        row, col = population.shape

        # get second column
        fitness_ratios = fr[:, 1]
        cumulative_sum_array = [fitness_ratios[0]]

        for i in range(1, len(fitness_ratios)):
            cumulative_sum_array.append(fitness_ratios[i] + cumulative_sum_array[i - 1])

        cumulative_sum_with_chromosome_index = np.zeros((row, 2))
        for i in range(len(cumulative_sum_array)):
            cumulative_sum_with_chromosome_index[i] = [i, cumulative_sum_array[i]]

        print("cumulative sum : ", "\n", cumulative_sum_with_chromosome_index)
        return cumulative_sum_with_chromosome_index


    # selection done through Roulette wheel method
    # returns the index of the chromosome/(one parent) that is selected
    def select_parent(self, cumulative_sum_array, population):
        cumulative_values = cumulative_sum_array[:, 1]
        random_val = random.randint(0, 100)

        for i in range(len(cumulative_values)):
            if random_val <= cumulative_values[i]:
                return cumulative_sum_array[i][0]


    # copied code
    def cross_over(self, parent_a, parent_b):
        child = []
        childP1 = []
        childP2 = []

        random_index_of_chromosome_1 = int(random.random() * len(parent_a))
        random_index_of_chromosome_2 = int(random.random() * len(parent_a))

        start_of_chromosome_gene = min(random_index_of_chromosome_1, random_index_of_chromosome_2)
        end_of_chromosome_gene = max(random_index_of_chromosome_1, random_index_of_chromosome_2)

        for i in range(start_of_chromosome_gene, end_of_chromosome_gene):
            childP1.append(parent_a[i])

        childP2 = [item for item in parent_b if item not in childP1]

        #child = childP1 + childP2
        child = childP2[:start_of_chromosome_gene] + childP1 + childP2[start_of_chromosome_gene:]
        return child


    # mutate chromosome by swapping
    # with specified low probability (mutation_rate), two products will swap places in the route
    def mutate(self, chromosome):
        if random.random() < self.mutation_rate:
            n = len(chromosome)
            index_a = random.randint(0, n - 1)
            index_b = random.randint(0, n - 1)

            if index_a == index_b:
                while index_a == index_b:
                    index_b = random.randint(0, n - 1)

            # swap
            temp = chromosome[index_a]
            chromosome[index_a] = chromosome[index_b]
            chromosome[index_b] = temp

        return chromosome


    # returns an array of indices of the chromosomes that are selected for mating
    # elitism is used
    def  select_chromosomes_for_mating(self, ranked_routes, population, fitness_ratios):
        selection_results = []

        # elitism used here
        # add the indices of the first elite_size routes to the selection_results
        for i in range(self.elite_size):
            selection_results.append(ranked_routes[i][0])

        cumulative_sum_array = self.generate_cumulative_sum_array(population, fitness_ratios)

        # fill up the mating pool with the rest of the chromosomes
        # using the Roulette wheel method
        # since a chromosome can be selected twice for reproduction,
        # we can use the original cumulative sum array which also contains the elite chromosomes
        for j in range(self.pop_size - self.elite_size):
            selection_results.append(self.select_parent(cumulative_sum_array, population))

        return selection_results


    # A mating pool is created which contains the actual chromosomes
    # which are used to reproduce
    # Here, elitism is used
    def create_mating_pool(self, population, selection_results):
        mating_pool = []
        for i in range(len(selection_results)):
            if(type(selection_results[i]) == None):
                index = int(selection_results[i])
                mating_pool.append(population[index])
        return np.array(mating_pool)


    # this method creates our offspring population
    def breedPopulation(self, mating_pool):
        new_population = []

        # use elitism to retain the best routes from the current population
        for i in range(self.elite_size):
            new_population.append(mating_pool[i])

        # use cross over to fill out the rest of the next generation
        l = len(mating_pool) - self.elite_size
        # randomly select two chromosomes, cross over them, and add it to the new population
        for j in range(l):
            # select two random chromosomes from the mating pool
            mini_mating_pool = mating_pool[np.random.randint(mating_pool.shape[0], size=2), :]
            # cross over the two chromosomes
            child = self.cross_over(mini_mating_pool[0], mini_mating_pool[1])
            new_population.append(child)

        return np.array(new_population)


    # mutate the offspring population
    def mutate_population(self, offspring_population):
        mutated_population = []

        # iterate over the complete population
        # for each chromosome, if the randomly generated > mutation rate, then mutation occurs
        for i in range(len(offspring_population)):
            mutated_chromosome = self.mutate(offspring_population[i])
            mutated_population.append(mutated_chromosome)

        return np.array(mutated_population)


    # produces a new generation
    def generate_next_population(self, current_generation, product_to_product, product_start_distances, product_end_distances):

        # fitness ratios of current generation
        fitness_ratios_of_chromosomes = self.fitness_ratio_of_chromosomes(current_generation, product_to_product, product_start_distances, product_end_distances)

        # rank the routes in the current generation
        ranked_population = self.ranked_routes(current_generation, fitness_ratios_of_chromosomes)

        # select parents using the select_parent() function
        selection_results = self.select_chromosomes_for_mating(ranked_population, current_generation, fitness_ratios_of_chromosomes)
        print("selection_results :", "\n", selection_results)

        # create a mating pool
        mating_pool = self.create_mating_pool(current_generation, selection_results)
        print("mating_pool :", "\n", mating_pool)

        # create new population by breeding
        children = self.breedPopulation(mating_pool)
        print("offsprings :", "\n", children)

        # apply mutation
        next_generation = self.mutate_population(children)
        return next_generation



    # This method should solve the TSP.
    # @param pd the TSP data.
    # @return the optimized product sequence.
    def solve_tsp(self, tsp_data):
        list = [0, 1]

        product_to_product_matrix = tsp_data.get_distances()
        product_start_distances = tsp_data.get_start_distances()
        product_end_distances = tsp_data.get_end_distances()

        # Initialize the population randomly
        population = self.generate_initial_population(list)
        print("initial_pop : ", "\n", population)

        for i in range(self.generations):
            population = self.generate_next_population(population, product_to_product_matrix, product_start_distances, product_end_distances)
            print("generation number : ", i, "(after mutation)", "\n", population)
        return None



# # Assignment 2.b
# if __name__ == "__main__":
#     # parameters
#     population_size = 5
#     generations = 5
#     elite_size = 1
#     mutation_rate = 0.2
#
#     tsp_data = Tsp_data()
#
#     ga = GeneticAlgorithm(generations, population_size, elite_size, mutation_rate)
#     solution = ga.solve_tsp(tsp_data)
#
#
#     # persistFile = "./../tmp/productMatrixDist"
#     #
#     # #setup optimization
#     # tsp_data = TSPData.read_from_file(persistFile)
#     # ga = GeneticAlgorithm(generations, population_size)
#     #
#     # #run optimzation and write to file
#     # solution = ga.solve_tsp(tsp_data)
#     # tsp_data.write_action_file(solution, "./../data/TSP solution.txt")

# Assignment 2.b
if __name__ == "__main__":
    #parameters
    population_size = 20
    generations = 20
    elite_size = 1
    mutation_rate = 0.2
    persistFile = "./../tmp/productMatrixDist"

    #setup optimization
    tsp_data = TSPData.read_from_file(persistFile)
    ga = GeneticAlgorithm(generations, population_size, elite_size, mutation_rate)

    #run optimzation and write to file
    solution = ga.solve_tsp(tsp_data)
    tsp_data.write_action_file(solution, "./../data/TSP solution.txt")











# import os, sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
#
# import random
# from src.TSPData import TSPData
#
# # TSP problem solver using genetic algorithms.
# class GeneticAlgorithm:
#
#     # Constructs a new 'genetic algorithm' object.
#     # @param generations the amount of generations.
#     # @param popSize the population size.
#     def __init__(self, generations, pop_size):
#         self.generations = generations
#         self.pop_size = pop_size
#
#      # Knuth-Yates shuffle, reordering a array randomly
#      # @param chromosome array to shuffle.
#     def shuffle(self, chromosome):
#         n = len(chromosome)
#         for i in range(n):
#             r = i + int(random.uniform(0, 1) * (n - i))
#             swap = chromosome[r]
#             chromosome[r] = chromosome[i]
#             chromosome[i] = swap
#         return chromosome
#
#     # This method should solve the TSP.
#     # @param pd the TSP data.
#     # @return the optimized product sequence.
#     def solve_tsp(self, tsp_data):
#         list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17]
#         return list
