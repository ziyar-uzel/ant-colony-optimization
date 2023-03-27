import os, sys

from src.SurroundingPheromone import SurroundingPheromone

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import traceback
import copy
import Coordinate
import Direction

# Class that holds all the maze data. This means the pheromones, the open and blocked tiles in the system as
# well as the starting and end coordinates.
class Maze:

    # Constructor of a maze
    # @param walls int array of tiles accessible (1) and non-accessible (0)
    # @param width width of Maze (horizontal)
    # @param length length of Maze (vertical)
    def __init__(self, walls, width, length, pheromone, evap=0.1):
        self.walls = list(map(list, zip(*walls)))
        self.length = length
        self.width = width
        self.start = None
        self.end = None
        self.evap = evap
        self.pheromone = pheromone
        self.initialize_pheromones()
        self.east = 0
        self.north = 1
        self.west = 2
        self.south = 3
        self.dead_ants = copy.deepcopy(self.walls)

    # Initialize pheromones to a start value.
    def initialize_pheromones(self):
        self.pheromone = copy.deepcopy(self.walls)
        return self.pheromone

    # Reset the maze for a new shortest path problem.
    def reset(self):
        self.initialize_pheromones()

    # Update the pheromones along a certain route according to a certain Q
    # @param r The route of the ants
    # @param Q Normalization factor for amount of dropped pheromone
    def add_pheromone_route(self, route, q):
        # self.pheromone[route[0].get_y()][route[0].get_x()] += q/len(route)
        for i in route:
            # x = self.pheromone[i.get_y()][i.get_x()]
            # print("pheromone before: " + str(x))
            self.pheromone[i.get_y()][i.get_x()] += q/len(route)
            # x += q/len(route)
            # print("pheromone after: " + str(x))

        return

     # Update pheromones for a list of routes
     # @param routes A list of routes
     # @param Q Normalization factor for amount of dropped pheromone
    def add_pheromone_routes(self, routes, q):
        for r in routes:
            self.add_pheromone_route(r, q)

    # Evaporate pheromone
    # @param rho evaporation factor
    def evaporate(self, rho):
        for i in range(len(self.pheromone)):
            for j in range(len(self.pheromone[i])):
                self.pheromone[i][j] = (1-rho) * self.pheromone[i][j]

        return

    # Width getter
    # @return width of the maze
    def get_width(self):
        return self.width

    # Length getter
    # @return length of the maze
    def get_length(self):
        return self.length

    # Returns a the amount of pheromones on the neighbouring positions (N/S/E/W).
    # @param position The position to check the neighbours of.
    # @return the pheromones of the neighbouring positions.
    def get_surrounding_pheromone(self, position):
        north = self.get_pheromone(position.add_direction(self.north))
        south = self.get_pheromone(position.add_direction(self.south))

        east = self.get_pheromone(position.add_direction(self.east))
        west = self.get_pheromone(position.add_direction(self.west))

        return SurroundingPheromone(north, east, south, west)

    def get_surrounding_pheromone_array(self, position):
        north = self.get_pheromone(position.add_direction(self.north))
        south = self.get_pheromone(position.add_direction(self.south))

        east = self.get_pheromone(position.add_direction(self.east))
        west = self.get_pheromone(position.add_direction(self.west))

        return [north, south, east, west]
    # Pheromone getter for a specific position. If the position is not in bounds returns 0
    # @param pos Position coordinate
    # @return pheromone at point
    def get_pheromone(self, pos):
        x = pos.get_x()
        y = pos.get_y()
        if self.in_bounds(pos):
            return self.pheromone[y][x]
        else:
            return 0

    # Check whether a coordinate lies in the current maze.
    # @param position The position to be checked
    # @return Whether the position is in the current maze
    def in_bounds(self, position):
        return position.x_between(0, self.width) and position.y_between(0, self.length)

    def is_accessible(self, position):
        return self.walls[position.get_y()][position.get_x()] is 1
    # Representation of Maze as defined by the input file format.
    # @return String representation
    def __str__(self):
        string = ""
        string += str(self.width)
        string += " "
        string += str(self.length)
        string += " \n"
        for y in range(self.length):
            for x in range(self.width):
                string += str(self.walls[x][y])
                string += " "
            string += "\n"
        return string

    # Method that builds a mze from a file
    # @param filePath Path to the file
    # @return A maze object withf pheromones initialized to 0's inaccessible and 1's accessible.
    @staticmethod
    def create_maze(file_path):
        try:
            f = open(file_path, "r")
            lines = f.read().splitlines()
            dimensions = lines[0].split(" ")
            width = int(dimensions[0])
            length = int(dimensions[1])
            
            #make the maze_layout
            maze_layout = []
            for x in range(width):
                maze_layout.append([])
            
            for y in range(length):
                line = lines[y+1].split(" ")
                for x in range(width):
                    if line[x] != "":
                        state = int(line[x])
                        maze_layout[x].append(state)
            print("Ready reading maze file " + file_path)
            return Maze(maze_layout, width, length, pheromone=maze_layout)
        except FileNotFoundError:
            print("Error reading maze file " + file_path)
            traceback.print_exc()
            sys.exit()