import math
import os, sys

from src.Coordinate import Coordinate
from src.Route import Route

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


import random
import enum

# Enum representing the directions an ant can take.
class Direction(enum.Enum):
    east = 0
    north = 1
    west = 2
    south = 3

    # Direction to an int.
    # @param dir the direction.
    # @return an integer from 0-3.
    @classmethod
    def dir_to_int(cls, dir):
        return dir.value
#Class that represents the ants functionality.
class Ant:

    # Constructor for ant taking a Maze and PathSpecification.
    # @param maze Maze the ant will be running in.
    # @param spec The path specification consisting of a start coordinate and an end coordinate.
    def __init__(self, maze, path_specification):
        self.maze = maze
        self.start = path_specification.get_start()
        self.end = path_specification.get_end()
        self.current_position = self.start
        self.rand = random
        self.east = 0
        self.north = 1
        self.west = 2
        self.south = 3

    # Method that performs a single run through the maze by the ant.
    # @return The route the ant found through the maze.
    def find_route(self):

        route = Route(self.start)
        history = []
        history.append(self.start)
        directions = []
        while (not self.current_position.__eq__(self.end)):
            # print("!!!!!")
            # print(self.current_position.get_x())
            # print(self.current_position.get_y())
            # print()
            # print("current pos:x: " + str(self.current_position.get_x()) + " y:  " + str(self.current_position.get_y()))
            possibilities = []
            north = self.current_position.add_direction(self.north)
            south = self.current_position.add_direction(self.south)
            east = self.current_position.add_direction(self.east)
            west = self.current_position.add_direction(self.west)
            # print("EAST: " + str(east.get_x()) + " " + str(east.get_y()))
            # print("NORTH: " + str(north.get_x()) + " " + str(north.get_y()))
            # print("WEST: " + str(west.get_x()) + " " + str(west.get_y()))
            # print("SOUTH: " + str(south.get_x()) + " " + str(south.get_y()))

            if self.maze.in_bounds(north) and self.maze.is_accessible(north) and not (north in history):
                possibilities.append(north)
            else:
                possibilities.append(None)

            if self.maze.in_bounds(south) and self.maze.is_accessible(south) and not (south in history):
                possibilities.append(south)
            else:
                possibilities.append(None)

            if self.maze.in_bounds(east) and self.maze.is_accessible(east) and not (east in history):
                possibilities.append(east)
            else:
                possibilities.append(None)

            if self.maze.in_bounds(west) and self.maze.is_accessible(west) and not (west in history):
                possibilities.append(west)
            else:
                possibilities.append(None)

            #If the route leads to a dead end which is not the end goal of the maze, then return None.
            surrounding_pheromone = self.maze.get_surrounding_pheromone_array(self.current_position)
            # print("surrounding Pheromone: " + str(surrounding_pheromone))
            count = 0
            # print("HEREEE???")
            # for i in possibilities:
            #     if i is not None:
            #         print("X: " + str(i.get_x()) + " Y: " + str(i.get_y()))
            #     else:
            #         print("NONE")
            for i in range(len(possibilities)):
                if possibilities[i] is None:
                    count += 1
                    surrounding_pheromone[i] = 0
            if count == 4:
                # print("returned")
                # self.maze.dead_ants[self.current_position.get_y()][self.current_position.get_x()] += 1
                # print("fucked up here")
                if route.size() > 0:

                    if route.size() == 1:
                        route.remove_last()
                        directions.pop()
                        self.current_position = self.start

                    else:


                        # print("cont?")
                        route.remove_last()
                        directions.pop()
                        # print("ROUTE SIZE: " + str(route.size()))
                        # print("DIRECTIONS LENGTH " + str(len(directions)))
                        self.current_position = directions[route.size()-1]
                        # print("CURRENT POS: " + str(self.current_position))
                        # print("HISTORY LEN: " + str(len(history)))
                        # print("ROUTE LEN: " + str(route.size()))
                        # print("ROUTE: " + str(route))
                else:
                    break
                continue
            chosen_coordinate = random.choices(population=possibilities, k=1, weights=surrounding_pheromone)
            history.append(chosen_coordinate[0])
            directions.append(chosen_coordinate[0])
            self.current_position = chosen_coordinate[0]

            if self.current_position.__eq__(north):
                route.add(1)
            elif self.current_position.__eq__(south):
                route.add(3)
            elif self.current_position.__eq__(east):
                route.add(0)
            elif self.current_position.__eq__(west):
                route.add(2)
            # print("ROUTE: " + str(route))

        return route, directions
