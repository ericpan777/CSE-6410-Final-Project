"""
TSP solver Script

This script provides olutions for the Traveling Salesman Problem (TSP) Using three types of methods
1. Brute Force
2. 2-Approximate
3. Local Search

Usage:
py file
python main.py -inst <filename> -alg [DF | Approx | LS] -time <cutoff_in_seconds> [-seed <random_seed>]

exe file:
./main -inst <filename> -alg [DF | Approx | LS] -time <cutoff_in_seconds> [-seed <random_seed>]

Parameters:
- inst: TSP filename
- alg: Algorithm choices (BF for Depth First, Approx for Approximation, LS for Local Search)
- time: Cutoff time in seconds
- seed: Random seed (optional) for the Local Search algorithm

Example:
python main.py -inst Roanoke.tsp -alg LS -time 60 -seed 123
"""

import math
import os
import re
import sys
import time 
import random
import itertools
from itertools import permutations
import argparse

class TSPGraph:
    def __init__(self):
        self.points = {}
        self.graph = {}
        self.num_nodes = 0

    def parse_tsp_file(self, file_path):
        with open(os.path.join('DATA', file_path), 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 3 and parts[0].isdigit():
                    index, x, y = parts
                    self.points[str(index)] = (float(x), float(y))
            self.num_nodes = int(index)

    def euclidean_distance(self, point1, point2):
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def calculate_graph(self):
        for i in self.points:
            for j in self.points:
                if (i != j):
                    self.graph[f'{i}-{j}'] = int(round(self.euclidean_distance(self.points[i], self.points[j]), 0))

    def display_distances_from_point(self, point):
        return self.graph[point]
    

class BF:
    def __init__(self, tsp_graph):
        self.tsp_graph = tsp_graph
        self.start_time = 0
        self.time_limit = 0

    def calculate_distance(self, n1, n2):
        x1, y1 = self.tsp_graph.points[n1]
        x2, y2 = self.tsp_graph.points[n2]
        return round(((x1 - x2)**2 + (y1 - y2)**2)**0.5)

    def total_distance(self, route):
        total_dist = 0
        for i in range(len(route)):
            total_dist += self.calculate_distance(route[i], route[(i + 1) % len(route)])
        return total_dist

    def brute_force_tsp(self):
        nodes = list(self.tsp_graph.points.keys())
        shortest_route = None
        min_distance = float('inf')

        for route in itertools.permutations(nodes):
            if time.time() - self.start_time > self.time_limit:
                print(f"Time limit reached for {self.time_limit} seconds")
                break

            current_distance = self.total_distance(route)
            if current_distance < min_distance:
                min_distance = current_distance
                shortest_route = route

        return shortest_route, min_distance

    def solve(self, time_limit):
        self.start_time = time.time()
        self.time_limit = time_limit
        best_route, best_distance = self.brute_force_tsp()
        return best_route, best_distance

class Approx:
    """
    Approx class solves the TSP using a 2-approximation algorithm:
        1. Constructs a MST using Prim's algorithm.
        2. Performs a preorder traversal of the MST to form a path.
        3. Calculates the total distance of the path.
    """
    
    #def __init__(self, tsp_graph, cut_off_time):
    def __init__(self, tsp_graph):
        """
        Initializes the Approx class with a given graph representing the TSP.

        Parameters:
            - tsp_graph (Graph): The graph representing the TSP.
        """

        self.tsp_graph = tsp_graph
        # self.cut_off_time = cut_off_time
        # self.start_time = time.time()

    def prim_mst(self):
        """
        Constructs a MST using Prim's algorithm.

        Returns:
            - list: A list of edges forming the MST.
        """
        
        num_nodes = self.tsp_graph.num_nodes
        selected = [False] * (num_nodes + 1)
        mst_edges = []
        edge_count = 0
        selected[1] = True

        while edge_count < num_nodes - 1:
            minimum = float('inf')
            x, y = 0, 0

            # Find the smallest edge connecting the MST to a new node
            for i in range(1, num_nodes + 1):
                if selected[i]:
                    for j in range(1, num_nodes + 1):
                        if not selected[j] and self.tsp_graph.graph.get(f'{i}-{j}', float('inf')) < minimum:
                            minimum = self.tsp_graph.graph[f'{i}-{j}']
                            x, y = i, j

            if x != 0 and y != 0:
                mst_edges.append((x, y))
                edge_count += 1
                selected[y] = True
                        
        return mst_edges

    def preorder(self, mst, start=1):
        """
        Performs a preorder traversal of the MST to create a path.

        Parameters:
            - mst (list): The list of edges forming the MST.
            - start (int): The starting node.

        Returns:
            - list: The path formed by the preorder traversal.
        """
        
        visited = set()
        path = []

        def visit(node):
            # Recursively visit nodes in the MST.
            if node in visited:
                return
            visited.add(node)
            path.append(node)
            for edge in mst:
                if edge[0] == node and edge[1] not in visited:
                    visit(edge[1])
                elif edge[1] == node and edge[0] not in visited:
                    visit(edge[0])
            
        visit(start)
        return path
    

    # def check_time(self):
    #    return time.time() - self.start_time > self.cut_off_time
        
    def solve(self):
        """
        Solves the TSP using the 2-approximation algorithm.

        Returns:
            - tuple: A tuple containing the tour and its total distance.
        """

        # Start timing
        # self.start_time = time.time()

        # Step 1: Construct the MST using Prim's algorithm
        mst_edges = self.prim_mst()
        # if self.check_time:
        #     return None, "Time limit exceeded"
        

        # Step 2: Perform a preorder traversal of the MST to form a path
        path = self.preorder(mst_edges)
        # if self.check_time:
        #     return None, "Time limit exceeded"

        # Step 3: Calculate the total distance of the path
        distance = 0
        for i in range(len(path)):
            if i == len(path) - 1:
                distance += self.tsp_graph.graph[f'{path[i]}-{path[0]}']
            else:
                distance += self.tsp_graph.graph[f'{path[i]}-{path[i + 1]}']
            # if self.check_time:
            #    return None, "Time limit exceeded"
            
                
        return path, distance


class LocalSearch:
    def __init__(self, tsp_graph, seed=64, steps=10,  initial='ascending', annealing='linear'):
        self.tsp_graph = tsp_graph
        self.seed = seed
        
        # simulated annealing parameters
        self.k = 1000
        self.T = 100
        self.steps = steps

        # initial solution method / annealing method for temperature(T)
        if initial not in ['ascending', 'decending', 'random']:
            initial = 'ascending'
        self.initial = initial
        if annealing not in ['linear', 'exponential']:
            annealing = 'linear'
        self.annealing = annealing
        self.annealing_factor = -15
        
        # optimal path / distance found in given steps
        self.path = []
        self.init_distance = 0
        self.distance = 0

        # run initialization
        self.initialize()

    def initialize(self):
        # set random seed
        random.seed(self.seed)

        # create initial path
        self.path = [i+1 for i in range(self.tsp_graph.num_nodes)]
        if self.initial == 'decending':
            self.path.reverse()
        if self.initial == 'random':
            random.shuffle(self.path)

        # calculate initial distance
        self.calculate_distance()
        self.init_distance = self.distance

    def calculate_distance(self):
        for i in range(len(self.path)):
            if i == len(self.path) - 1:
                self.distance += self.tsp_graph.graph[f'{self.path[i]}-{self.path[0]}']
            else:
                self.distance += self.tsp_graph.graph[f'{self.path[i]}-{self.path[i+1]}']

    def update_distance(self, indices):
        idx1, idx2 = sorted(indices)

        # calculate the indices of previous and next node of node 1 and node 2
        n1_prev_node = idx1 - 1 if idx1 > 0 else self.tsp_graph.num_nodes - 1
        n1_next_node = idx1 + 1 if idx1 < self.tsp_graph.num_nodes - 1 else 0
        n2_prev_node = idx2 - 1 if idx2 > 0 else self.tsp_graph.num_nodes - 1
        n2_next_node = idx2 + 1 if idx2 < self.tsp_graph.num_nodes - 1 else 0

        # old distances of the two nodes (before swap)
        old_n1_prev = self.tsp_graph.graph[f'{self.path[idx1]}-{self.path[n1_prev_node]}']
        old_n1_next = self.tsp_graph.graph[f'{self.path[idx1]}-{self.path[n1_next_node]}']
        old_n2_prev = self.tsp_graph.graph[f'{self.path[idx2]}-{self.path[n2_prev_node]}']
        old_n2_next = self.tsp_graph.graph[f'{self.path[idx2]}-{self.path[n2_next_node]}']
        sum_old = old_n1_prev + old_n1_next + old_n2_prev + old_n2_next

        # new distances of the two nodes (after swap)
        new_n1_prev = self.tsp_graph.graph[f'{self.path[idx1]}-{self.path[idx2]}'] if idx1 == n2_prev_node else self.tsp_graph.graph[f'{self.path[idx1]}-{self.path[n2_prev_node]}']
        new_n1_next = self.tsp_graph.graph[f'{self.path[idx1]}-{self.path[idx2]}'] if abs(idx1 - idx2) == (self.tsp_graph.num_nodes - 1) else self.tsp_graph.graph[f'{self.path[idx1]}-{self.path[n2_next_node]}']
        new_n2_prev = self.tsp_graph.graph[f'{self.path[idx1]}-{self.path[idx2]}'] if abs(idx1 - idx2) == (self.tsp_graph.num_nodes - 1) else self.tsp_graph.graph[f'{self.path[idx2]}-{self.path[n1_prev_node]}']
        new_n2_next = self.tsp_graph.graph[f'{self.path[idx1]}-{self.path[idx2]}'] if idx2 == n1_next_node else self.tsp_graph.graph[f'{self.path[idx2]}-{self.path[n1_next_node]}']
        sum_new = new_n1_prev + new_n1_next + new_n2_prev + new_n2_next

        return self.distance - sum_old + sum_new

    def annealing_function(self, progress):
        # linearly decay
        if self.annealing == 'linear':
            return self.T * (1 - progress)
        # exponentially decay
        if self.annealing == 'exponential':
            if progress == 1.:
                return 0
            else:
                return self.T * math.exp(self.annealing_factor * progress)

    def random_swap_node(self):
        (n1, n2) = random.sample(self.path, 2)
        idx1, idx2 = n1 - 1, n2 - 1
        new_path = self.path.copy()
        new_path[idx1] = self.path[idx2]
        new_path[idx2] = self.path[idx1]

        return new_path, (idx1, idx2)

    def run(self):
        for i in range(self.steps):
            # randomly swap two nodes and calculate the corresponding new distance
            new_path, indices = self.random_swap_node()
            new_distance = self.update_distance(indices)
            
            # update if the distance is shorter
            if new_distance <= self.distance:
                self.distance = new_distance
                self.path = new_path
            # update to longer distance under certain probability
            else:
                delta_e = new_distance - self.distance
                cur_T = self.annealing_function(((i+1) / self.steps))
                p = math.exp(-delta_e / (self.k * cur_T + 1e-10))
                # print("DELTA_E: {0}; P: {1}".format(delta_e, p))
                if random.random() < p:
                    self.distance = new_distance
                    self.path = new_path


def output(filename, cost, path):
    with open(filename, 'w') as file:
        file.write(str(cost) + '\n')
        path_str = ",".join(map(str, path))
        file.write(path_str)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # add command-line argument
    parser.add_argument('-inst', dest = 'filename',help="tsp file name", required=True)
    parser.add_argument('-alg', dest = 'algo', choices=['DF', 'Approx', 'LS'], help='Algorithm choices', required=True)
    parser.add_argument('-time', dest = 'cutoff_time', type = int, help = 'Cuttoff time in second')
    parser.add_argument('-seed', dest = 'random_seed', type = int, help = "Randome seed (optional) for Local search algorithm")

    args = parser.parse_args()

    # Create an instance of TSPGraph
    tsp_graph = TSPGraph()

    # Process each TSP file in the list
    # for file_path in path_list:
    # file_path = sys.argv[1]
    

    tsp_graph.parse_tsp_file(args.filename)
    tsp_graph.calculate_graph()

    
    # Example: Display distances from point 1 to others in the last TSP file processed

    # Algotype = sys.argv[2]
    Algotype = args.algo
    # cutoffTime = sys.argv[3]
    cutoffTime = args.cutoff_time


    if len(sys.argv) > 3:
        random_seed = args.random_seed

    perm = permutations(range(1,4))
    
    for order in perm:
        for i in range(len(order)-1):
            dist = tsp_graph.graph[f'{order[i]}-{order[i+1]}']
            # print(dist)
    
    if Algotype == 'BF':
        bf_solver = BF(tsp_graph)
        try:
            t1 = time.time()
            path, distance = bf_solver.solve(cutoffTime)
            print("Total Time:", time.time() - t1)
            print("Path:", path)
            print("Total Distance:", distance)
            output_filename = "_".join([str(args.filename.replace(".tsp", "")), str(Algotype), str(cutoffTime)]) + ".sol"
            output(output_filename, distance, path)
        except Exception as e:
            print("Error occurred during brute force:", e)
    elif Algotype == 'Approx':
        approx_solver = Approx(tsp_graph)
        try:
            t1 = time.time()
            path, distance = approx_solver.solve()
            print("Total Time:", time.time() - t1)
            print("Path:", path)
            print("Total Distance:", distance)
            output_filename = "_".join([str(args.filename.replace(".tsp", "")), str(Algotype), str(cutoffTime)]) + ".sol"
            output(output_filename, distance, path)
        except Exception as e:
            print("Error occurred during 2-Approximation:", e)
    elif Algotype == 'LS':
        ls = LocalSearch(tsp_graph, random_seed, steps=1000000, initial='random', annealing='exponential')
        t1 = time.time()
        ls.run()
        print("Total Time:", time.time()-t1)
        print("Path:", ls.path)
        print("Initial Distance:", ls.init_distance)
        print("Final Distance:  ", ls.distance)
        output("_".join([str(args.filename.replace(".tsp", "")), str(Algotype), str(cutoffTime), str(random_seed)]) + ".sol", ls.distance, ls.path)
    else:
        print("Please specify correct algorithm.")

    
