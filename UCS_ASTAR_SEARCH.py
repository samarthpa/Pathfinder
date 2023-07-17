import heapq
import math
import os
import pickle
from math import sqrt


class PriorityQueue(object):

    def __init__(self):
        """Initialize a new Priority Queue."""
        self.index = 0
        self.queue = []

    def pop(self):
        node = heapq.heappop(self.queue)
        return (node[0], node[2])

    def remove(self, node):


        self.queue.remove(node)
        heapq.heapify(self.queue)
        return node


    def __iter__(self):
 
        return iter(sorted(self.queue))

    def __str__(self):


        return 'PQ:%s' % self.queue

    def append(self, node):
        heapq.heappush(self.queue, (node[0], self.index, node[1]))
        self.index += 1
        return self.queue
        
    def __contains__(self, key):


        return key in [n[-1] for n in self.queue]

    def __eq__(self, other):


        return self.queue == other.queue

    def size(self):


        return len(self.queue)

    def clear(self):

        self.queue = []

    def top(self):

        return self.queue[0]


def breadth_first_search(graph, start, goal):
    """
    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    visited = set()
    queue = PriorityQueue()
    visited.add(start)
    queue.append((0,[start]))
    list = []

    if goal == start:
        return list

    while queue:
        popped = queue.pop()
        visited.add(popped[1][-1])
        for neighbor in sorted(graph.neighbors((popped[1][-1]))):
            if neighbor == goal:
                list = popped[1].copy()
                list.append(neighbor)
                return list
            if neighbor not in visited:
                visited.add(neighbor)
                list = popped[1].copy()
                list.append(neighbor)
                queue.append((0,list))

def uniform_cost_search(graph, start, goal):
    """
    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    currentCost = {}
    queue = PriorityQueue()
    queue.append((0,[start]))
    list = []

    if goal == start:
        return list

    while queue:
        popped = queue.pop()
        if popped[1][-1] == goal:
            return popped[1]
        if popped[1][-1] not in currentCost.keys():
            for neighbor in sorted(graph.neighbors((popped[1][-1]))):
                path = popped[1] + [neighbor]
                cost = popped[0] + graph.get_edge_weight(popped[1][-1], neighbor)
                queue.append((cost, path))
        currentCost[popped[1][-1]] = cost
            
    # TODO: finish this function!



def null_heuristic(graph, v, goal):
    """
    Null heuristic used as a base line.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        0
    """

    return 0


def euclidean_dist_heuristic(graph, v, goal):
    """
    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Euclidean distance between `v` node and `goal` node
    """
    return sqrt((graph.nodes[v]['pos'][0] - graph.nodes[goal]['pos'][0]) ** 2 + (graph.nodes[v]['pos'][1] - graph.nodes[goal]['pos'][1]) ** 2) 
    # TODO: finish this function!


def a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """
    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    currentCost = {start: 0}
    queue = PriorityQueue()
    queue.append((0,[start]))

    if goal == start:
        return []

    while queue:
        popped = queue.pop()
        if popped[1][-1] == goal:
            return popped[1]
        for neighbor in sorted(graph.neighbors((popped[1][-1]))):
            edge = graph.get_edge_weight(popped[1][-1], neighbor)
            cost = currentCost[popped[1][-1]] + edge 
            if neighbor not in currentCost.keys() or cost < currentCost[neighbor]:
                currentCost[neighbor] = cost
                path = popped[1] + [neighbor]
                queue.append((cost + heuristic(graph, neighbor, goal), path))
    return None

def bidirectional_ucs(graph, start, goal):
    """
    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    startVisited = dict()
    goalVisited  = dict()
    startQueue = PriorityQueue()
    startQueue.append((0, [start]))
    startVisited[start] = (0, [start])
    goalQueue = PriorityQueue()
    goalQueue.append((0, [goal]))
    goalVisited[goal] = (0, [goal])
    mu = float('inf')
    solution = None
    currNum = 0

    if start == goal: 
        return []

    while startQueue and goalQueue:
        if startQueue.top()[0] + goalQueue.top()[0] >= mu:
            break
        
        currQueue = startQueue if currNum == 0 else goalQueue
        currVisited = startVisited if currNum == 0 else goalVisited
        oppVisited = goalVisited if currNum == 0 else startVisited
        curr = currQueue.pop()


        for neighbor in sorted(graph.neighbors(curr[1][-1])):
            pathCost = curr[0] + graph.get_edge_weight(curr[1][-1], neighbor)
            if neighbor not in currVisited or pathCost < currVisited[neighbor][0]:
                currPath = curr[1] + [neighbor]
                currQueue.append((pathCost, currPath))
                currVisited[neighbor] = (pathCost, currPath)
                if neighbor in oppVisited and pathCost + oppVisited[neighbor][0] < mu: 
                    mu = pathCost + oppVisited[neighbor][0]
                    solution = (currPath[:-1] + oppVisited[neighbor][1][::-1]) if currNum == 0 else (oppVisited[neighbor][1][:-1] + currPath[::-1]) 
        currNum ^= 1

    return solution


def bidirectional_a_star(graph, start, goal,
                         heuristic=euclidean_dist_heuristic):
    """
    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).2
    """
    startVisited = {start: (0, [start])}
    goalVisited  = {goal: (0, [goal])}
    startQueue = PriorityQueue()
    startQueue.append((heuristic(graph, start, goal), [start]))
    goalQueue = PriorityQueue()
    goalQueue.append((heuristic(graph, goal, start), [goal]))
    mu = float('inf')
    solution = None
    currNum = 0
    startExplored = set()
    goalExplored = set()

    if start == goal: 
        return []

    while startQueue.size() > 0 and goalQueue.size() > 0:
        if startQueue.top()[0] + goalQueue.top()[0] >= mu + heuristic(graph, start, goal):
            break
        
        currQueue = startQueue if currNum == 0 else goalQueue
        oppQueue = goalQueue if currNum == 0 else startQueue
        currVisited = startVisited if currNum == 0 else goalVisited
        oppVisited = goalVisited if currNum == 0 else startVisited
        currExplored = startExplored if currNum == 0 else goalExplored
        curr = currQueue.pop()
        if curr[1][-1] in currExplored:
            continue
        for neighbor in sorted(graph.neighbors(curr[1][-1])):
            pathCost = currVisited[curr[1][-1]][0] + graph.get_edge_weight(curr[1][-1], neighbor)
            if neighbor not in currVisited or pathCost < currVisited[neighbor][0]:
                priority = pathCost + (0.5 * (heuristic(graph, neighbor, goal) - heuristic(graph, start, neighbor)) if currNum % 2 == 0 else 0.5 * (heuristic(graph, start, neighbor) - heuristic(graph, neighbor, goal))) + 0.5 * heuristic(graph, start, goal)
                currPath = curr[1] + [neighbor]
                currQueue.append((priority, currPath))
                currVisited[neighbor] = (pathCost, currPath)
                if neighbor in oppVisited and pathCost + oppVisited[neighbor][0] < mu: 
                    mu = pathCost + oppVisited[neighbor][0]
                    solution = (currPath[:-1] + oppVisited[neighbor][1][::-1]) if currNum == 0 else (oppVisited[neighbor][1][:-1] + currPath[::-1]) 
        currNum ^= 1
        currExplored.add(curr[1][-1])
    return solution