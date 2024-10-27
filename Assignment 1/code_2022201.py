import numpy as np
import pickle
from queue import Queue
from queue import PriorityQueue 
import time
import tracemalloc
import copy
import matplotlib.pyplot as plt

# General Notes:
# - Update the provided file name (code_<RollNumber>.py) as per the instructions.
# - Do not change the function name, number of parameters or the sequence of parameters.
# - The expected output for each function is a path (list of node names)
# - Ensure that the returned path includes both the start node and the goal node, in the correct order.
# - If no valid path exists between the start and goal nodes, the function should return None.


# Algorithm: Iterative Deepening Search (IDS)

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 2, 9, 8, 5, 97, 98, 12]

def construct_path(goal_node, parent): 
    path = []

    # Path from start_node/root to goal_node
    node = goal_node
    while (node != "root"):
        path.append(node)
        node = parent[node]

    # Reverse the path for start -> goal
    path.reverse()
    return path 

def depth_limited_search(adj_matrix, start_node, goal_node, limit):
    n = len(adj_matrix)

    stack = [start_node] # Frontier # Stack(LIFO queue) to implement DFS
    DEPTH = {start_node : 0} # Depth of each node
    PATH_COST = {start_node : 0} # The total cost of the path from the initial state to this node.
    PARENT = {start_node : "root"} # Parent of each node
    reached = {start_node} # Set of visited nodes (not necessarily expanded)

    result = "failure"
    while (len(stack) != 0):
        node = stack.pop()
        # If we find the goal node, return the path by reconstructing it
        if (node == goal_node):
            return construct_path(goal_node, PARENT),PATH_COST[goal_node]
        # EXPAND node
        for child in range(n - 1, -1, -1):
            # Skip if there is no edge from the current node to the child
            if((adj_matrix[node][child] == 0)): 
                continue
            proposed_child_depth = DEPTH[node] + 1
            # If the child node has not been visited or can be reached with a shorter depth (This also prevent Cycle)
            if((child not in reached) or (proposed_child_depth < DEPTH[child])): 
                reached.add(child)
                PARENT[child] = node
                PATH_COST[child] = PATH_COST[node] + adj_matrix[node][child]    
                DEPTH[child] = proposed_child_depth 
                if(DEPTH[child]>limit):
                    result = "cutoff"
                else:
                    stack.append(child)
    return result  # Return "failure" if the goal node is not found within the depth limit

def iterative_deepning_search(adj_matrix,node_attributes, start_node, goal_node):
    n = len(adj_matrix)
    # maximum possible depth can be number of nodes in the graph
    for limit in range(n): 
        result = depth_limited_search(adj_matrix, start_node, goal_node, limit)
        # If depth-limited search hits "cutoff", try the next depth limit
        if (result == "cutoff"):
            continue
        # If search fails to find the goal, return None
        elif (result == "failure"):
            return None
        # If a valid path is found, return the path and cost
        else:
            return result

def get_ids_path(adj_matrix, start_node, goal_node):
    result =  iterative_deepning_search(adj_matrix, None, start_node, goal_node)
    if result is None:
        return None
    return result[0]



# Algorithm: Bi-Directional Search

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 2, 9, 8, 5, 97, 98, 12]


def construct_bidirectional_path(meeting_node, parentf, parentb):
    # Construct path from start_node to meeting_node and from meeting_node to goal_node
    path = []
    
    # Path from start_node to meeting_node
    node = meeting_node
    while (node != "root"):
        path.append(node)
        node = parentf[node]
    path.reverse()

    # Path from meeting_node to goal_node
    node = parentb[meeting_node]
    while (node != "root"):
        path.append(node)
        node = parentb[node]

    return path

def bidirectional_bfs(adj_matrix, node_attributes, start_node, goal_node):
    n = len(adj_matrix)
    
    # Initialize forward BFS queue with start_node
    queuef = Queue()
    queuef.put(start_node)  
    # Initialize backward BFS queue with goal_node
    queueb = Queue()
    queueb.put(goal_node)  
    
    PATH_COSTf = {start_node : 0} # Cost to reach each node from start_node in forward BFS
    PATH_COSTb = {goal_node : 0} # Cost to reach each node from goal_node in backward BFS
    PARENTf = {start_node : "root"} # Parent of each node in forward BFS
    PARENTb = {goal_node : "root"} # Parent of each node in backward BFS
    reachedf = {start_node} # Set of visited nodes in forward BFS
    reachedb = {goal_node} # Set of visited nodes in backward BFS

    while ((not queuef.empty()) and (not queueb.empty())):
        # Explores from both the start and goal nodes simultaneously, layer by layer

        # Forward BFS
        queuef_size = queuef.qsize()
        for _ in range(queuef_size): # Explore all nodes at the same level for front search
            nodef = queuef.get() 
            # Meeting point at nodef
            if (nodef in reachedb): 
                total_cost = PATH_COSTf[nodef] + PATH_COSTb[nodef]
                return construct_bidirectional_path(nodef, PARENTf, PARENTb),total_cost
            # Expand the children of the nodef
            for childf in range(n):
                if (adj_matrix[nodef][childf] == 0):  # Skip if there is no forward edge from the current node to the child
                    continue
                if (childf not in reachedf):
                    reachedf.add(childf)
                    PARENTf[childf] = nodef
                    PATH_COSTf[childf] = PATH_COSTf[nodef] + adj_matrix[nodef][childf]  
                    queuef.put(childf)  


        # Backward BFS
        queueb_size = queueb.qsize()
        for _ in range(queueb_size): # Expand all nodes at the same level for back search
            nodeb = queueb.get()  
            # Meeting point at nodeb
            if (nodeb in reachedf):
                total_cost = PATH_COSTf[nodeb] + PATH_COSTb[nodeb]
                return construct_bidirectional_path(nodeb, PARENTf, PARENTb),total_cost
            # Expand the children of the nodeb
            for childb in range(n):
                if (adj_matrix[childb][nodeb] == 0): # Skip if there is no backward edge from the child to the current node
                    continue
                if (childb not in reachedb):
                    reachedb.add(childb)
                    PARENTb[childb] = nodeb
                    PATH_COSTb[childb] = PATH_COSTb[nodeb] + adj_matrix[childb][nodeb]  # Update path cost
                    queueb.put(childb) 


    return None # No path / failure

def get_bidirectional_search_path(adj_matrix, start_node, goal_node):
    result = bidirectional_bfs(adj_matrix, None, start_node, goal_node)
    if result is None:
        return None
    return result[0]


# Algorithm: A* Search Algorithm

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - node_attributes: Dictionary of node attributes containing x, y coordinates for heuristic calculations.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 27, 9, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 28, 10, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 27, 9, 8, 5, 97, 28, 10, 12]


def dist(node_u,node_v,node_attributes): # Euclidean distance between two nodes

    node_u_x,node_u_y = node_attributes[node_u]['x'],node_attributes[node_u]['y']
    node_v_x,node_v_y = node_attributes[node_v]['x'],node_attributes[node_v]['y']

    return ((node_u_x - node_v_x)**2 + (node_u_y - node_v_y)**2)**0.5

def heuristic(node_w,node_u,node_v,node_attributes): # heuristic function, h(n) = dist(u,w) + dist(w,v)
    dist_u_w = dist(node_u,node_w,node_attributes)
    dist_w_v = dist(node_w,node_v,node_attributes)
    return dist_u_w + dist_w_v

def evaluation(node,start_node,goal_node,PATH_COST,node_attributes): # evaluation function, f(n) = g(n) + h(n)
    g_node = PATH_COST[node]
    h_node = heuristic(node,start_node,goal_node,node_attributes)
    return g_node + h_node

def astar_search(adj_matrix, node_attributes, start_node, goal_node):
    n = len(adj_matrix)

    pqueue = PriorityQueue()  # Priority queue to implement A* search # (f(node), node)
    PATH_COST = {start_node: 0}  # Cost to reach each node from start_node, g(n)
    PARENT = {start_node: "root"}  # Parent of each node
    reached = {start_node}  # Set of visited nodes (not necessarily expanded)

    # Calculate f(start_node) and add it to the priority queue
    f_start_node = evaluation(start_node, start_node, goal_node, PATH_COST, node_attributes)
    pqueue.put((f_start_node, start_node))  

    while(not pqueue.empty()):
        node = pqueue.get()[1]
        if(node == goal_node): # If we find the goal node, return the path by reconstructing it
            return construct_path(goal_node,PARENT),PATH_COST[goal_node]
        for child in range(n):
            if (adj_matrix[node][child] == 0): # Skip if there is no edge from the current node to the child
                continue
            proposed_child_path_cost = PATH_COST[node] + adj_matrix[node][child]
            # If the child node has not been visited or can be reached with a shorter path/cost
            if((child not in reached) or (proposed_child_path_cost < PATH_COST[child])): 
                reached.add(child)
                PARENT[child] = node
                PATH_COST[child] = proposed_child_path_cost
                f_child = evaluation(child,start_node,goal_node,PATH_COST,node_attributes)
                pqueue.put((f_child,child))

    return None # No path / failure

def get_astar_search_path(adj_matrix, node_attributes, start_node, goal_node):
    result = astar_search(adj_matrix,node_attributes,start_node,goal_node)
    if result is None:
        return None
    return result[0]



# Algorithm: Bi-Directional Heuristic Search

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - node_attributes: Dictionary of node attributes containing x, y coordinates for heuristic calculations.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 27, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 34, 33, 11, 32, 31, 3, 5, 97, 28, 10, 12]


def bidirectional_astar_search(adj_matrix, node_attributes, start_node, goal_node):
    n = len(adj_matrix)

    pqueuef = PriorityQueue()  # Forward A* priority queue # (f(node), node)
    pqueueb = PriorityQueue()  # Backward A* priority queue # (f(node), node)
    PATH_COSTf = {start_node: 0}  # Forward path costs
    PATH_COSTb = {goal_node: 0}  # Backward path costs
    PARENTf = {start_node: "root"}  # Forward parents
    PARENTb = {goal_node: "root"}  # Backward parents
    reachedf = {start_node}  # Forward visited nodes set
    reachedb = {goal_node}  # Backward visited nodes set

    # Calculate f(start_node) and add it to the forward priority queue
    f_start_node_forward = evaluation(start_node, start_node, goal_node, PATH_COSTf, node_attributes)
    pqueuef.put((f_start_node_forward, start_node))

    # Calculate f(goal_node) and add it to the backward priority queue
    f_goal_node_backward = evaluation(goal_node, goal_node, start_node, PATH_COSTb, node_attributes)
    pqueueb.put((f_goal_node_backward, goal_node))

    while ((not pqueuef.empty()) and (not pqueueb.empty())):
        
        # Forward A* search
        evalf,nodef = pqueuef.get()
        # Meeting point at nodef
        if (nodef in reachedb): 
            total_cost = PATH_COSTf[nodef] + PATH_COSTb[nodef]
            return construct_bidirectional_path(nodef, PARENTf, PARENTb),total_cost
        # EXPAND nodef
        for childf in range(n):
            if (adj_matrix[nodef][childf] == 0):
                continue
            proposed_childf_path_cost = PATH_COSTf[nodef] + adj_matrix[nodef][childf]
            if ((childf not in reachedf) or (proposed_childf_path_cost < PATH_COSTf[childf])):
                reachedf.add(childf)
                PARENTf[childf] = nodef
                PATH_COSTf[childf] = proposed_childf_path_cost
                f_childf = evaluation(childf, start_node, goal_node, PATH_COSTf, node_attributes) # forward evaluation
                pqueuef.put((f_childf, childf))

        # Backward A* search
        evalb,nodeb = pqueueb.get()
        # Meeting point at nodeb
        if (nodeb in reachedf):
            total_cost = PATH_COSTf[nodeb] + PATH_COSTb[nodeb]
            return construct_bidirectional_path(nodeb, PARENTf, PARENTb),total_cost
        # EXPAND nodeb
        for childb in range(n):
            if (adj_matrix[childb][nodeb] == 0):
                continue
            proposed_childb_path_cost = PATH_COSTb[nodeb] + adj_matrix[childb][nodeb]
            if ((childb not in reachedb) or (proposed_childb_path_cost < PATH_COSTb[childb])):
                reachedb.add(childb)
                PATH_COSTb[childb] = proposed_childb_path_cost
                PARENTb[childb] = nodeb
                f_childb = evaluation(childb, goal_node, start_node, PATH_COSTb, node_attributes) # backward evaluation
                pqueueb.put((f_childb, childb))

    return None  # No path / failure

def get_bidirectional_heuristic_search_path(adj_matrix, node_attributes, start_node, goal_node):
    result = bidirectional_astar_search(adj_matrix, node_attributes, start_node, goal_node)
    if result is None:
        return None
    return result[0]


# Bonus Problem

# Input:
# - adj_matrix: A 2D list or numpy array representing the adjacency matrix of the graph.

# Return:
# - A list of tuples where each tuple (u, v) represents an edge between nodes u and v.
#   These are the vulnerable roads whose removal would disconnect parts of the graph.

# Note:
# - The graph is undirected, so if an edge (u, v) is vulnerable, then (v, u) should not be repeated in the output list.
# - If the input graph has no vulnerable roads, return an empty list [].

def is_bridge_dfs(adj_matrix, u, v): # Check if the nodes u and v are still connected after removing the edge (u, v) using DFS
    n = len(adj_matrix)
    
    visited = [False] * n
    stack = [u]
    visited[u] = True

    while (len(stack)!=0):
        node = stack.pop()
        for child in range(n):
            if ((adj_matrix[node][child] == 1) and (not visited[child])):
                visited[child] = True
                stack.append(child)
                if(child == v): # v is reachable from u, so the edge (u, v) is not a bridge
                    return False
    return True # v is not reachable from u, so the edge (u, v) is a bridge

def bonus_problem(adj_matrix):
    n = len(adj_matrix)
    bridges = []

    adj_matrix = copy.deepcopy(adj_matrix)
    # Make adj_matrix undirected and unweighted
    for u in range(n):
        for v in range(u,n):
            if((adj_matrix[u][v] != 0) or (adj_matrix[v][u] != 0)):
                adj_matrix[u][v] = 1
                adj_matrix[v][u] = 1
    
    for u in range(n):
        for v in range(u,n):
            if(adj_matrix[u][v] != 0):
                # Remove the edge (u, v) 
                adj_matrix[u][v] = 0
                # Check if the removed edge (u, v) was a bridge 
                if(is_bridge_dfs(adj_matrix,u,v)): 
                    bridges.append((u,v))
                # Restore the edge (u, v)
                adj_matrix[u][v] = 1

    return bridges

# Utility function to measure time and space complexity of the search algorithms
def get_time_and_space(func, *args, **kwargs):
    start_time = time.time() # Measure start time
    tracemalloc.start() # Start tracking memory usage

    result = func(*args, **kwargs)

    _, peak = tracemalloc.get_traced_memory() # Get peak memory usage
    tracemalloc.stop() # Stop tracking memory usage
    end_time = time.time() # Measure end time
    
    # Calculate time and memory used
    execution_time = end_time - start_time
    peak_memory_used = peak / 1024  # Convert to KB
    
    return execution_time, peak_memory_used, result


def compare_performance(adj_matrix, node_attributes, *search_algorithms):
    n = len(adj_matrix)
    n = 25

    avg_time_list = []
    avg_space_list = []
    avg_cost_list = []
    algo_names = []
    
    for algo in search_algorithms:

        time_array = np.zeros((n, n))
        space_array = np.zeros((n, n))
        path_list = [[] for _ in range(n)]  
        cost_array = np.zeros((n, n))

        for u in range(n):
            for v in range(n):
                # Get time, space, and path for the given algorithm
                time, space, result = get_time_and_space(algo, adj_matrix, node_attributes, u, v)
                path,cost = None,0 # Using cost as 0 if path is None
                if result is not None:
                    path = result[0]
                    cost = result[1]
                time_array[u][v] = time
                space_array[u][v] = space
                path_list[u].append(path)
                cost_array[u][v] = cost

        # Collect the average time, space, and cost for the current algorithm
        avg_time = np.mean(time_array, axis=1)
        avg_space = np.mean(space_array, axis=1)
        avg_cost = np.mean(cost_array, axis=1)
        
        algo_names.append(algo.__name__)
        avg_time_list.append(avg_time)
        avg_space_list.append(avg_space)
        avg_cost_list.append(avg_cost)

        # Print total time, space, and cost for the current algorithm
        print(f'-----{algo.__name__}-----')
        print(f'Total Time: {np.sum(time_array):.4f}')
        print(f'Total Space: {np.sum(space_array):.4f}')
        print(f'Total Cost: {np.sum(cost_array):.4f}')
        # print(f'Paths: {path_list}')

    # Convert lists to numpy arrays for easier plotting
    avg_time_list = np.array(avg_time_list)
    avg_space_list = np.array(avg_space_list)
    avg_cost_list = np.array(avg_cost_list)

    # Plot Average Time
    plt.figure(figsize=(12, 6))
    for i, algo_name in enumerate(algo_names):
        plt.scatter(range(n), avg_time_list[i], label=algo_name)
    plt.xlabel('Nodes')
    plt.ylabel('Average Time')
    plt.title('Average Time Comparison for Different Algorithms')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Average Space
    plt.figure(figsize=(12, 6))
    for i, algo_name in enumerate(algo_names):
        plt.scatter(range(n), avg_space_list[i], label=algo_name)
    plt.xlabel('Nodes')
    plt.ylabel('Average Space')
    plt.title('Average Space Comparison for Different Algorithms')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Average Cost
    plt.figure(figsize=(12, 6))
    for i, algo_name in enumerate(algo_names):
        plt.scatter(range(n), avg_cost_list[i], label=algo_name)
    plt.xlabel('Nodes')
    plt.ylabel('Average Cost')
    plt.title('Average Cost Comparison for Different Algorithms')
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    adj_matrix = np.load('IIIT_Delhi.npy')
    with open('IIIT_Delhi.pkl', 'rb') as f:
        node_attributes = pickle.load(f)

    start_node = int(input("Enter the start node: "))
    end_node = int(input("Enter the end node: "))

    print(f'Iterative Deepening Search Path: {get_ids_path(adj_matrix,start_node,end_node)}')
    print(f'Bidirectional Search Path: {get_bidirectional_search_path(adj_matrix,start_node,end_node)}')
    print(f'A* Path: {get_astar_search_path(adj_matrix,node_attributes,start_node,end_node)}')
    print(f'Bidirectional Heuristic Search Path: {get_bidirectional_heuristic_search_path(adj_matrix,node_attributes,start_node,end_node)}')
    print(f'Bonus Problem: {bonus_problem(adj_matrix)}')

    # compare_performance(adj_matrix,node_attributes,iterative_deepning_search,bidirectional_bfs)
    # compare_performance(adj_matrix,node_attributes,astar_search,bidirectional_astar_search)
    # compare_performance(adj_matrix,node_attributes,iterative_deepning_search,bidirectional_bfs,astar_search,bidirectional_astar_search)
