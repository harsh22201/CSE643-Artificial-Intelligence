# AI Assignment — Knowledge Representation, Reasoning and Planning
# CSE 643

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import networkx as nx
from pyDatalog import pyDatalog
from collections import defaultdict, deque

## ****IMPORTANT****
## Don't import or use any other libraries other than defined above
## Otherwise your code file will be rejected in the automated testing

# ------------------ Global Variables ------------------
route_to_stops = defaultdict(list)  # Mapping of route IDs to lists of stops
trip_to_route = {}                   # Mapping of trip IDs to route IDs
stop_trip_count = defaultdict(int)    # Count of trips for each stop
fare_rules = {}                      # Mapping of route IDs to fare information
merged_fare_df = None                # To be initialized in create_kb()

# Load static data from GTFS (General Transit Feed Specification) files
df_stops = pd.read_csv('GTFS/stops.txt')
df_routes = pd.read_csv('GTFS/routes.txt')
df_stop_times = pd.read_csv('GTFS/stop_times.txt')
df_fare_attributes = pd.read_csv('GTFS/fare_attributes.txt')
df_trips = pd.read_csv('GTFS/trips.txt')
df_fare_rules = pd.read_csv('GTFS/fare_rules.txt')

# ------------------ Function Definitions ------------------

# Function to create knowledge base from the loaded data
def create_kb():
    """
    Create knowledge base by populating global variables with information from loaded datasets.
    It establishes the relationships between routes, trips, stops, and fare rules.
    
    Returns:
        None
    """
    global route_to_stops, trip_to_route, stop_trip_count, fare_rules, merged_fare_df

    # Create trip_id to route_id mapping
    trip_to_route = dict(zip(df_trips["trip_id"], df_trips["route_id"]))

    # Map route_id to a list of stops in order of their sequence
    for trip_id, stop_id, stop_sequence in zip(df_stop_times["trip_id"], df_stop_times["stop_id"],df_stop_times["stop_sequence"]):
        route_id = trip_to_route[trip_id]
        if(stop_id not in route_to_stops[route_id]):
            route_to_stops[route_id].append(stop_id)

    # Ensure each route only has unique stops
    # for route_id in route_to_stops:
    #     route_to_stops[route_id] = list(set(route_to_stops[route_id]))
    
    # Count trips per stop
    stop_trip_count.update(dict(df_stop_times["stop_id"].value_counts()))

    # Create fare rules for routes

    # Merge fare rules and attributes into a single DataFrame
    merged_fare_df = pd.merge(df_fare_attributes, df_fare_rules, on="fare_id", how='inner')

# Function to find the top 5 busiest routes based on the number of trips
def get_busiest_routes():
    """
    Identify the top 5 busiest routes based on trip counts.

    Returns:
        list: A list of tuples, where each tuple contains:
            - route_id (int): The ID of the route.
            - trip_count (int): The number of trips for that route.
    """

    # Count the number of trips per route_id
    route_trip_count = df_trips['route_id'].value_counts()

    # Get the top 5 busiest routes
    top_busy_routes = route_trip_count.nlargest(5)

    busiest_routes = list(zip(top_busy_routes.index, top_busy_routes.values))
    return busiest_routes

# Function to find the top 5 stops with the most frequent trips
def get_most_frequent_stops():
    """
    Identify the top 5 stops with the highest number of trips.

    Returns:
        list: A list of tuples, where each tuple contains:
            - stop_id (int): The ID of the stop.
            - trip_count (int): The number of trips for that stop.
    """
    # Count the number of trips per stop_id and get the top 5 most frequent stops
    top_freq_stops = df_stop_times['stop_id'].value_counts().nlargest(5)

    most_frequent_stops = list(zip(top_freq_stops.index, top_freq_stops.values))
    return most_frequent_stops

# Function to find the top 5 busiest stops based on the number of routes passing through them
def get_top_5_busiest_stops():
    """
    Identify the top 5 stops with the highest number of different routes.

    Returns:
        list: A list of tuples, where each tuple contains:
            - stop_id (int): The ID of the stop.
            - route_count (int): The number of routes passing through that stop.
    """
    # List to hold all stops from different routes
    all_stops = []  

    # populate all_stops 
    for route_id, stops in route_to_stops.items():
        for stop in stops:
            all_stops.append(stop)  

    # Count occurrences of each stop and find the top stops
    stop_route_count = pd.Series(all_stops).value_counts()

    # Get the top 5 busiest stops
    top_busy_stops = stop_route_count.nlargest(5)

    busiest_stops = list(zip(top_busy_stops.index, top_busy_stops.values))
    return busiest_stops

# Function to identify the top 5 pairs of stops with only one direct route between them
def get_stops_with_one_direct_route():
    """
    Identify the top 5 pairs of consecutive stops (start and end) connected by exactly one direct route. 
    The pairs are sorted by the combined frequency of trips passing through both stops.

    Returns:
        list: A list of tuples, where each tuple contains:
              - pair (tuple): A tuple with two stop IDs (stop_1, stop_2).
              - route_id (int): The ID of the route connecting the two stops.
    """
    stop_pair = defaultdict(list) 

    # Iterate over all routes and their corresponding stops
    for route_id, stops in route_to_stops.items():
        for i in range(len(stops)):
            for j in range(i + 1, len(stops)):
                stop_1 = stops[i]
                stop_2 = stops[j]
                # Sort the stops to ensure (stop_1, stop_2) and (stop_2, stop_1) are treated as the same pair
                stop_pair_key = tuple(sorted((stop_1, stop_2)))
                stop_pair[stop_pair_key].append(route_id)

    # filter pairs that are connected by exactly one route
    one_direct_route_pairs = {pair: routes for pair, routes in stop_pair.items() if len(routes) == 1}

    # Calculate the combined trip count for each pair
    stop_pair_with_frequency = []
    for pair, routes in one_direct_route_pairs.items():
        # Get the route_id (since there's only one route for each pair)
        route_id = routes[0]
        
        # Calculate combined trip frequency using stop_trip_count
        stop_1, stop_2 = pair
        combined_trip_count = stop_trip_count[stop_1] + stop_trip_count[stop_2]
        
        # Append the result as a tuple (pair, route_id, combined_trip_count)
        stop_pair_with_frequency.append((pair, route_id, combined_trip_count))

    # Sort the pairs by combined trip count in descending order
    top_5_pairs = sorted(stop_pair_with_frequency, key=lambda x: x[2], reverse=True)[:5]
    return [(pair, route_id) for pair, route_id, _ in top_5_pairs]  # Return only pairs and route_ids


# Function to get merged fare DataFrame
# No need to change this function
def get_merged_fare_df():
    """
    Retrieve the merged fare DataFrame.

    Returns:
        DataFrame: The merged fare DataFrame containing fare rules and attributes.
    """
    global merged_fare_df
    return merged_fare_df

# Visualize the stop-route graph interactively
def visualize_stop_route_graph_interactive(route_to_stops):
    """
    Visualize the stop-route graph using Plotly for interactive exploration.

    Args:
        route_to_stops (dict): A dictionary mapping route IDs to lists of stops.

    Returns:
        None
    """
    pass  # Implementation here

# Brute-Force Approach for finding direct routes
def direct_route_brute_force(start_stop, end_stop):
    """
    Find all valid routes between two stops using a brute-force method.

    Args:
        start_stop (int): The ID of the starting stop.
        end_stop (int): The ID of the ending stop.

    Returns:
        list: A list of route IDs (int) that connect the two stops directly.
    """
    direct_routes = []

    # Iterate through each route and its stops
    for route_id, route_stops in route_to_stops.items():
        if ((start_stop in route_stops) and (end_stop in route_stops)):
            direct_routes.append(route_id)

    return direct_routes

# Initialize Datalog predicates for reasoning
pyDatalog.create_terms('RouteHasStop, DirectRoute, OptimalRoute, X, Y, Z, R, R1, R2')  
def initialize_datalog():
    """
    Initialize Datalog terms and predicates for reasoning about routes and stops.

    Returns:
        None
    """
    pyDatalog.clear()  # Clear previous terms
    print("Terms initialized: DirectRoute, RouteHasStop, OptimalRoute")  # Confirmation print

    # Define Datalog predicates
    DirectRoute(R, X, Y) <= (RouteHasStop(R, X)) & (RouteHasStop(R, Y)) & (X != Y)
    OptimalRoute(R1, R2, X, Y, Z) <= DirectRoute(R1, X, Z) & DirectRoute(R2, Z, Y) & (R1 != R2)

    create_kb()  # Populate the knowledge base
    add_route_data(route_to_stops)  # Add route data to Datalog
    
# Adding route data to Datalog
def add_route_data(route_to_stops):
    """
    Add the route data to Datalog for reasoning.

    Args:
        route_to_stops (dict): A dictionary mapping route IDs to lists of stops.

    Returns:
        None
    """
    for route_id, route_stops in route_to_stops.items():
        for stop_id in route_stops:
            # Add the route-stop relationship to the Datalog knowledge base
            +RouteHasStop(route_id, stop_id) # route_id has stop_id

# Function to query direct routes between two stops
def query_direct_routes(start, end):
    """
    Query for direct routes between two stops.

    Args:
        start (int): The ID of the starting stop.
        end (int): The ID of the ending stop.

    Returns:
        list: A sorted list of route IDs (str) connecting the two stops.
    """
    # Querying for all routes where there is a DirectRoute between start and end
    result = DirectRoute(R, start, end).ask()

    # Extracting the route IDs from the result
    direct_routes = sorted([route_id for route_id, in result])

    return direct_routes

# Forward chaining for optimal route planning
def forward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Perform forward chaining to find optimal routes considering transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID where a transfer occurs.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id1 (int): The ID of the first route.
              - stop_id (int): The ID of the intermediate stop.
              - route_id2 (int): The ID of the second route.
    """
    # Query for optimal routes using OptimalRoute predicate
    result = OptimalRoute(R1, R2, start_stop_id, end_stop_id, stop_id_to_include).ask()
    
    # Extracting the optimal routes 
    optimal_routes = sorted([(r1, stop_id_to_include, r2) for r1, r2 in result])
    return optimal_routes

# Backward chaining for optimal route planning
def backward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Perform backward chaining to find optimal routes considering transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID where a transfer occurs.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id1 (int): The ID of the first route.
              - stop_id (int): The ID of the intermediate stop.
              - route_id2 (int): The ID of the second route.
    """
    # Query for optimal routes using OptimalRoute predicate
    result = OptimalRoute(R1, R2, start_stop_id, end_stop_id, stop_id_to_include).ask()
    
    # Extracting the optimal routes 
    optimal_routes = sorted([(r2, stop_id_to_include, r1) for r1, r2 in result])
    return optimal_routes

# PDDL-style planning for route finding
def pddl_planning(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Implement PDDL-style planning to find routes with optional transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID for a transfer.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id1 (int): The ID of the first route.
              - stop_id (int): The ID of the intermediate stop.
              - route_id2 (int): The ID of the second route.
    """
    # Query for optimal routes using OptimalRoute predicate
    result = OptimalRoute(R1, R2, start_stop_id, end_stop_id, stop_id_to_include).ask()
    
    # Extracting the optimal routes 
    optimal_routes = sorted([(r1, stop_id_to_include, r2) for r1, r2 in result])
    return optimal_routes

# Function to filter fare data based on an initial fare limit
def prune_data(merged_fare_df, initial_fare):
    """
    Filter fare data based on an initial fare limit.

    Args:
        merged_fare_df (DataFrame): The merged fare DataFrame.
        initial_fare (float): The maximum fare allowed.

    Returns:
        DataFrame: A filtered DataFrame containing only routes within the fare limit.
    """
    pruned_df = merged_fare_df[merged_fare_df['price'] <= initial_fare]
    return pruned_df

# Pre-computation of Route Summary
def compute_route_summary(pruned_df):
    """
    Generate a summary of routes based on fare information.

    Args:
        pruned_df (DataFrame): The filtered DataFrame containing fare information.

    Returns:
        dict: A summary of routes with the following structure:
              {
                  route_id (int): {
                      'min_price': float,          # The minimum fare for the route
                      'stops': set                # A set of stop IDs for that route
                  }
              }
    """
    route_summary = {}
    
    route_fares = pruned_df.groupby('route_id')['price'].min()

    for route_id, min_price in route_fares.items():
        route_summary[route_id] = {
            'min_price': min_price,
            'stops': set(route_to_stops.get(route_id, [])) # Retrieve stops as a set for unique stops
        }
    
    return route_summary

# BFS for optimized route planning
def bfs_route_planner_optimized(start_stop_id, end_stop_id, initial_fare, route_summary, max_transfers=3):
    """
    Use Breadth-First Search (BFS) to find the optimal route while considering fare constraints.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        initial_fare (float): The available fare for the trip.
        route_summary (dict): A summary of routes with fare and stop information.
        max_transfers (int): The maximum number of transfers allowed (default is 3).

    Returns:
        list: A list representing the optimal route with stops and routes taken, structured as:
              [
                  (route_id (int), stop_id (int)),  # Tuple for each stop taken in the route
                  ...
              ]
    """
    visited_stops = set()
    search_queue = deque([(start_stop_id, None, [], 0, 0)])  

    while (len(search_queue)!=0):
        current_stop, previous_route, current_route_path, accumulated_fare, transfers_made = search_queue.popleft()

        if current_stop == end_stop_id:
            result_path = current_route_path + [(previous_route, current_stop)]
            final_result = []

            for i in range(len(result_path) - 1):
                final_result.append((result_path[i][0], result_path[i + 1][1]))

            return final_result

        if transfers_made >= max_transfers or accumulated_fare > initial_fare:
            continue
        
        visited_stops.add((current_stop, previous_route))

        for route_id, route_info in route_summary.items():
            if route_id == previous_route:
                continue  

            if current_stop in route_info['stops'] and route_info['min_price'] + accumulated_fare <= initial_fare:
                new_route_path = current_route_path + [(route_id, current_stop)]
                new_fare = accumulated_fare + route_info['min_price']
                new_transfers = transfers_made + (1 if previous_route else 0)

                for next_stop in route_info['stops']:
                    if (next_stop, route_id) not in visited_stops:
                        search_queue.append((next_stop, route_id, new_route_path, new_fare, new_transfers))

    return []
