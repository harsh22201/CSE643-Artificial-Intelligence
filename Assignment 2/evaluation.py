import time
import tracemalloc
import itertools
import numpy as np
import random
import matplotlib.pyplot as plt
from code_2022201 import *

initialize_datalog()
print("Datalog initialized")


def measure_time_and_space(func, *args, **kwargs):
    # Start tracking memory
    tracemalloc.start()

    # Start timing
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()

    # Stop tracking memory
    current, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Calculate execution time
    execution_time = end_time - start_time

    return result, execution_time, peak_memory


def evaluate_reasoning():
    stops = df_stops['stop_id'].tolist()  # Use all stops
    all_pairs = list(itertools.combinations(stops, 2))
    sampled_pairs = random.sample(all_pairs, 100)  # Randomly select 45 pairs
    brute_force_times = []
    brute_force_memories = []
    fol_times = []
    fol_memories = []

    # Measure for brute force function
    for start, end in sampled_pairs:
        _, time_taken, memory_used = measure_time_and_space(direct_route_brute_force, start, end)
        brute_force_times.append(time_taken * 1000)  # Convert to ms
        brute_force_memories.append(memory_used / (1024 * 1024))  # Convert to MB

    # Measure for FOL query function
    for start, end in sampled_pairs:
        _, time_taken, memory_used = measure_time_and_space(query_direct_routes, start, end)
        fol_times.append(time_taken * 1000)  # Convert to ms
        fol_memories.append(memory_used / (1024 * 1024))  # Convert to MB

    # Calculate averages
    avg_brute_force_time = np.mean(brute_force_times)
    avg_brute_force_memory = np.mean(brute_force_memories)
    avg_fol_time = np.mean(fol_times)
    avg_fol_memory = np.mean(fol_memories)

    print("-------REASONING EVALUATION RESULTS-------")
    print("Brute force average time (ms):", avg_brute_force_time)
    print("Brute force average memory (MB):", avg_brute_force_memory)
    print("FOL query average time (ms):", avg_fol_time)
    print("FOL query average memory (MB):", avg_fol_memory)

    # Plotting scatter plots for time and space
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Scatter plot for execution time
    axs[0].scatter(range(len(brute_force_times)), brute_force_times, color='blue', label='Brute Force Time (ms)')
    axs[0].scatter(range(len(fol_times)), fol_times, color='orange', label='FOL Query Time (ms)')
    axs[0].set_title('Execution Time for Sampled Pairs')
    axs[0].set_xlabel('Sampled Pair Index')
    axs[0].set_ylabel('Time (ms)')
    axs[0].legend()

    # Scatter plot for memory usage
    axs[1].scatter(range(len(brute_force_memories)), brute_force_memories, color='blue', label='Brute Force Memory (MB)')
    axs[1].scatter(range(len(fol_memories)), fol_memories, color='orange', label='FOL Query Memory (MB)')
    axs[1].set_title('Memory Usage for Sampled Pairs')
    axs[1].set_xlabel('Sampled Pair Index')
    axs[1].set_ylabel('Memory (MB)')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

# evaluate forward chaining and backward chaining
def evaulate_planning():
    inputs = [(22540, 2573, 4686, 1),(951, 340, 300, 1)]
    forward_times = []
    forward_memories = []
    backward_times = []
    backward_memories = []

    # Measure for forward chaining function
    for start_stop_id, end_stop_id, stop_id_to_include, max_transfers in inputs:
        _, time_taken, memory_used = measure_time_and_space(forward_chaining, start_stop_id, end_stop_id, stop_id_to_include, max_transfers)
        forward_times.append(time_taken * 1000)  # Convert to ms
        forward_memories.append(memory_used / (1024 * 1024))  # Convert to MB

    # Measure for backward chaining function
    for start_stop_id, end_stop_id, stop_id_to_include, max_transfers in inputs:
        _, time_taken, memory_used = measure_time_and_space(backward_chaining, start_stop_id, end_stop_id, stop_id_to_include, max_transfers)
        backward_times.append(time_taken * 1000)  # Convert to ms
        backward_memories.append(memory_used / (1024 * 1024))  # Convert to MB

    # Calculate averages
    avg_forward_time = np.mean(forward_times)
    avg_forward_memory = np.mean(forward_memories)
    avg_backward_time = np.mean(backward_times)
    avg_backward_memory = np.mean(backward_memories)

    print("-------PLANNING EVALUATION RESULTS-------")
    print("Forward chaining average time (ms):", avg_forward_time)
    print("Forward chaining average memory (MB):", avg_forward_memory)
    print("Backward chaining average time (ms):", avg_backward_time)
    print("Backward chaining average memory (MB):", avg_backward_memory)

    # Plotting scatter plots for time and space
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Scatter plot for execution time
    axs[0].scatter(range(len(forward_times)), forward_times, color='blue', label='Forward Chaining Time (ms)')
    axs[0].scatter(range(len(backward_times)), backward_times, color='orange', label='Backward Chaining Time (ms)')
    axs[0].set_title('Execution Time for Sampled Pairs')
    axs[0].set_xlabel('Sampled Pair Index')
    axs[0].set_ylabel('Time (ms)')


# Run the evaluation
evaluate_reasoning()
evaulate_planning()