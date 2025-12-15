import sys
import math
import pandas as pd  # type: ignore
from mip import Model, xsum, BINARY, INTEGER, CONTINUOUS, OptimizationStatus, MINIMIZE

def calculate_time(p1, p2, is_upward=None):
    """
    Calculate travel time between points p1 and p2.
    p1, p2: dictionaries or series with keys 'x', 'y', 'z'
    """
    dx = abs(p1['x'] - p2['x'])
    dy = abs(p1['y'] - p2['y'])
    dz = abs(p1['z'] - p2['z'])
    
    lateral_dist = math.sqrt(dx**2 + dy**2)
    vertical_dist = dz
    
    # Same point (zero distance)
    if lateral_dist == 0 and vertical_dist == 0:
        return 0.0

    # Determine vertical direction
    if p2['z'] > p1['z']:  # Ascending
        time = max(lateral_dist / 1.5, vertical_dist / 1.0)
    elif p2['z'] < p1['z']:  # Descending
        time = max(lateral_dist / 1.5, vertical_dist / 2.0)
    else:  # Horizontal only
        time = lateral_dist / 1.5
        
    return time

def is_connected(p1, p2):
    """
    Check if two grid points are connected according to the rules.
    """
    dist = math.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2 + (p1['z'] - p2['z'])**2)
    
    # Condition 1: Euclidean distance <= 4
    if dist <= 4:
        return True
        
    # Condition 2: Distance <= 11 AND two coordinates differ by <= 0.5
    if dist <= 11:
        diffs = [
            abs(p1['x'] - p2['x']),
            abs(p1['y'] - p2['y']),
            abs(p1['z'] - p2['z'])
        ]
        # Count how many differences are <= 0.5
        count_small_diffs = sum(1 for d in diffs if d <= 0.5)
        if count_small_diffs >= 2:
            return True
            
    return False

def solve_drone_problem(filename, max_time=600):
    # --- 1. Data Loading ---
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        return
    
    points = df.to_dict('records')
    n_points = len(points)
    
    # Problem parameters
    K = 4  # Number of drones
    
    # Define base point and entry points based on file
    # Note: Different coordinates for Building1 and Building2
    # Detect which set to use based on filename
    # Default to Building1 logic
    
    base_coords = {'x': 0, 'y': -16, 'z': 0}
    entry_condition_y = -12.5
    
    if "Edificio2" in filename or "Building4" in filename:
        base_coords = {'x': 0, 'y': -40, 'z': 0}
        entry_condition_y = -20
        
    # Node 0 is the Base. Nodes 1..N are the CSV points.
    # Build unified list: nodes[0] = Base, nodes[1..N] = Points
    nodes = [base_coords] + points
    num_nodes = len(nodes)  # N + 1
    
    # Indices
    # V = {0, ..., num_nodes-1}
    # V_grid = {1, ..., num_nodes-1}
    
    # --- 2. Graph Construction (Valid Arcs and Costs) ---
    # Build sparse adjacency matrix or list of valid arcs
    # Format: arcs = [(i, j, time_cost), ...]
    arcs = []
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            
            p_i = nodes[i]
            p_j = nodes[j]
            
            is_valid = False
            
            # Case A: Base <-> Grid connection
            if i == 0:  # Base -> Grid
                # Valid only if j is an entry point
                if p_j['y'] <= entry_condition_y:
                    is_valid = True
            elif j == 0:  # Grid -> Base
                # Valid only if i is an entry point
                if p_i['y'] <= entry_condition_y:
                    is_valid = True
            else:  # Grid <-> Grid
                # Valid if connectivity conditions are met
                if is_connected(p_i, p_j):
                    is_valid = True
            
            if is_valid:
                cost = calculate_time(p_i, p_j)
                arcs.append((i, j, cost))

    # --- 3. MIP Modeling ---
    m = Model(sense=MINIMIZE, solver_name='HIGHS')
    
    # Decision variables
    # x[i][j][k] = 1 if drone k traverses arc i -> j
    # Create variables only for valid arcs to save memory
    # x[(i,j,k)] -> var
    x = {}
    for (i, j, cost) in arcs:
        for k in range(K):
            x[(i, j, k)] = m.add_var(var_type=BINARY, name=f'x_{i}_{j}_{k}')
            
    # Variable for maximum time (to minimize)
    T = m.add_var(var_type=CONTINUOUS, name='T_max', lb=0)  # type: ignore
    
    # MTZ variables for subtour elimination (u[i])
    # u[i] represents the visit order for node i
    u = [m.add_var(var_type=INTEGER, name=f'u_{i}', lb=0, ub=num_nodes) for i in range(num_nodes)]  # type: ignore

    # --- 4. Constraints ---
    
    # A. Each grid point must be visited exactly once by one drone
    for j in range(1, num_nodes):  # For each grid node (excluding Base 0)
        # Sum over all drones and all source nodes i
        incoming_arcs = [x[(i, j, k)] for (i, dest, cost) in arcs for k in range(K) if dest == j]
        m += xsum(incoming_arcs) == 1
        
    # B. Drone flow at Base
    # Each drone leaves the base (0) exactly once
    for k in range(K):
        outgoing_base = [x[(0, j, k)] for (src, j, cost) in arcs if src == 0]
        m += xsum(outgoing_base) == 1
        
    # Each drone returns to the base (0) exactly once
    for k in range(K):
        incoming_base = [x[(i, 0, k)] for (i, dest, cost) in arcs if dest == 0]
        m += xsum(incoming_base) == 1
        
    # C. Flow conservation
    # If drone k arrives at j, it must depart from j
    for k in range(K):
        for node in range(1, num_nodes):
            incoming = [x[(i, node, k)] for (i, dest, cost) in arcs if dest == node]
            outgoing = [x[(node, j, k)] for (src, j, cost) in arcs if src == node]
            m += xsum(incoming) - xsum(outgoing) == 0

    # D. Time calculation and Min-Max
    # Total time of drone k must be <= T
    for k in range(K):
        drone_time = xsum(x[(i, j, k)] * cost for (i, j, cost) in arcs)
        m += drone_time <= T

    # E. Subtour Elimination (MTZ constraints)
    # u_i - u_j + M * x_ijk <= M - 1
    # M can be num_nodes
    M = num_nodes
    for (i, j, cost) in arcs:
        if i != 0 and j != 0:  # Only between grid nodes
            for k in range(K):
                m += u[i] - u[j] + M * x[(i, j, k)] <= M - 1

    # --- 5. Objective Function ---
    m.objective = T  # Minimize T
    
    # --- 6. Solve ---
    # Solver configuration:
    # - Time limit: 10 minutes (600 seconds)
    # - Optimality gap: 5% (accept solutions within 5% of optimum)
    m.max_mip_gap = 0.05  # type: ignore
    m.max_seconds = max_time
    status = m.optimize()
    
    if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:
        if T.x is not None:
            print(f"Maximum time (objective): {T.x:.2f}")
            # --- 7. Formatted Output ---
            for k in range(K):
                # Path reconstruction
                path = []
                current_node = 0
                
                # Loop to find the sequence
                while True:
                    path.append(current_node)
                    # Find the next node
                    next_node = -1
                    for (i, j, cost) in arcs:
                        if i == current_node and x[(i, j, k)].x is not None and x[(i, j, k)].x >= 0.99:
                            next_node = j
                            break
                    
                    if next_node == -1:
                        break  # Should not happen if model is correct
                        
                    current_node = next_node
                    if current_node == 0:
                        path.append(0)
                        break
                
                # Format string "0-4-11-..."
                path_str = "-".join(map(str, path))
                print(f"Drone {k+1}: {path_str}")
        else:
            print("No valid solution found.")
    else:
        print(f"No solution found. Status: {status}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <filename.csv>")
    else:
        solve_drone_problem(sys.argv[1])
