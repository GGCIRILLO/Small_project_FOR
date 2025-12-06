import csv
import math
import mip
import sys


def parse_input(filename):
    points = []
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        next(reader) # Salta l'header "x,y,z"

        for idx, row in enumerate(reader):
            x = float(row[0])
            y = float(row[1])
            z = float(row[2])
            points.append((x, y, z, idx+1))  # idx+1 perché 0 è la base

    return points

def get_entry_points(points, y_threshold):
    entry_points = []
    for point in points:
        x, y, z, idx = point
        if y <= y_threshold:
            entry_points.append(idx)
    return entry_points

def euclidean_distance(p1, p2):
    """Calcola distanza euclidea tra due punti"""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]
    return math.sqrt(dx**2 + dy**2 + dz**2)

def coords_differ_by_max(p1, p2, threshold=0.5):
    """Conta quante coordinate differiscono di al più threshold"""
    count = 0
    if abs(p1[0] - p2[0]) <= threshold:
        count += 1
    if abs(p1[1] - p2[1]) <= threshold:
        count += 1
    if abs(p1[2] - p2[2]) <= threshold:
        count += 1
    return count

def are_connected(p1, p2):
    """
    Due punti sono connessi se:
    - distanza <= 4m, OPPURE
    - distanza <= 11m E almeno 2 coordinate differiscono di <= 0.5m
    """
    dist = euclidean_distance(p1, p2)

    # Regola 1
    if dist <= 4.0:
        return True

    # Regola 2
    if dist <= 11.0 and coords_differ_by_max(p1, p2) >= 2:
        return True

    return False

def calculate_travel_time(p1, p2):
    """
    Calcola il tempo di viaggio considerando velocità asimmetriche
    """
    # Componenti del movimento
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]

    # Movimento orizzontale (proiezione su piano xy)
    horizontal_dist = math.sqrt(dx**2 + dy**2)

    # Movimento verticale
    vertical_dist = abs(dz)

    # Calcola tempo per componente orizzontale
    time_horizontal = horizontal_dist / 1.5  # 1.5 m/s

    # Calcola tempo per componente verticale
    if dz > 0:  # Salita
        time_vertical = vertical_dist / 1.0  # 1 m/s
    else:  # Discesa
        time_vertical = vertical_dist / 2.0  # 2 m/s

    # Per movimento obliquo: prendi il massimo
    return max(time_horizontal, time_vertical)

def build_graph_with_times(points, base_point, entry_points):
    """
    Ritorna: {(i,j): travel_time}
    Se l'arco non esiste, non è nel dict
    """
    graph = {}
    all_points = [base_point] + points
    n = len(points)

    # Archi tra punti griglia
    for i in range(n):
        for j in range(i+1, n):
            if are_connected(points[i], points[j]):
                idx_i = points[i][3]
                idx_j = points[j][3]

                time_ij = calculate_travel_time(points[i], points[j])
                time_ji = calculate_travel_time(points[j], points[i])

                graph[(idx_i, idx_j)] = time_ij
                graph[(idx_j, idx_i)] = time_ji  # Tempi asimmetrici!

    # Archi base <-> entry points
    for point in points:
        idx = point[3]
        if idx in entry_points:
            graph[(0, idx)] = calculate_travel_time(base_point, point)
            graph[(idx, 0)] = calculate_travel_time(point, base_point)

    return graph

def solve_drone_routing(points, base_point, graph, entry_points, num_drones=4):
    """
    Risolve il problema di routing dei droni usando MIP
    
    Args:
        points: lista di punti (x, y, z, idx)
        base_point: punto di partenza (x, y, z, 0)
        graph: {(i,j): travel_time}
        entry_points: set di indici degli entry points
        num_drones: numero di droni (default 4)
    
    Returns:
        dict: {drone_id: [sequenza di nodi]}
    """
    
    # Crea il modello
    model = mip.Model(sense=mip.MINIMIZE, solver_name=mip.CBC)
    
    # Nodi
    n = len(points)
    nodes = [0] + [p[3] for p in points]  # 0 = base, poi indici punti
    grid_nodes = [p[3] for p in points]   # Solo punti griglia (no base)
    
    # Droni
    K = range(1, num_drones + 1)
    
    print(f"Creazione modello: {n} punti, {num_drones} droni, {len(graph)} archi")
    
    # ========== VARIABILI ==========
    
    # x[i,j,k] = 1 se drone k usa arco (i,j)
    x = {}
    for (i, j) in graph.keys():
        for k in K:
            x[i, j, k] = model.add_var(var_type=mip.BINARY, 
                                       name=f'x_{i}_{j}_{k}')
    
    # T = makespan (tempo massimo)
    T = model.add_var(var_type=mip.CONTINUOUS, name='makespan', lb=0)
    
    # ========== VINCOLI ==========
    
    # 1. Coverage: ogni punto visitato esattamente una volta
    print("Aggiunta vincoli coverage...")
    for i in grid_nodes:
        model.add_constr(
            mip.xsum(x[j, i, k] 
                    for k in K 
                    for j in nodes 
                    if (j, i) in graph) == 1,
            name=f'coverage_{i}'
        )
    
    # 2. Flow conservation: se entri devi uscire
    print("Aggiunta vincoli flow conservation...")
    for i in nodes:
        for k in K:
            # Archi entranti in i
            in_arcs = mip.xsum(x[j, i, k] 
                              for j in nodes 
                              if (j, i) in graph)
            # Archi uscenti da i
            out_arcs = mip.xsum(x[i, j, k] 
                               for j in nodes 
                               if (i, j) in graph)
            
            model.add_constr(in_arcs == out_arcs, 
                           name=f'flow_{i}_{k}')
    
    # 3. Departure from base: ogni drone parte al massimo una volta
    print("Aggiunta vincoli departure...")
    for k in K:
        model.add_constr(
            mip.xsum(x[0, j, k] 
                    for j in grid_nodes 
                    if (0, j) in graph) <= 1,
            name=f'departure_{k}'
        )
    
    # 4. Return to base è già garantito da flow conservation
    
    # 5. Makespan: tempo di ogni drone <= T
    print("Aggiunta vincoli makespan...")
    for k in K:
        drone_time = mip.xsum(graph[i, j] * x[i, j, k] 
                             for (i, j) in graph.keys())
        model.add_constr(drone_time <= T, 
                        name=f'makespan_{k}')
  
    
    # ========== FUNZIONE OBIETTIVO ==========
    model.objective = T
    
    print(f"Modello creato: {model.num_cols} variabili, {model.num_rows} vincoli")
    
    # ========== RISOLUZIONE ==========
    print("\nInizio ottimizzazione...")
    
    # Imposta parametri del solver
    model.max_gap = 0.05  # Gap di ottimalità 5%
    model.max_seconds = 300  # Timeout 5 minuti
    
    status = model.optimize()
    
    # ========== ESTRAZIONE SOLUZIONE ==========
    
    if status == mip.OptimizationStatus.OPTIMAL:
        print(f"\n✓ Soluzione ottima trovata!")
    elif status == mip.OptimizationStatus.FEASIBLE:
        print(f"\n✓ Soluzione feasible trovata (non ottima)")
    else:
        print(f"\n✗ Nessuna soluzione trovata")
        return None
    
    print(f"Makespan: {T.x:.2f} secondi")
    print(f"Gap: {model.gap:.2%}")
    
    # Estrai i percorsi
    routes = {k: [] for k in K}
    
    for k in K:
        # Trova il percorso del drone k
        current = 0  # Parti dalla base
        route = [0]
        visited = set([0])
        
        while True:
            # Trova il prossimo nodo
            next_node = None
            for j in nodes:
                if (current, j) in graph and x[current, j, k].x > 0.5:
                    next_node = j
                    break
            
            if next_node is None or next_node == 0:
                route.append(0)  # Torna alla base
                break
            
            if next_node in visited and next_node != 0:
                print(f"WARNING: ciclo rilevato nel drone {k}")
                break
            
            route.append(next_node)
            visited.add(next_node)
            current = next_node
        
        routes[k] = route
    
    return routes, T.x

def main(filename):
    """
    Main function che orchestra tutto
    """
    # Determina quale edificio (per base ed entry points)
    if "Buildinig1" in filename or "Building1" in filename:
        base_point = (0, -16, 0, 0)
        y_threshold = -12.5
    elif "Building2" in filename or "building2" in filename:
        base_point = (0, -40, 0, 0)
        y_threshold = -20
    else:
        print("File non riconosciuto. Uso default Edificio1.")
        base_point = (0, -16, 0, 0)
        y_threshold = -12.5
    
    # Step 1: Parse input
    print(f"Lettura file: {filename}")
    points = parse_input(filename)
    print(f"Punti letti: {len(points)}")
    
    # Step 2: Identifica entry points
    entry_points = get_entry_points(points, y_threshold)
    print(f"Entry points: {len(entry_points)}")
    
    # Step 3: Build graph
    print("Costruzione grafo...")
    graph = build_graph_with_times(points, base_point, entry_points)
    print(f"Archi creati: {len(graph)}")
    
    # Step 4: Solve MIP
    print("\n" + "="*50)
    result = solve_drone_routing(points, base_point, graph, entry_points)
    
    if result is None:
        print("Impossibile trovare una soluzione!")
        return
    
    routes, makespan = result
    
    # Step 5: Output
    print("\n" + "="*50)
    print("SOLUZIONE:")
    print("="*50)
    # print_solution(routes)
    print(f"\nTempo totale: {makespan:.2f} secondi")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python main.py <input_file.csv>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    main(input_file)

