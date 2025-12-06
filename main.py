import csv
import math
import pulp
import sys
from collections import defaultdict


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
    Risolve il problema di routing dei droni usando PuLP
    
    Args:
        points: lista di punti (x, y, z, idx)
        base_point: punto di partenza (x, y, z, 0)
        graph: {(i,j): travel_time}
        entry_points: set di indici degli entry points
        num_drones: numero di droni (default 4)
    
    Returns:
        tuple: (routes, makespan) dove routes = {drone_id: [sequenza di nodi]}
    """
    
    # Crea il modello
    model = pulp.LpProblem("DroneRouting", pulp.LpMinimize)
    
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
            x[i, j, k] = pulp.LpVariable(f'x_{i}_{j}_{k}', cat=pulp.LpBinary)
    
    # T = makespan (tempo massimo)
    T = pulp.LpVariable('makespan', lowBound=0, cat=pulp.LpContinuous)
    
    # u[i,k] = variabile di ordine per eliminazione subtour (MTZ)
    # u[i,k] indica la posizione del nodo i nel tour del drone k
    u = {}
    for i in grid_nodes:
        for k in K:
            u[i, k] = pulp.LpVariable(f'u_{i}_{k}', lowBound=1, upBound=n, cat=pulp.LpContinuous)
    
    # ========== VINCOLI ==========
    
    # 1. Coverage: ogni punto visitato esattamente una volta (da un solo drone)
    print("Aggiunta vincoli coverage...")
    for i in grid_nodes:
        model += (
            pulp.lpSum(x[j, i, k] 
                      for k in K 
                      for j in nodes 
                      if (j, i) in graph) == 1,
            f'coverage_{i}'
        )
    
    # 2. Flow conservation: se entri in un nodo devi uscire
    print("Aggiunta vincoli flow conservation...")
    for i in grid_nodes:
        for k in K:
            # Archi entranti in i
            in_arcs = pulp.lpSum(x[j, i, k] 
                                for j in nodes 
                                if (j, i) in graph)
            # Archi uscenti da i
            out_arcs = pulp.lpSum(x[i, j, k] 
                                 for j in nodes 
                                 if (i, j) in graph)
            
            model += (in_arcs == out_arcs, f'flow_{i}_{k}')
    
    # 3. Ogni drone parte dalla base al massimo una volta
    print("Aggiunta vincoli departure...")
    for k in K:
        model += (
            pulp.lpSum(x[0, j, k] 
                      for j in grid_nodes 
                      if (0, j) in graph) <= 1,
            f'departure_{k}'
        )
    
    # 4. Ogni drone torna alla base al massimo una volta (bilanciamento con partenza)
    for k in K:
        departures = pulp.lpSum(x[0, j, k] for j in grid_nodes if (0, j) in graph)
        arrivals = pulp.lpSum(x[j, 0, k] for j in grid_nodes if (j, 0) in graph)
        model += (departures == arrivals, f'return_{k}')
    
    # 5. Subtour elimination (MTZ constraints)
    print("Aggiunta vincoli subtour elimination (MTZ)...")
    for k in K:
        for (i, j) in graph.keys():
            if i != 0 and j != 0:  # Non applicare per archi da/verso la base
                # Se x[i,j,k] = 1, allora u[j,k] >= u[i,k] + 1
                model += (
                    u[j, k] >= u[i, k] + 1 - n * (1 - x[i, j, k]),
                    f'mtz_{i}_{j}_{k}'
                )
    
    # 6. Makespan: tempo di ogni drone <= T
    print("Aggiunta vincoli makespan...")
    for k in K:
        drone_time = pulp.lpSum(graph[i, j] * x[i, j, k] 
                               for (i, j) in graph.keys())
        model += (drone_time <= T, f'makespan_{k}')
    
    # ========== FUNZIONE OBIETTIVO ==========
    model += T
    
    print(f"Modello creato: {len(model.variables())} variabili, {len(model.constraints)} vincoli")
    
    # ========== RISOLUZIONE ==========
    print("\nInizio ottimizzazione...")
    
    # Usa CBC solver con timeout
    solver = pulp.PULP_CBC_CMD(msg=1, timeLimit=300, gapRel=0.05)  # type: ignore
    status = model.solve(solver)
    
    # ========== ESTRAZIONE SOLUZIONE ==========
    
    if status == pulp.LpStatusOptimal:
        print(f"\n✓ Soluzione ottima trovata!")
    elif status == pulp.LpStatusNotSolved:
        print(f"\n✗ Problema non risolto")
        return None
    elif status == pulp.LpStatusInfeasible:
        print(f"\n✗ Problema infeasible")
        return None
    elif status == pulp.LpStatusUnbounded:
        print(f"\n✗ Problema unbounded")
        return None
    else:
        print(f"\n✓ Soluzione trovata (status: {pulp.LpStatus[status]})")
    
    makespan_value = pulp.value(T)
    print(f"Makespan: {makespan_value:.2f} secondi")
    
    # Estrai i percorsi
    routes = {k: [] for k in K}
    
    for k in K:
        # Trova il percorso del drone k partendo dalla base
        current = 0
        route = [0]
        visited = set()
        
        while True:
            # Trova il prossimo nodo
            next_node = None
            for j in nodes:
                if (current, j) in graph:
                    val = pulp.value(x[current, j, k])
                    if val is not None and val > 0.5:
                        next_node = j
                        break
            
            if next_node is None or next_node == 0:
                if current != 0:  # Se non siamo già alla base, torniamo
                    route.append(0)
                break
            
            if next_node in visited:
                print(f"WARNING: ciclo rilevato nel drone {k} al nodo {next_node}")
                route.append(0)
                break
            
            route.append(next_node)
            visited.add(next_node)
            current = next_node
        
        routes[k] = route
    
    return routes, makespan_value


def print_solution(routes):
    """
    Stampa la soluzione nel formato richiesto:
    Drone 1: 0-4-11-17-...-2-0
    """
    for k in sorted(routes.keys()):
        route = routes[k]
        if len(route) > 2:  # Ha visitato almeno un punto
            route_str = '-'.join(map(str, route))
            print(f"Drone {k}: {route_str}")
        else:
            # Drone non usato
            print(f"Drone {k}: 0-0")

def main(filename):
    """
    Main function che orchestra tutto
    """
    # Determina quale edificio (per base ed entry points)
    if "Edificio1" in filename or "Building1" in filename:
        base_point = (0, -16, 0, 0)
        y_threshold = -12.5
    elif "Edificio2" in filename or "Building2" in filename:
        base_point = (0, -40, 0, 0)
        y_threshold = -20
    else:
        print("File non riconosciuto. Uso default Edificio1.")
        base_point = (0, -16, 0, 0)
        y_threshold = -12.5
    
    # Step 1: Parse input
    print(f"Lettura file: {filename}")
    all_points = parse_input(filename)
    print(f"Punti totali nel file: {len(all_points)}")
    
    # Per test con subset ridotto: imposta TEST_SIZE a None per usare tutti i punti
    TEST_SIZE = 25  
    
    if TEST_SIZE and TEST_SIZE < len(all_points):
        # Trova entry points e punti normali
        entry_point_list = [p for p in all_points if p[1] <= y_threshold]
        normal_points = [p for p in all_points if p[1] > y_threshold]
        
        print(f"Entry points totali: {len(entry_point_list)}")
        print(f"Punti normali totali: {len(normal_points)}")
        
        # Target: ~30% entry points (necessari per connettività e feasibility)
        num_entry_points = max(6, TEST_SIZE // 3)  # ~8 entry points per 25 punti
        num_normal_points = TEST_SIZE - num_entry_points
        
        print(f"Target: {num_entry_points} entry points, {num_normal_points} punti normali")
        
        # Seleziona entry points più vicini alla base
        entry_point_list.sort(key=lambda p: euclidean_distance(base_point, p))
        selected_entry_points = entry_point_list[:num_entry_points]
        
        # Set di TUTTI gli entry point indices (per escluderli durante l'espansione)
        all_entry_indices = {p[3] for p in entry_point_list}
        
        # Inizia con gli entry points selezionati
        selected_points = selected_entry_points.copy()
        selected_indices = {p[3] for p in selected_entry_points}
        
        # Espandi BFS-style ma SOLO verso punti normali (non entry points)
        frontier = selected_entry_points.copy()
        
        while len(selected_points) < TEST_SIZE and frontier:
            # Prendi il punto più vicino alla base dal frontier
            frontier.sort(key=lambda p: euclidean_distance(base_point, p))
            current = frontier.pop(0)
            
            # Trova vicini non ancora selezionati, ESCLUDENDO altri entry points
            candidates = []
            for p in all_points:
                if p[3] in selected_indices:
                    continue
                # ESCLUDI entry points non ancora selezionati
                if p[3] in all_entry_indices and p[3] not in {ep[3] for ep in selected_entry_points}:
                    continue
                if are_connected(current, p):
                    candidates.append(p)
            
            # Ordina candidati per distanza dalla base (più vicini prima)
            candidates.sort(key=lambda p: euclidean_distance(base_point, p))
            
            for p in candidates:
                if len(selected_points) >= TEST_SIZE:
                    break
                if p[3] not in selected_indices:
                    selected_points.append(p)
                    selected_indices.add(p[3])
                    frontier.append(p)
        
        # Re-indicizza
        points = []
        for new_idx, p in enumerate(selected_points, start=1):
            points.append((p[0], p[1], p[2], new_idx))
        
        print(f"Punti selezionati: {len(points)}")
        
        # Verifica distribuzione
        num_entry_in_selection = sum(1 for p in selected_points if p[1] <= y_threshold)
        print(f"Entry points nella selezione: {num_entry_in_selection}/{len(points)} ({num_entry_in_selection/len(points)*100:.1f}%)")
    else:
            points = all_points
    
    # Step 2: Identifica entry points
    entry_points = get_entry_points(points, y_threshold)
    print(f"Entry points: {len(entry_points)}")
    
    # Step 3: Build graph
    print("Costruzione grafo...")
    graph = build_graph_with_times(points, base_point, entry_points)
    print(f"Archi creati: {len(graph)}")
    
    reachable = set([0])  # Base è raggiungibile
    changed = True
    while changed:
        changed = False
        for (i, j) in graph.keys():
            if i in reachable and j not in reachable:
                reachable.add(j)
                changed = True

    unreachable = set([p[3] for p in points]) - reachable
    print(f"Punti raggiungibili dalla base: {len(reachable)-1} su {len(points)}")
    if unreachable:
        print(f"⚠️ WARNING: {len(unreachable)} punti NON raggiungibili dalla base!")
        print(f"Punti isolati: {sorted(unreachable)}")
    
    # Step 4: Solve MIP
    print("\n" + "="*50)
    result = solve_drone_routing(points, base_point, graph, entry_points)
    
    if result is None:
        print("Impossibile trovare una soluzione!")
        return
    
    routes, makespan = result
    
    # Step 5: Output nel formato richiesto
    print("\n" + "="*50)
    print("SOLUZIONE:")
    print("="*50)
    print_solution(routes)
    print(f"\nMakespan (tempo ultimo drone): {makespan:.2f} secondi")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python main.py <input_file.csv>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    main(input_file)