import sys
import math
import pandas as pd  # type: ignore
from pulp import LpProblem, LpMinimize, LpVariable, LpBinary, LpInteger, LpContinuous, lpSum, LpStatus, value, PULP_CBC_CMD

def calculate_time(p1, p2, is_upward=None):
    """
    Calcola il tempo di percorrenza tra due punti p1 e p2.
    p1, p2: dizionari o serie con chiavi 'x', 'y', 'z'
    """
    dx = abs(p1['x'] - p2['x'])
    dy = abs(p1['y'] - p2['y'])
    dz = abs(p1['z'] - p2['z'])
    
    lateral_dist = math.sqrt(dx**2 + dy**2)
    vertical_dist = dz
    
    # Se p1 e p2 sono lo stesso punto (distanza 0)
    if lateral_dist == 0 and vertical_dist == 0:
        return 0.0

    # Determina la direzione verticale
    if p2['z'] > p1['z']: # Salita
        time = max(lateral_dist / 1.5, vertical_dist / 1.0)
    elif p2['z'] < p1['z']: # Discesa
        time = max(lateral_dist / 1.5, vertical_dist / 2.0)
    else: # Solo orizzontale
        time = lateral_dist / 1.5
        
    return time

def is_connected(p1, p2):
    """
    Verifica se due punti della griglia sono connessi secondo le regole.
    """
    dist = math.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2 + (p1['z'] - p2['z'])**2)
    
    # Condizione 1: Distanza Euclidea <= 4
    if dist <= 4:
        return True
        
    # Condizione 2: Distanza <= 11 AND due coordinate differiscono di <= 0.5
    if dist <= 11:
        diffs = [
            abs(p1['x'] - p2['x']),
            abs(p1['y'] - p2['y']),
            abs(p1['z'] - p2['z'])
        ]
        # Conta quante differenze sono <= 0.5
        count_small_diffs = sum(1 for d in diffs if d <= 0.5)
        if count_small_diffs >= 2:
            return True
            
    return False

def solve_drone_problem(filename, max_time=600):
    # --- 1. Caricamento Dati ---
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Errore: File {filename} non trovato.")
        return
    
    points = df.to_dict('records')
    n_points = len(points)
    
    # Parametri del problema
    K = 4  # Numero di droni
    
    # Definizione punto base e entry points in base al file
    # Nota: Il problema specifica coordinate diverse per Edificio1 e Edificio2
    # Rileviamo quale set usare in base al nome del file o euristica sulle coordinate
    # Per sicurezza usiamo la logica descritta per Edificio1 come default o controllo
    
    base_coords = {'x': 0, 'y': -16, 'z': 0}
    entry_condition_y = -12.5
    
    if "Edificio2" in filename or "Building4" in filename:
        base_coords = {'x': 0, 'y': -40, 'z': 0}
        entry_condition_y = -20
        
    # Nodo 0 sarà la Base. I nodi 1..N saranno i punti del CSV.
    # Costruiamo una lista unificata: nodes[0] = Base, nodes[1..N] = Points
    nodes = [base_coords] + points
    num_nodes = len(nodes) # N + 1
    
    # Indici
    # V = {0, ..., num_nodes-1}
    # V_grid = {1, ..., num_nodes-1}
    
    # --- 2. Costruzione del Grafo (Archi validi e Costi) ---
    # Costruiamo una matrice di adiacenza sparsa o lista di archi validi
    # Format: arcs = [(i, j, time_cost), ...]
    arcs = []
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            
            p_i = nodes[i]
            p_j = nodes[j]
            
            is_valid = False
            
            # Caso A: Connessione Base <-> Griglia
            if i == 0: # Base -> Griglia
                # Valido solo se j è un "entry point"
                if p_j['y'] <= entry_condition_y:
                    is_valid = True
            elif j == 0: # Griglia -> Base
                # Valido solo se i è un "entry point"
                if p_i['y'] <= entry_condition_y:
                    is_valid = True
            else: # Griglia <-> Griglia
                # Valido se rispettano le condizioni di connettività
                if is_connected(p_i, p_j):
                    is_valid = True
            
            if is_valid:
                cost = calculate_time(p_i, p_j)
                arcs.append((i, j, cost))

    # --- 3. Modellazione MIP ---
    m = LpProblem("Drone_Routing", LpMinimize)
    
    # Variabili decisionali
    # x[i][j][k] = 1 se il drone k percorre l'arco i -> j
    # Per risparmiare memoria, creiamo variabili solo per gli archi validi
    # x[(i,j,k)] -> var
    x = {}
    for (i, j, cost) in arcs:
        for k in range(K):
            x[(i, j, k)] = LpVariable(f'x_{i}_{j}_{k}', cat=LpBinary)
            
    # Variabile per il tempo massimo (da minimizzare)
    T = LpVariable('T_max', lowBound=0, cat=LpContinuous)
    
    # Variabili MTZ per eliminazione sottogiri (u[i])
    # u[i] rappresenta l'ordine di visita per il nodo i
    u = [LpVariable(f'u_{i}', lowBound=0, upBound=num_nodes, cat=LpInteger) for i in range(num_nodes)]

    # --- 4. Vincoli ---
    
    # A. Ogni punto della griglia deve essere visitato esattamente una volta da un solo drone
    for j in range(1, num_nodes): # Per ogni nodo griglia (escluso Base 0)
        # Somma su tutti i droni e tutti i nodi di provenienza i
        incoming_arcs = [x[(i, j, k)] for (i, dest, cost) in arcs for k in range(K) if dest == j]
        m += lpSum(incoming_arcs) == 1
        
    # B. Flusso dei droni alla Base
    # Ogni drone parte dalla base (0) esattamente una volta
    for k in range(K):
        outgoing_base = [x[(0, j, k)] for (src, j, cost) in arcs if src == 0]
        m += lpSum(outgoing_base) == 1
        
    # Ogni drone ritorna alla base (0) esattamente una volta
    for k in range(K):
        incoming_base = [x[(i, 0, k)] for (i, dest, cost) in arcs if dest == 0]
        m += lpSum(incoming_base) == 1
        
    # C. Conservazione del flusso
    # Se un drone k arriva in j, deve ripartire da j
    for k in range(K):
        for node in range(1, num_nodes):
            incoming = [x[(i, node, k)] for (i, dest, cost) in arcs if dest == node]
            outgoing = [x[(node, j, k)] for (src, j, cost) in arcs if src == node]
            m += lpSum(incoming) - lpSum(outgoing) == 0

    # D. Calcolo del tempo e Min-Max
    # Il tempo totale del drone k deve essere <= T
    for k in range(K):
        drone_time = lpSum(x[(i, j, k)] * cost for (i, j, cost) in arcs)
        m += drone_time <= T

    # E. Eliminazione Sottogiri (MTZ constraints)
    # u_i - u_j + M * x_ijk <= M - 1
    # M può essere num_nodes
    M = num_nodes
    for (i, j, cost) in arcs:
        if i != 0 and j != 0: # Solo tra nodi griglia
            for k in range(K):
                m += u[i] - u[j] + M * x[(i, j, k)] <= M - 1

    # --- 5. Funzione Obiettivo ---
    m += T  # Minimizza T (già impostato come LpMinimize)
    
    # --- 6. Risoluzione ---
    # Time limit di 2 ore (7200 secondi)
    
    m.solve()
    
    if LpStatus[m.status] == 'Optimal':
        # --- 7. Output Formattato ---
        for k in range(K):
            # Ricostruzione percorso
            path = []
            current_node = 0
            
            # Loop per trovare la sequenza
            while True:
                path.append(current_node)
                # Trova il prossimo nodo
                next_node = -1
                for (i, j, cost) in arcs:
                    if i == current_node and value(x[(i, j, k)]) is not None and value(x[(i, j, k)]) >= 0.99:
                        next_node = j
                        break
                
                if next_node == -1:
                    break # Dovrebbe non succedere se il modello è corretto
                    
                current_node = next_node
                if current_node == 0:
                    path.append(0)
                    break
            
            # Formattazione stringa "0-4-11-..."
            path_str = "-".join(map(str, path))
            print(f"Drone {k+1}: {path_str}")
    else:
        print(f"Nessuna soluzione trovata. Status: {LpStatus[m.status]}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python main.py <filename.csv>")
    else:
        solve_drone_problem(sys.argv[1])
