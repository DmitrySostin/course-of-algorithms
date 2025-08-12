import json
import math
import heapq
import sys
import time
from matplotlib import pyplot as plt

with open("points.json", "r") as f:
    points = json.load(f)["points"]               # Графы
with open("points.json", "r") as f:
    orders = json.load(f)["orders"]               # Параметры груза
with open("points.json", "r") as f:
    vehicles = json.load(f)["vehicles"]           # Параметры транспортного средства

def slow_print(text, delay=0.05, color=None, end="\n"):
    """Выводит цветной текст посимвольно"""
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'reset': '\033[0m'
    }
    
    if color in colors:
        sys.stdout.write(colors[color])
    
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    
    if color in colors:
        sys.stdout.write(colors['reset'])
    sys.stdout.write(end)
    sys.stdout.flush()

def slow_input(prompt, delay=0.05, color=None):
    """Аналог input() с посимвольным выводом приглашения"""
    slow_print(prompt, delay, color, end=' ')
    return input()

def build_weighted_adjacency_matrix(points):
    """Строит матрицу смежности с весами = расстояниями."""
    # Сопоставляем каждой вершине индекс (для удобства)
    vertex_index = {point['id']: idx for idx, point in enumerate(points)}
    n = len(points)
    INF = float('inf')
    
    # Инициализация матрицы (изначально все расстояния = ∞)
    adj_matrix = [[INF] * n for _ in range(n)]
    
    # Заполняем матрицу
    for point in points:
        src_idx = vertex_index[point['id']]
        adj_matrix[src_idx][src_idx] = 0  # Расстояние до себя = 0
        
        # Перебираем всех соседей и вычисляем расстояния
        for neighbor_id in point['connections']:
            dst_idx = vertex_index[neighbor_id]
            distance = euclidean_distance(point, points[dst_idx])
            adj_matrix[src_idx][dst_idx] = distance
            # Если граф неориентированный, добавляем обратное ребро:
            adj_matrix[dst_idx][src_idx] = distance
    
    return adj_matrix, vertex_index


    """ Алгоритм Флойда-Уоршелла """
    n = len(graph)
    dist = [[0] * n for _ in range(n)]
    
    # Инициализация матрицы расстояний
    for i in range(n):
        for j in range(n):
            dist[i][j] = graph[i][j]
    
    # Основной цикл алгоритма
    for k in range(n):
        for i in range(n):
            for j in range(n):
                # Если путь через вершину k короче
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    
    return dist


    """ Алгоритм Дейкстры """
    n = len(graph)
    visited = [False] * n
    distances = [sys.maxsize] * n
    distances[start] = 0
    
    for _ in range(n):
        # Находим вершину с минимальным расстоянием
        min_dist = sys.maxsize
        u = -1
        for i in range(n):
            if not visited[i] and distances[i] < min_dist:
                min_dist = distances[i]
                u = i
                
        if u == -1:
            break
            
        visited[u] = True
        
        # Обновляем расстояния до соседей
        for v in range(n):
            if graph[u][v] > 0 and not visited[v]:
                new_dist = distances[u] + graph[u][v]
                if new_dist < distances[v]:
                    distances[v] = new_dist
                    
    return distances

def dijkstra(graph, start_index):
    n = len(graph)
    distances = [float('inf')] * n
    distances[start_index] = 0
    visited = [False] * n

    for _ in range(n):
        u = min((dist, idx) for idx, dist in enumerate(distances) if not visited[idx])[1]
        visited[u] = True
        for v in range(n):
            if graph[u][v] > 0 and not visited[v]:
                new_dist = distances[u] + graph[u][v]
                if new_dist < distances[v]:
                    distances[v] = new_dist
    return distances

def floyd_warshall_from_source(points, start_id):
    """ Алгоритм Флойда-Уоршелла для нахождения кратчайших путей от заданной точки """
    # Инициализация
    n = len(points)
    INF = float('inf')
    vertex_index = {point['id']: i for i, point in enumerate(points)}
    start_idx = vertex_index[start_id]
    
    # Матрицы расстояний и путей
    dist = [[INF] * n for _ in range(n)]
    next_node = [[None] * n for _ in range(n)]
    
    # Заполнение начальных значений
    for i, point in enumerate(points):
        dist[i][i] = 0
        for neighbor_id in point['connections']:
            j = vertex_index[neighbor_id]
            distance = math.sqrt((point['x']-points[j]['x'])**2 + (point['y']-points[j]['y'])**2)
            dist[i][j] = distance
            next_node[i][j] = j
    
    # Основной алгоритм
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next_node[i][j] = next_node[i][k]
    
    # Функция восстановления пути
    def reconstruct_path(i, j):
        if next_node[i][j] is None:
            return []
        path = [points[i]['id']]
        while i != j:
            i = next_node[i][j]
            path.append(points[i]['id'])
        return path
    
    # Собираем результаты для стартовой точки
    results = {}
    for j, point in enumerate(points):
        if j != start_idx:
            path = reconstruct_path(start_idx, j)
            if path:
                results[point['id']] = {
                    'distance': dist[start_idx][j],
                    'path': path,
                    'name': point['name']
                }
    
    return results

def print_shortest_paths(results, start_id, start_name):
    """
    Красивый вывод кратчайших путей в текстовом виде
    slow_print("\nДетали маршрута:", color="yellow")
    """
    slow_print(f"\nКратчайшие пути от {start_name} ({start_id}) к другим звёздам🌟:", color="green")
    slow_print(f"-" * 60, color="green")
    slow_print(f"{'Цель':<20} {'Расстояние':11} {'Маршрут'}", color="green")
    slow_print("-" * 60, color="green")
    
    for target_id, data in sorted(results.items(), key=lambda x: x[1]['distance']):
        path_str = " → ".join(data['path'])
        slow_print(f"{data['name']:<15} ({target_id:<1}): {data['distance']:<10.2f}  {path_str:<16}", color="green")


def reconstruct_path(came_from, current):
    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    return total_path[::-1]

def euclidean_distance(p1, p2):
    """Вычисляет евклидово расстояние между двумя точками."""
    return math.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)

def heuristic(node, goal_node):
    """Эвристическая функция для A* - евклидово расстояние до цели."""
    return euclidean_distance(node, goal_node)

def a_star(points, start_id, goal_id):   #a_star(points, 'S', 'P')
    """Реализация алгоритма A* с выводом промежуточных расстояний."""
    nodes = {point['id']: point for point in points}
    
    if start_id not in nodes or goal_id not in nodes:
        raise ValueError("Стартовая или целевая вершина не найдена")
    
    open_set = []
    heapq.heappush(open_set, (0, start_id))
    
    came_from = {}
    g_score = {point['id']: float('inf') for point in points}
    g_score[start_id] = 0
    
    f_score = {point['id']: float('inf') for point in points}
    f_score[start_id] = heuristic(nodes[start_id], nodes[goal_id])
    
    while open_set:
        _, current_id = heapq.heappop(open_set)
        
        if current_id == goal_id:
            path = [current_id]
            path_distances = []
            total_distance = 0
            
            # Восстанавливаем путь и вычисляем расстояния
            while current_id in came_from:
                next_id = came_from[current_id]
                distance = euclidean_distance(nodes[next_id], nodes[current_id])
                path_distances.append((next_id, current_id, distance))
                total_distance += distance
                current_id = next_id
                path.append(current_id)
            
            return path[::-1], path_distances[::-1], total_distance
        
        for neighbor_id in nodes[current_id]['connections']:
            tentative_g_score = g_score[current_id] + euclidean_distance(
                nodes[current_id], nodes[neighbor_id])
            
            if tentative_g_score < g_score[neighbor_id]:
                came_from[neighbor_id] = current_id
                g_score[neighbor_id] = tentative_g_score
                f_score[neighbor_id] = g_score[neighbor_id] + heuristic(
                    nodes[neighbor_id], nodes[goal_id])
                
                if neighbor_id not in [node_id for (_, node_id) in open_set]:
                    heapq.heappush(open_set, (f_score[neighbor_id], neighbor_id))
    
    return None, None, 0

def plot_path(points, path_ids):
    """Визуализация с подписанными расстояниями."""
    plt.figure(figsize=(14, 10))
    
    # Рисуем все точки
    for point in points:
        color = 'red' if point['id'] in path_ids else 'blue'
        plt.scatter(point['x'], point['y'], color=color, s=100)
        plt.text(point['x'], point['y'], 
                f"{point['id']}: {point['name']}\n({point['x']},{point['y']})",
                fontsize=8, ha='center')
    
    # Рисуем все соединения
    for point in points:
        for neighbor_id in point['connections']:
            neighbor = next(p for p in points if p['id'] == neighbor_id)
            plt.plot([point['x'], neighbor['x']], [point['y'], neighbor['y']], 
                    'gray', alpha=0.3)
    
    # Выделяем путь и подписываем расстояния
    if path_ids:
        path_points = [next(p for p in points if p['id'] == pid) for pid in path_ids]
        for i in range(len(path_points)-1):
            x1, y1 = path_points[i]['x'], path_points[i]['y']
            x2, y2 = path_points[i+1]['x'], path_points[i+1]['y']
            
            # Рисуем линию пути
            plt.plot([x1, x2], [y1, y2], 'r-', linewidth=2)
            
            # Подписываем расстояние
            distance = euclidean_distance(path_points[i], path_points[i+1])
            plt.text((x1+x2)/2, (y1+y2)/2, 
                    f"{distance:.1f}", 
                    backgroundcolor='white', fontsize=8)
    
    plt.title("Путь между звездными системами с расстояниями (A*)")
    plt.xlabel("X координата")
    plt.ylabel("Y координата")
    plt.grid(True)
    plt.show()


def print_star():
    for point in points:
        slow_print(f'🌟 ({point['id']})-{point['name']}', delay=0.01, color='blue', end=' ')
    print()

run = True
if __name__ == "__main__":
    slow_print(">>> Добро пожаловать на борт ", color='yellow', end=" ")
    slow_print("🚀 Planet Express 🚀", color='red')
    while run:
        #print_star()
        choice = slow_input(">>> Мы поможем Вам доставить груз в пределах нашей галактики\n>>> Продолжем? (Y)es/(N)o (Q)uit: ", color='green')
        if choice.lower() == "q":
            run = False
        elif choice.lower() == "y":
            print_star()
            from_star = slow_input(">>> От куда везем груз: ", color='green')
            to_star = slow_input(">>> Куда его необходимо доставить: ", color='green')
            path, distances, total = a_star(points, from_star.upper(), to_star.upper())
            if path:
                slow_print("\nДетали маршрута:", color="yellow")
                # Выводим таблицу с расстояниями
                slow_print("\n" + "═"*55, color="yellow")
                slow_print(f"{'От':<20} {'До':<20} {'Расстояние':<15}", color="yellow")
                slow_print("-"*55, color="yellow")
                for src, dst, dist in distances:
                    src_name = next(p['name'] for p in points if p['id'] == src)
                    dst_name = next(p['name'] for p in points if p['id'] == dst)
                    slow_print(f"{src_name:<14} ({src:^5}): {dst_name:<14} ({dst:^5}): {dist:7.2f}", color="yellow")
                
                slow_print("═"*55, color="yellow")
                slow_print(f"Общая длина пути: {total:>.2f}", color="yellow")
                slow_print("═"*55, color="yellow")
                choice = slow_input(">>> Показать маршрут на карте (Y)es/(N)o: ", color='green')
                if choice.lower() == "y":
                    plot_path(points, path)
            else:
                slow_print("Путь не найден", color="yellow")
        adj_matrix, vertex_index = build_weighted_adjacency_matrix(points)
        
        slow_print("Посмотреть карту смежности c расстояниями (M)ap", color="yellow")
        slow_print("Посмотреть кротчайшие пути между звёздами (W)ay", color="yellow")
        choice = slow_input(">>> Что вы хотите посмотреть: ", color='green').upper()
        if choice.lower() == "m":
            slow_print("Матрица смежности с весами (расстояниями):")
            for row in adj_matrix:
                print([f"{x:.1f}" if isinstance(x, float) else x for x in row])
        elif choice.lower() =="w":
            print_star()
            start_id = slow_input(">>> Что вы хотите посмотреть: ", color='green').upper()
            print(start_id)
            start_name = next(p['name'] for p in points if p['id'] == start_id)

            # Вычисляем кратчайшие пути
            results = floyd_warshall_from_source(points, start_id)

            # Выводим результаты
            print_shortest_paths(results, start_id, start_name)

  