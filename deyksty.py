import json
import math
import sys


with open("points.json", "r") as f:
    points = json.load(f)["points"]               # Графы
with open("points.json", "r") as f:
    orders = json.load(f)["orders"]               # Параметры груза
with open("points.json", "r") as f:
    vehicles = json.load(f)["vehicles"]           # Параметры транспортного средства

def create_adjacency_matrix(points):
    """ Создание матрици смежности """
    vertex_ids = {point['id']: idx for idx, point in enumerate(points)}
    n = len(vertex_ids)
    INF = math.inf  # Бесконечность означает отсутствие связи
    # Создаём пустую матрицу n×n, заполненную INF
    adj_matrix = [[INF] * n for _ in range(n)]
    # Заполняем диагональ нулями (расстояние от вершины до себя = 0)
    for i in range(n):
        adj_matrix[i][i] = 0
    # Заполняем связи между вершинами
    for point in points:
        src_idx = vertex_ids[point['id']]
        for neighbor_id in point['connections']:
            dst_idx = vertex_ids[neighbor_id]
            # Если граф невзвешенный, ставим 1
            adj_matrix[src_idx][dst_idx] = 1
            # Если граф неориентированный, добавляем обратную связь
            adj_matrix[dst_idx][src_idx] = 1
    return adj_matrix, vertex_ids

def floyd_warshall(graph):
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

def dijkstra_naive(graph, start):
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


if __name__ == "__main__":
    adj_matrix, vertex_ids = create_adjacency_matrix(points)

    # Вывод матрицы смежности (для примера)
    print("Матрица смежности:")
    for row in adj_matrix:
        print(row)

    # Вывод соответствия id вершин и индексов
    #print("\nСоответствие id и индексов:")
    #print(vertex_ids)
    print("#" * 20)

    shortest_paths = floyd_warshall(adj_matrix)

    for row in shortest_paths:
        print(row)

    print("#" * 20)

    start_vertex = 12
    distances = dijkstra_naive(adj_matrix, start_vertex)

    print(f"Кратчайшие расстояния от вершины {start_vertex}:")
    for i, dist in enumerate(distances):
        print(f"До вершины {i}: {dist}")