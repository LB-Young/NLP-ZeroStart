# 实现最短路径计算dijkstra算法
def shortestways(start, graph):
    visited = [start]
    unvisited = [x for x in range(len(graph)) if x != start]
    distance = graph[start]

    while len(unvisited):
        index = unvisited[0]
        for i in unvisited:
            if distance[i] < distance[index]:
                index = i

        visited.append(index)
        unvisited.remove(index)
        
        for i in range(len(distance)):
            if distance[index] + graph[index][i] < distance[i]:
                distance[i] = distance[index] + graph[index][i]
    return distance


if __name__ == "__main__":
    start = 0
    inf = 9999
    graph = [[0, 1, 12, inf, inf, inf],
             [inf, 0, 9, 3, inf, inf],
             [inf, inf, 0, inf, 5, inf],
             [inf, inf, 4, 0, 13, 15],
             [inf, inf, inf, inf, 0, 4],
             [inf, inf, inf, inf, inf, 0]]

    distance = shortestways(start, graph)
    print(distance)
