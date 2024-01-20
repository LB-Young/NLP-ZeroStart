#实现最短路径计算floyd算法
def startwith(start, graph):
    allways = []
    while start < len(graph):
        visited = [start]
        unvisited = [x for x in range(len(graph)) if x != start]
        distance = graph[start]

        while len(unvisited):
            idx = unvisited[0]
            for i in unvisited:
                if distance[i] < distance[idx]:
                    idx = i

            unvisited.remove(idx)
            visited.append(idx)

            for i in unvisited:
                if distance[idx] + graph[idx][i] < distance[i]:
                    distance[i] = distance[idx] + graph[idx][i]
        allways.append(distance)
        start += 1
    return allways



if __name__ == "__main__":
    inf = 9999
    graph = [[0, 1, 12, inf, inf, inf],
              [inf, 0, 9, 3, inf, inf],
              [inf, inf, 0, inf, 5, inf],
              [inf, inf, 4, 0, 13, 15],
              [inf, inf, inf, inf, 0, 4],
              [inf, inf, inf, inf, inf, 0]]

    allways = startwith(0, graph)
    print(allways)