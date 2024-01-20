import copy


class Fence:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def score(self, node1, node2):
        # if node1 == "start":
        #     return (node2[0] + node2[1] + 1) / (node2[0] * node2[1] + 1)
        # assert node1[1] + 1 == node2[1]  # 保证两个节点处于相邻列
        # mod = (node1[0] + node1[1] + node2[0] + node2[1]) % 3 + 1
        # mod /= node1[0] * 4 + node1[1] * 3 + node2[0] * 2 + node2[1] * 1
        # return mod

        if node1 == "start":
            # print("+++" * 5)
            return (node2[0] + node2[1] + 1) / self.height
        else:
            assert node1[1] + 1 == node2[1]
            # print("---" * 5)
            return ((node1[0] + node1[1] + node2[0] + node2[1]) % 3 + 1) / \
                   (node1[0] * 4 + node1[1] * 3 + node2[0] * 2 + node2[1] * 1)


class Path:
    def __init__(self):
        self.nodes = ["start"]
        self.score = 0

    def __len__(self):
        return len(self.nodes)


def viterbi1(fence):
    width = fence.width
    height = fence.height
    starter = Path()
    beam_buffer = [starter]
    new_beam_buffer = []
    while True:
        for h in range(height):
            new_col = len(beam_buffer[0]) - 1
            new_node = [h, new_col]
            node_path = []
            for path in beam_buffer:
                new_path = copy.deepcopy(path)
                node_score = fence.score(path.nodes[-1], new_node)
                new_path.score += node_score
                new_path.nodes.append(new_node)
                node_path.append(new_path)
            max_score_node_path = sorted(node_path, key=lambda x: x.score)
            # max_score_node_path = node_path.sort(key=lambda x: x.score)
            new_beam_buffer.append(max_score_node_path[0])
        beam_buffer = new_beam_buffer
        new_beam_buffer = []
        if len(beam_buffer[0]) == width + 1:
            break
    return sorted(beam_buffer, key=lambda x: x.score)
    # return beam_buffer.sort(key=lambda x: x.score)


def beam_search(fence, beam_size):
    width = fence.width
    height = fence.height
    starter = Path()
    beam_buffer = [starter]
    new_beam_buffer = []
    while True:
        for h in range(height):
            new_col = len(beam_buffer[0]) - 1
            new_node = [h, new_col]
            for path in beam_buffer:
                new_path = copy.deepcopy(path)
                new_path.score = fence.score(path.nodes[-1], new_node)
                new_path.nodes.append(new_node)
                new_beam_buffer.append(new_path)
        new_beam_buffer = sorted(new_beam_buffer, key=lambda x: x.score)
        beam_buffer = new_beam_buffer[:beam_size]
        new_beam_buffer = []
        if len(beam_buffer[0]) == width + 1:
            break
    return sorted(beam_buffer, key=lambda x: x.score)


def main():
    fence = Fence(6, 4)
    res = viterbi1(fence)
    for item in res:
        print(item.nodes, item.score)
    print("+" * 20)
    beam_size = 2
    res = beam_search(fence, beam_size)
    for item in res:
        print(item.nodes, item.score)


if __name__ == "__main__":
    main()
