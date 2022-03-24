import numpy as np
import queue
import threading
import sys
import pandas as pd

NUM_WORKER = 16

def create_graph(segmentation, max_dist=200):

    centers = get_cell_centers(segmentation)
    positions = np.concatenate(centers.cell_position).reshape(-1,2)
    cell_distances = np.linalg.norm(np.array(positions) - np.array(positions[:, None]), axis=-1)

    return get_neighbors(cell_distances, centers, max_dist)


def get_neighbors(cell_distances, centers, max_dist):

    neighborhood = []
    for i in range(len(centers)):
        neighbors = [(centers.cell_id[j], centers.cell_position[j], dist) for j, dist in enumerate(cell_distances[i,:]) if dist < max_dist and dist != 0]
        neighborhood.append(CellNeighborhood(centers.cell_id[i], centers.cell_position[i], neighbors))

    return neighborhood

class CellNeighborhood():
    def __init__(self, id, position, neighbors):
        self.id = id
        self.position = position
        self.neighbors = [Neighbor(n_id, n_position, distance) for n_id, n_position, distance in neighbors] 


class Neighbor():
    def __init__(self, id, position, distance):
        self.id = id 
        self.position = position
        self.distance = distance


def get_cell_centers(segmentation):

    num_cells = segmentation.max()
    cell_centers = np.zeros_like(segmentation)

    all_centers = []
    cell_list = list(range(1, num_cells))
    CenterQueue = queue.Queue()
    for cell in cell_list:
        CenterQueue.put_nowait(cell)
    
    for _ in range(NUM_WORKER):
        Worker(
        CenterQueue,
        all_centers,
        segmentation,
        cell_list,
        num_cells).start()

    CenterQueue.join()

    return pd.DataFrame(all_centers)

class Worker(threading.Thread):
    def __init__(self, q, ret, mask, cell_list, n, *args, **kwargs):
        self.q = q
        self.ret = ret
        self.mask = mask
        self.tmp = np.zeros_like(mask)
        self.cell_list = cell_list
        self.n = n
        super().__init__(*args, **kwargs)

    def run(self):
        while True:
            try:
                cell = self.q.get_nowait()
                self.cell_list.pop()
                self.__progressBar()
                self.ret.append(self.get_center(cell))

            except queue.Empty:
                return

            self.q.task_done()

    def get_center(self, i):

        self.tmp[self.mask != i] = 0
        self.tmp[self.mask == i] = 1

        r = self.tmp.sum(axis=0)
        c = self.tmp.sum(axis=1)
        r = int(np.sum([a * b for a,b in zip(range(len(r)), r)])/np.sum(r))
        c = int(np.sum([a * b for a,b in zip(range(len(c)), c)])/np.sum(c))

        return {"cell_id": i, "cell_position": [r,c]}

    def __progressBar(self, barLength=50):
    
        percent = int((self.n - len(self.cell_list)) * 100 / self.n)
        arrow = '|' * int(percent/100 * barLength - 1) + '|'
        spaces = ' ' * (barLength - len(arrow))

        sys.stdout.write("\r" + f'Progress: |{arrow}{spaces}| {percent}% [{self.n-len(self.cell_list)}/{self.n}]' + "\r")
        sys.stdout.flush()