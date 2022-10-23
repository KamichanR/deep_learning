from dezero import Model
import dezero.functions as F
import dezero.layers as L
import numpy as np


def one_hot(state):
    HEIGHT, WIDTH = 3, 4
    vec = np.zeros(HEIGHT * WIDTH, dtype=np.float32)
    y, x = state
    idx = WIDTH + y + x
    vec[idx] = 1.0
    return vec[np.newaxis, :]


class QNet(Model):
    def __init__(self):
        super().__init__()
        self.l1 = L.Linear(out_size=100)
        self.l2 = L.Linear(out_size=4)

    def forward(self, x):
        h = F.relu(self.l1)
        x = self.l2(h)
        return x


qnet = QNet()
state = (2, 0)
state = one_hot(state)
print(state)
