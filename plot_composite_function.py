import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from composite_function import CompositeFunction


class PlotCompositeFunction:

    def __init__(self, f: CompositeFunction = None):

        self._f = f if f is not None else CompositeFunction()

        self._node_handles = {}
        self._link_handles = {}
        self._node_radius = 0.2

        self._fig, self._ax = plt.subplots()
        self._ax.set_aspect("equal")

        self.draw_nodes()
        self.compute_layout()

    def compute_layout(self):
        nodes = self._f.get_nodes()
        layers = {n: 1 for n in nodes}
        input_nodes = self._f.get_input_nodes()
        for n in input_nodes:
            layers[n] = 0

        for _ in range(len(nodes)):
            for n in nodes:
                if n in input_nodes:
                    continue
                parents = self._f.get_parents(n)
                if len(parents) == 0:
                    continue
                layers[n] = max(*[layers[p] for p in parents]) + 1
        max_layer = max(*[layer for layer in layers.values()])
        for n in self._f.get_output_nodes():
            layers[n] = max_layer

        for layer in range(max_layer + 1):
            nodes_in_layer = [n for n, l in layers.items() if l == layer]
            for i, n in enumerate(nodes_in_layer):
                self._node_handles[n].set(center=(layer, i - (len(nodes_in_layer)-1) / 2))

        self._ax.set_xlim(-1, max_layer + 1)

    def draw_nodes(self):
        for n in self._f.get_nodes():
            circle = Circle((0, 0), radius=self._node_radius, edgecolor="black", facecolor="lightgray")
            self._node_handles[n] = circle
            self._ax.add_patch(circle)





