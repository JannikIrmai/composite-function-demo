import functions


class CompositeFunction:

    def __init__(self, inputs: int = 0, outputs: int = 0):

        self._outputs = list(range(outputs))
        self._inputs = list(range(outputs, outputs + inputs))
        self._idx = inputs + outputs
        self._parents = {o: [] for o in self._outputs}
        self._params = {}
        self._param_idx = 0
        self._functions = {o: functions.MLP(self._params) for o in self._outputs}

    def inference(self, *args):
        assert len(args) == len(self._inputs)

        def recursion(n):
            if n in self._inputs:
                i = self._inputs.index(n)
                return args[i]
            return self._functions[n].forward(*[recursion(p) for p in self._parents[n]])

        return [recursion(o) for o in self._outputs]

    def add_input(self):
        self._inputs.append(self._idx)
        self._idx += 1
        return self._idx - 1

    def add_output(self):
        self._outputs.append(self._idx)
        self._parents[self._idx] = []
        self._functions[self._idx] = functions.MLP(self._params)
        self._idx += 1
        return self._idx - 1

    def add_interior_node(self):
        self._parents[self._idx] = []
        self._functions[self._idx] = functions.MLP(self._params)
        self._idx += 1
        return self._idx - 1

    def add_constant_one(self):
        self._parents[self._idx] = []
        self._functions[self._idx] = functions.ConstantOne(self._params)
        self._idx += 1
        return self._idx - 1

    def add_link(self, u, v, param_idx=None):
        if param_idx is None:
            # add a  new parameter to the model
            param_idx = self._param_idx
            self._params[param_idx] = 1
            self._param_idx += 1

        # add link to compute graph
        self._parents[v].append(u)
        self._functions[v].add_param(param_idx)
        return param_idx

    def set_activation(self, u, activation):
        self._functions[u].set_activation(activation)

    def get_nodes(self):
        return self._inputs.copy() + list(self._parents.keys())

    def get_parents(self, n):
        return self._parents[n].copy()

    def get_output_nodes(self):
        return self._outputs.copy()

    def get_input_nodes(self):
        return self._inputs.copy()

    def get_links(self):
        links = []
        for u, parents in self._parents.items():
            for p in parents:
                links.append((p, u))
        return links

    def num_params(self):
        return len(self._params)

    def set_param(self, param_id, param_val):
        self._params[param_id] = param_val

    def get_param(self, param_id):
        return self._params[param_id]












