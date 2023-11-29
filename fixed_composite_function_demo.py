import matplotlib.pyplot as plt
import numpy as np
from composite_function import CompositeFunction
from functions import sigmoid
from matplotlib.backend_bases import MouseEvent


def logistic_loss(f: CompositeFunction, x, y):
    y_pred = f.inference(x)[0]
    loss = - y * y_pred + np.log2(1 + np.exp2(y_pred))
    return loss


def l2_regularizer(f: CompositeFunction, sigma):
    reg = 0
    for p in range(f.num_params()):
        reg += f.get_param(p)**2
    return reg * np.log2(np.e)/(2*sigma**2)


def main():
    np.random.seed(seed=0)

    data = np.linspace(-10, 10, 21)
    labels = np.zeros(data.shape, dtype=int)
    labels[5:-5] = 1
    # labels = np.random.randint(0, 2, data.shape)
    sigma = 10

    f = CompositeFunction()

    nodes = [f.add_input(), f.add_constant_one(), f.add_interior_node(), f.add_interior_node(),  f.add_output()]
    names = ["in", "1", "sig", "sig", "out"]
    pos = [(0, 0), (1.5, 1), (1, 0.5), (1, -0.5), (2, 0)]

    f.set_activation(nodes[2], sigmoid)
    f.set_activation(nodes[3], sigmoid)

    links = [(0, 2), (0, 3), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4)]
    for i, j in links:
        f.add_link(nodes[i], nodes[j])

    params = [1.0, -1.0, 6.0, 6.0, -15.0, 10.0, 10.0]
    for p_id, p in enumerate(params):
        f.set_param(p_id, np.random.normal())
        # f.set_param(p_id, p)

    fig = plt.figure(figsize=(12, 8))
    ax_graph = fig.add_axes([0.05, 0.47, 0.45, 0.45])
    ax_graph.set_aspect("equal")
    ax_data = fig.add_axes([0.05, 0.07, 0.45, 0.35])
    ax_params = []
    for p in range(f.num_params()):
        frac = 1 / f.num_params()
        ax = fig.add_axes([0.55, 0.05 + p * frac * 0.9, 0.4, frac*0.8])
        if p > 0:
            ax.sharex(ax_params[0])
            ax.tick_params(labelbottom=False)
        ax_params.append(ax)

    x = np.linspace(-10, 10, 1001)
    param_vals = np.linspace(-20, 20, 1001)

    def draw_graph():
        ax_graph.set_axis_off()
        ax_graph.set_title("Compute graph")
        ax_graph.scatter(*np.array(pos).T, ec="black", fc="white", s=500)
        for i, name in enumerate(names):
            ax_graph.text(*pos[i], name, horizontalalignment='center', verticalalignment='center')
        for link in links:
            ax_graph.annotate("", xy=pos[link[1]], xytext=pos[link[0]], arrowprops=dict(facecolor='black', shrink=0.1))

    def make_plot():
        ax_data.clear()
        ax_data.set_xlabel("x")
        ax_data.set_ylabel("y")
        y = f.inference(x)[0]
        probs = 1 / (1 + np.exp2(-y))
        ax_data.plot(x, probs)
        ax_data.scatter(data, labels, color=["tab:blue" if l == 0 else "tab:red" for l in labels])

        current_loss = np.sum(logistic_loss(f, data, labels))
        current_regularizer = l2_regularizer(f, sigma)
        current_obj = current_loss + current_regularizer
        ax_data.set_title(f"Loss = {current_loss:.3f}, Reg = {current_regularizer:.3f}, Obj = {current_obj:.3f}")

        for p in range(f.num_params()):
            ax_params[p].clear()
            ax_params[p].parent_id = p
            val = f.get_param(p)
            f.set_param(p, param_vals)
            loss = 0
            for d, l in zip(data, labels):
                loss += logistic_loss(f, d, l)
            reg = l2_regularizer(f, sigma)
            obj = loss + reg
            ax_params[p].plot(param_vals, obj, color="tab:blue")
            f.set_param(p, val)
            ax_params[p].scatter([val], [current_obj], fc="tab:blue", ec="black")

    def on_click(event: MouseEvent):
        try:
            parent_id = event.inaxes.parent_id
        except AttributeError:
            return
        f.set_param(parent_id, event.xdata)
        make_plot()
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)
    draw_graph()
    make_plot()
    plt.show()


if __name__ == "__main__":
    main()


