import matplotlib.pyplot as plt

from plot_composite_function import PlotCompositeFunction, CompositeFunction


def main():

    f = CompositeFunction()
    n_input = f.add_input()
    n_interior = f.add_interior_node()
    n_output = f.add_output()
    f.add_link(n_input, n_output)
    f.add_link(n_interior, n_output)

    p = PlotCompositeFunction(f)

    plt.show()


if __name__ == "__main__":
    main()
