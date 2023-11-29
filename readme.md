# Composite Functions Demonstrator

**This repository is work in progress!**


![demo](https://github.com/JannikIrmai/composite-function-demo/blob/master/composit-function-demo.gif)

Depicted on the top left is a compute graph for a composite function with one input node, one output node, 
two hidden nodes and a constant one node.
The hidden nodes and the output node compute a weighted sum of their parents and apply a sigmoid activation.
The coefficients of the weighted sum are the parameters of the model.

Depicted on the bottom left is some labeled data (blue and red dots) as well as the function that is defined by
the compute graph.
Depicted on the right is the learning objective in dependence of the parameters. 
To change a parameter, click on the desired value in the plot.

From the plots on the right it can be seen that the learning objective is not convex. 
In fact, it is quite difficult to find parameters that minimize the learning objective.