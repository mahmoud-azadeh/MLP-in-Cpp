### MLP in C++

This is a multi-layer neural network implemented from scratch in a C++ program.
The dataset is the CFAR10 binary files, which are available at here (https://www.cs.toronto.edu/~kriz/cifar.html).

-----
To compile and run the C++ code in Linux:
```bash
$ g++ -O2 mlp.cpp -o mlp
$ ./mlp
```
-----
The implementation involves pre-processing the dataset, adding a batch normalization layer, incorporating a dropout layer, and utilizing Adam optimization.

-----
reference: http://cs231n.stanford.edu/index.html