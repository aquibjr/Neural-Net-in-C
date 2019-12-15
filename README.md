# Neural-Net-in-C

This project is part of my assignment for the Machine Learning course at BITS Pilani. 

We were asked to implement a neural network in C without using any external libraries. It has to be generalized to have an arbritrary number of layers, ordering of activation functions, and has to be robust to different datasets. 

This project helped me learn the intricacies and difficulties that one experiences when trying to build a neural network from scratch. 

The course required me to submit the code in one file, and hence, I have not created separate files for the functions. 

## HOW TO RUN

Run the program using the desired parameters. The order of the parameters is stated below.

Argument 0: Executable file name
Argument 1: Number of hidden layers
Argument 2: Number of nodes in each hidden layer arranged from left to right. The numbers have to be separated by comma
Argument 3: Activation function of each hidden layer arrange from left to right. It has to be separated by commas. It has to fall within the following – “identity”, ”sigmoid”, ”tanh”, “relu”
Argument 4: Number of iterations
Argument 5: Number of nodes in output
Argument 6: Activation function of output
Argument 7: Learning rate
Argument 8: Path of csv file containing the train dataset
Argument 9: Number of examples in the train dataset
Argument 10: Number of features in the dataset
Argument 11: Path of csv file containing the test dataset
Argument 12: Number of examples in the test dataset
Argument 13: Code for the type of gradient descent. ( 1 – Stochastic gradient descent | 2 – Batch gradient descent | 3 – Mini-batch gradient descent ) 
Argument 14: Size of the batch for mini-batch gradient descent. If it is one of the other 2 types of gradient descent, enter any positive number. 


Example: 
~$  ./a.out 2 4,5 tanh,relu 20 3 sigmoid 0.01 data/data_train.csv 10000 10 data/data_test.csv 2000 3 200



## DATASET FORMAT

1.	The dataset should be in csv format.
2.	There should not be a header row, nor a row index column.
3.	The output variable should be the last column.
4.	The binary classification can take values from 0 or 1 only.
