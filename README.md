# Neural-Net-in-C

This project is part of my assignment for the Machine Learning course at BITS Pilani. 

We were asked to implement a neural network in C without using any external libraries. It has to be generalized to have an arbritrary number of layers, ordering of activation functions, and has to be robust to different datasets. 

This project helped me learn the intricacies and difficulties that one experiences when trying to build a neural network from scratch. 

The course required me to submit the code in one file, and hence, I have not created separate files for the functions. 

## HOW TO RUN

Run the program using the desired parameters. The order of the parameters is stated below.

Argument 0: Executable file name
Argument 1: Number of hidden layers


Example: 
~$  ./a.out 2 4,5 tanh,relu 20 3 sigmoid 0.01 data/data_train.csv 10000 10 data/data_test.csv 2000 3 200



## DATASET FORMAT

1.	The dataset should be in csv format.
2.	There should not be a header row, nor a row index column.
3.	The output variable should be the last column.
4.	The binary classification can take values from 0 or 1 only.
