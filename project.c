#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define MAX_LINE_SIZE 1048576 // 2^20
#define max(x, y) (x > y? x : y)

typedef struct{
	int no_hid;
	int* hid_size;
	int* hid_act;
	int no_iter;
	int out_size;
	int out_act;
	int grad_type;
	int mini_size;
	double learn_rate;
	double** data_train;
	double** data_test;
	double*** weight;
	int no_cols;
	int no_rows_train;
	int no_rows_test;
} parameters;

void read_csv(char* file, int rows, int cols, double** data);
void init_weights(parameters* param, int no_layers, int* layer_size);
void rand_shuffle(int* a, int n);
void mlp_train(parameters* param, int* layer_size);
void mat_mul(double* a, double** b, double* out, int n, int p);
void identity(int n, double* input, double* output);
void sigmoid(int n, double* input, double* output);
void tan_h(int n, double* input, double* output);
void relu(int n, double* input, double* output);
void forward_prop(parameters* param, int train_ex, int no_layers, int* layer_size, double** inputs, double** outputs);
void d_identity(int layer_size, double* input, double* output, double* deriv);
void d_sigmoid(int layer_size, double* input, double* output, double* deriv);
void d_tanh(int layer_size, double* input, double* output, double* deriv);
void d_relu(int layer_size, double* input, double* output, double* deriv);
void calc_local_grad(parameters* param, int layer_no, int no_layers, int* layer_size, double** inputs, double** outputs, double* exp_output, double** local_grad);
void back_prop(parameters* param, int train_ex, int no_layers, int* layer_size, double** inputs, double** outputs, double*** weight_correction);
void mlp_classify(parameters* param, int* layer_size);




int main(int argc, char** argv){
	
	if(argc != 15){
	   printf("Please enter the correct number of variables in the correct format");
	   exit(0);
	}

    parameters* param = (parameters*)malloc(sizeof(parameters));
    
    //Assigning all the values from argv to param.

    param->no_hid = atoi(argv[1]);
    if(param->no_hid < 0){
       printf("No. of hidden layers must be greater than or equal to 0");
       exit(0);
    }

    param->hid_size = (int*)malloc(param->no_hid * sizeof(int));
    int i;
    char* tok;
    for(i =0, tok = strtok(argv[2], ","); i < param->no_hid; i++){
       param->hid_size[i] = atoi(tok);
       if(param->hid_size[i] <= 0){
          printf("Hidden layer sizes should be positive");
          exit(0);
       }
       tok = strtok(NULL, ",");
    }

    param->hid_act = (int*)malloc(param->no_hid * sizeof(int));
    for (i = 0, tok = strtok(argv[3], ","); i < param->no_hid; i++) {
        if (strcmp(tok, "identity") == 0) {
            param->hid_act[i] = 1;
        }
        else if (strcmp(tok, "sigmoid") == 0) {
            param->hid_act[i] = 2;
        }
        else if (strcmp(tok, "tanh") == 0) {
            param->hid_act[i] = 3;
        }
        else if (strcmp(tok, "relu") == 0) {
            param->hid_act[i] = 4;
        }
        else {
            printf("Invalid value for hidden activation function\n");
            printf("Input identity, sigmoid, tanh or relu for hidden activation function\n");
            exit(0);
        }
    
        tok = strtok(NULL, ",");

    }

    param->no_iter = atoi(argv[4]);
    if(param->no_iter <=0){
       printf("Number of iterations must be positive");
       exit(0);
    }

    param->out_size = atoi(argv[5]);
    if(param->out_size <= 0){
       printf("Size of output layer must be positive");
       exit(0);
    }

    if (strcmp(argv[6], "identity") == 0) {
        param->out_act = 1;
    }
    else if (strcmp(argv[6], "sigmoid") == 0) {
        param->out_act = 2;
    }
    else if (strcmp(argv[6], "tanh") == 0) {
        param->out_act = 3;
    }
    else if (strcmp(argv[6], "relu") == 0) {
        param->out_act = 4;
    }
    else {
        printf("Invalid value for output activation function\n");
        printf("Input either identity, sigmoid, tanh or relu for output activation function\n");
        exit(0);
    }

    param->learn_rate = atof(argv[7]);

    char* train_filename = argv[8];
    param->no_rows_train = atoi(argv[9]);
    param->no_cols = atoi(argv[10]);

    param->data_train = (double**)malloc(param->no_rows_train * sizeof(double*));
    for(i =0; i<param->no_rows_train; i++)
       param->data_train[i] = (double*)malloc(param->no_cols * sizeof(double));

    char* test_filename = argv[11];
    param->no_rows_test = atoi(argv[12]);

    param->grad_type = atoi(argv[13]);
    if(param->grad_type != 1 && param->grad_type != 2 && param->grad_type != 3 ){
       printf("Enter valid gradient descent type code, that is, 1, 2 or 3.");
       exit(0);
    }
    param->mini_size = atoi(argv[14]);
    if(param->mini_size <= 0){
       printf("Number of training examples in mini batch must be positive");
       exit(0);
    }

    param->data_test = (double**)malloc(param->no_rows_test * sizeof(double*));
    for(i =0; i<param->no_rows_test; i++)
       param->data_test[i] = (double*)malloc(param->no_cols * sizeof(double));

    read_csv(train_filename, param->no_rows_train, param->no_cols, param->data_train);
    read_csv(test_filename, param->no_rows_test, param->no_cols, param->data_test);

    int no_layers = param->no_hid + 2;

    int* layer_size = (int*)calloc(no_layers, sizeof(int));

    layer_size[0] = param->no_cols - 1;
    layer_size[no_layers-1] = param->out_size;

    for (i = 1; i < no_layers-1 ; i++)
        layer_size[i] = param->hid_size[i-1];

    param->weight = (double***)calloc(no_layers - 1, sizeof(double**));

    for (i = 0; i < no_layers-1; i++)
        param->weight[i] = (double**)calloc(layer_size[i]+1, sizeof(double*));

    int j;
    for (i = 0; i < no_layers-1; i++)
        for (j = 0; j < layer_size[i]+1; j++)
            param->weight[i][j] = (double*)calloc(layer_size[i+1], sizeof(double));



    printf("Training...\n");
    mlp_train(param, layer_size);
    printf("\nDone\n\n");

    printf("Classifying...\n");
    mlp_classify(param, layer_size);
    printf("Done.\n\n");

    //Free the memory

    for (i = 0; i < no_layers-1; i++)
        for (j = 0; j < layer_size[i]+1; j++)
            free(param->weight[i][j]);

    for (i = 0; i < no_layers-1; i++)
        free(param->weight[i]);

    free(param->weight);

    free(layer_size);

    for (i = 0; i < param->no_rows_train; i++)
        free(param->data_train[i]);

    for (i = 0; i < param->no_rows_test; i++)
        free(param->data_test[i]);

    free(param->data_train);
    free(param->data_test);
    free(param->hid_act);
    free(param->hid_size);
    free(param);

    return 0;
}

//Function to read a csv file nad store it.
void read_csv(char* file, int rows, int cols, double** data){
	FILE* fl = fopen(file, "r");
	if(NULL == fl){
	   printf("Error opening file. Enter path correctly.");
	   exit(0);
	}

    char* ln = (char*)malloc(MAX_LINE_SIZE * sizeof(char));

    for(int i = 0; fgets(ln, MAX_LINE_SIZE, fl) && i < rows; i++){
       char* tok = strtok(ln, ",");
       for(int j = 0; tok && *tok; j++){
          data[i][j] = atof(tok);
          tok = strtok(NULL, ",\n");
       }
    }    
    
    free(ln);
    fclose(fl);

}


//Function to initialize weights
void init_weights(parameters* param, int no_layers, int* layer_size){
	
	srand(time(0));

	double* eps = (double*)calloc(no_layers - 1, sizeof(double));
	for(int i = 0; i < no_layers - 1; i++)
	   eps[i] = sqrt(5.0 / (layer_size[i] + layer_size[i + 1]));

	for(int i = 0; i < no_layers - 1; i++)
	   for(int j = 0; j < layer_size[i] + 1; j++)
	      for(int k = 0; k < layer_size[i+1]; k++)
	         param->weight[i][j][k] = -eps[i] + ((double)rand() / ((double)RAND_MAX / (2.0 * eps[i])));

	free(eps);

}

//Function to generate the examples in the training set.
void rand_shuffle(int* a, int n){
	srand(time(NULL));
	int j;
	for(int i = n-1; i>0; i--){
	   j = rand() % (i+1);
	   int temp = a[i];
	   a[i] = a[j];
	   a[j] = temp;
	}
}

//Function to train the neural network.
void mlp_train(parameters* param, int* layer_size){
    
    int no_layers = param->no_hid + 2;

    double** inputs = (double**)calloc(no_layers, sizeof(double*));

    for (int i = 0; i < no_layers; i++)
        inputs[i] = (double*)calloc(layer_size[i], sizeof(double));

    double** outputs = (double**)calloc(no_layers, sizeof(double*));

    for (int i = 0; i < no_layers; i++)
        outputs[i] = (double*)calloc(layer_size[i]+1, sizeof(double));

    init_weights(param, no_layers, layer_size);

    int* indices = (int*)calloc(param->no_rows_train, sizeof(int));
    for (int i = 0; i < param->no_rows_train; i++)
        indices[i] = i;

    double*** weight_correction = (double***)calloc(no_layers-1, sizeof(double**));

    int i;
    for (i = 0; i < no_layers-1; i++)
        weight_correction[i] = (double**)calloc(layer_size[i]+1, sizeof(double*));

    int j;
    for (i = 0; i < no_layers-1; i++)
        for (j = 0; j < layer_size[i]+1; j++)
            weight_correction[i][j] = (double*)calloc(layer_size[i+1], sizeof(double));

    int train_ex, p, q, r, k;

    for(p = 0; p < no_layers - 1; p++)
        for(q = 0; q < layer_size[p] + 1; q++)
            for(r = 0; r < layer_size[p + 1]; r++)
                weight_correction[p][q][r] = 0.0;
                             
    for (i = 0; i < param->no_iter; i++) {
        printf("\nIteration %d \r", i+1);
        rand_shuffle(indices, param->no_rows_train);

        switch(param->grad_type){
            //Stochastic Gradient Descent
            case 1: for (j = 0; j < param->no_rows_train; j++) {
                       train_ex = indices[j];
            
                       forward_prop(param, train_ex, no_layers, layer_size, inputs, outputs);

                       back_prop(param, train_ex, no_layers, layer_size, inputs, outputs, weight_correction);

                       for(p = 0; p < no_layers - 1; p++)
                          for(q = 0; q < layer_size[p] + 1; q++)
                             for(r = 0; r < layer_size[p + 1]; r++){
                                param->weight[p][q][r] -= weight_correction[p][q][r];
                                weight_correction[p][q][r] = 0.0;
                             }

                    } 
                    break;
            //Batch gradient descent
            case 2: for(j = 0; j < param->no_rows_train; j++){
                       train_ex = indices[j];

                       forward_prop(param, train_ex, no_layers, layer_size, inputs, outputs);

                       back_prop(param, train_ex, no_layers, layer_size, inputs, outputs, weight_correction);

                    }
                    
                    for(p = 0; p < no_layers - 1; p++)
                       for(q = 0; q < layer_size[p] + 1; q++)
                          for(r = 0; r < layer_size[p + 1]; r++){
                             param->weight[p][q][r] -= weight_correction[p][q][r];
                             weight_correction[p][q][r] = 0.0;
                          }

                    break;
            //Mini-batch gradient descent
            case 3: j = 0;
                    while(j + param->mini_size <= param->no_rows_train){
                       
                       for(k = j; k < j + param->mini_size; k++){
                          train_ex = indices[k];

                          forward_prop(param, train_ex, no_layers, layer_size, inputs, outputs);

                          back_prop(param, train_ex, no_layers, layer_size, inputs, outputs, weight_correction);
                       }

                       for(p = 0; p < no_layers - 1; p++)
                          for(q = 0; q < layer_size[p] + 1; q++)
                             for(r = 0; r < layer_size[p + 1]; r++){
                                param->weight[p][q][r] -= weight_correction[p][q][r];
                                weight_correction[p][q][r] = 0.0;
                             }
                        j += param->mini_size;
                    }

                    for(k = j; param->no_rows_train; k++){
                        train_ex = indices[k];

                        forward_prop(param, train_ex, no_layers, layer_size, inputs, outputs);

                        back_prop(param, train_ex, no_layers, layer_size, inputs, outputs, weight_correction);
                    }

                    for(p = 0; p < no_layers - 1; p++)
                       for(q = 0; q < layer_size[p] + 1; q++)
                          for(r = 0; r < layer_size[p + 1]; r++){
                             param->weight[p][q][r] -= weight_correction[p][q][r];
                             weight_correction[p][q][r] = 0.0;
                          }

                    break;
        
        }
        
    }
    //Free memory.
    free(indices);

    for(i = 0; i < no_layers-1; i++)
       for(j = 0; j < layer_size[i]+1; j++)
          free(weight_correction[i][j]);

    for (i = 0; i < no_layers-1; i++)
       free(weight_correction[i]);

    free(weight_correction);

    for (i = 0; i < no_layers; i++)
        free(outputs[i]);

    free(outputs);

    for (i = 0; i < no_layers; i++)
        free(inputs[i]);

    free(inputs);
}
//Function to multiply two matrices of sizes 1xn and nxp
void mat_mul(double* a, double** b, double* out, int n, int p){
	int i, j;
	for(i = 0; i < p; i++){
	   out[i] = 0.0;
	   for(j = 0; j < n; j++)
          out[i] += (a[j] * b[j][i]);
	}
}

void identity(int n, double* input, double* output){
	output[0] = 1;

	int i;
	for(i = 0; i < n; i++)
	   output[i + 1] = input[i];
}

void sigmoid(int n, double* input, double* output){
	output[0] = 1;
    
    int i;
    for(i = 0; i < n; i++)
       output[i + 1] = 1.0 / (1.0 / exp(-input[i]));
}

void tan_h(int n, double* input, double* output){
	output[0] = 1;

	int i;
	for(i = 0; i < n; i++)
	   output[i + 1] = tanh(input[i]);
}

void relu(int n, double* input, double* output){
	output[0] = 1;

	int i;
    for(i = 0; i < n; i++)
       output[i+1] = max(0.0, input[i]);
}
//Function to execute forward propogation
void forward_prop(parameters* param, int train_ex, int no_layers, int* layer_size, double** inputs, double** outputs){
	int i;
	outputs[0][0] = 1;
	for(i = 0; i < param->no_cols - 1; i++){
	   outputs[0][i+1] = inputs[0][i] = param->data_train[train_ex][i];
	}


	for(i = 1; i < no_layers - 1; i++){
	   mat_mul(outputs[i-1], param->weight[i-1],inputs[i],layer_size[i-1] + 1,layer_size[i]);

	   switch(param->hid_act[i-1]){

	      case 1: 
	         identity(layer_size[i], inputs[i], outputs[i]);
	         break;
	      case 2:
	         sigmoid(layer_size[i], inputs[i], outputs[i]);
	         break;
	      case 3:
	         tan_h(layer_size[i], inputs[i], outputs[i]);
	         break;
	      case 4:
	         relu(layer_size[i], inputs[i], outputs[i]);
	         break;
	      default:
	         printf("Invalid function for activation");
	         exit(0);
	         break;
	   }
	}

	mat_mul(outputs[no_layers - 2], param->weight[no_layers - 2], inputs[no_layers - 1], layer_size[no_layers - 2] + 1, layer_size[no_layers - 1]);

	switch(param->out_act){

	    case 1: 
	        identity(layer_size[no_layers - 1], inputs[no_layers - 1], outputs[no_layers - 1]);
	        break;
	    case 2:
	        sigmoid(layer_size[no_layers - 1], inputs[no_layers - 1], outputs[no_layers - 1]);
	        break;
	    case 3:
	        tan_h(layer_size[no_layers - 1], inputs[no_layers - 1], outputs[no_layers - 1]);
	        break;
	    case 4:
	        relu(layer_size[no_layers - 1], inputs[no_layers - 1], outputs[no_layers - 1]);
	        break;
	    default:
	        printf("Invalid function for activation");
	        exit(0);
	        break;
	   
	}	

}

void d_identity(int layer_size, double* input, double* output, double* deriv){
	
	int i;
    for(i = 0; i < layer_size; i++)
       deriv[i] = 1;

}

void d_sigmoid(int layer_size, double* input, double* output, double* deriv){
	
	int i;
    for(i = 0; i < layer_size; i++ )
       deriv[i] = output[i + 1] * (1.0 - output[i + 1]);
    
}

void d_tanh(int layer_size, double* input, double* output, double* deriv){
	
	int i;
    for(i = 0; i < layer_size; i++ )
       deriv[i] = 1.0 - output[i + 1] * output[i + 1];

}

void d_relu(int layer_size, double* input, double* output, double* deriv){
	
	int i;
    for(i = 0; i < layer_size; i++ ){
       if(input[i] > 0)
          deriv[i] = 1;
       else if(input[i] < 0)
          deriv[i] = 0;
       else
          deriv[i] = 0.5;
    }

}
//Function to calculate the local gradient
void calc_local_grad(parameters* param, int layer_no, int no_layers, int* layer_size, double** inputs, double** outputs, double* exp_output, double** local_grad){
	
	double** layer_derivs = (double**)calloc(no_layers, sizeof(double*));

	int i;
	for(i = 0; i < no_layers; i++)
	   layer_derivs[i] = (double*)calloc(layer_size[i], sizeof(double));

	if(layer_no == no_layers - 1){
	   
	   double* error_output = (double*)calloc(param->out_size, sizeof(double));

	   for(i = 0; i < param->out_size; i++){
	      error_output[i] = exp_output[i] - outputs[layer_no][i + 1];
	   }

	   switch(param->out_act){
	      
	      case 1: d_identity(param->out_size, inputs[layer_no], outputs[layer_no], layer_derivs[layer_no]);
	              for(i = 0; i < param->out_size; i++)
	                 local_grad[layer_no][i] = error_output[i] * layer_derivs[layer_no][i];
	              break;

	      case 2: d_sigmoid(param->out_size, inputs[layer_no], outputs[layer_no], layer_derivs[layer_no]);
	              for(i = 0; i < param->out_size; i++){
	                 local_grad[layer_no][i] = error_output[i] * layer_derivs[layer_no][i];
	              }

	              break;

	      case 3: d_tanh(param->out_size, inputs[layer_no], outputs[layer_no], layer_derivs[layer_no]);
	              for(i = 0; i < param->out_size; i++)
	                 local_grad[layer_no][i] = error_output[i] * layer_derivs[layer_no][i];
	              break;

	      case 4: d_relu(param->out_size, inputs[layer_no], outputs[layer_no], layer_derivs[layer_no]);
	              for(i = 0; i < param->out_size; i++)
	                 local_grad[layer_no][i] = error_output[i] * layer_derivs[layer_no][i];
	              break;

	       default: printf("Error in calculating local gradient. Invalid output activation function\n");
	                exit(0);
	                break;


	   }

	   free(error_output);

       
	}

	else{

       int j;
       switch(param->hid_act[layer_no - 1]){
          case 1: d_identity(layer_size[layer_no], inputs[layer_no], outputs[layer_no], layer_derivs[layer_no]);
                  for(i = 0; i < layer_size[layer_no]; i++){
                     double error = 0.0;
                     for(j = 0; j < layer_size[layer_no + 1];j++)
                        error+=local_grad[layer_no+1][j] * param->weight[layer_no][i][j];
                     local_grad[layer_no][i] = error * layer_derivs[layer_no][i];
                  }   

                  break;
          case 2: d_sigmoid(layer_size[layer_no], inputs[layer_no], outputs[layer_no], layer_derivs[layer_no]);
                  for(i = 0; i < layer_size[layer_no]; i++){
                     double error = 0.0;
                     for(j = 0; j < layer_size[layer_no + 1];j++)
                        error+=local_grad[layer_no+1][j] * param->weight[layer_no][i][j];
                     local_grad[layer_no][i] = error * layer_derivs[layer_no][i];
                  }   

                  break;
          case 3: d_tanh(layer_size[layer_no], inputs[layer_no], outputs[layer_no], layer_derivs[layer_no]);
                  for(i = 0; i < layer_size[layer_no]; i++){
                     double error = 0.0;
                     for(j = 0; j < layer_size[layer_no + 1];j++)
                        error+=local_grad[layer_no+1][j] * param->weight[layer_no][i][j];
                     local_grad[layer_no][i] = error * layer_derivs[layer_no][i];
                  }   

                  break;
          case 4: d_relu(layer_size[layer_no], inputs[layer_no], outputs[layer_no], layer_derivs[layer_no]);
                  for(i = 0; i < layer_size[layer_no]; i++){
                     double error = 0.0;
                     for(j = 0; j < layer_size[layer_no + 1];j++)
                        error+=local_grad[layer_no+1][j] * param->weight[layer_no][i][j];
                     local_grad[layer_no][i] = error * layer_derivs[layer_no][i];
                  }   

                  break;  
           default: printf("Invalid hidden activation function\n");
                    exit(0);
                    break;                
       }    
                 

	}

	for(i = 0; i < no_layers; i++)
	   free(layer_derivs[i]);

	free(layer_derivs);
}

//Function to execute back propogation
void back_prop(parameters* param, int train_ex, int no_layers, int* layer_size, double** inputs, double** outputs, double*** weight_correction){
   
   double* exp_output = (double*)calloc(param->out_size, sizeof(double));
   if (param->out_size == 1)
      exp_output[0] = param->data_train[train_ex][param->no_cols-1];
   else
      exp_output[(int)(param->data_train[train_ex][param->no_cols-1] - 1 )] = 1;

   double** local_grad = (double**)calloc(no_layers, sizeof(double*));

   for(int i = 0; i < no_layers; i++)
      local_grad[i] = (double*)calloc(layer_size[i],sizeof(double));

   calc_local_grad(param, no_layers - 1, no_layers, layer_size, inputs, outputs, exp_output, local_grad);

   for(int i = 0; i < param->out_size; i++)
      for(int j = 0; j < layer_size[no_layers - 2] + 1; j++)
         weight_correction[no_layers - 2][j][i] += (param->learn_rate) * local_grad[no_layers - 1][i] * outputs[no_layers - 2][j];

   int k;
   for(int i = no_layers - 2; i >= 1; i--){
      calc_local_grad(param, i, no_layers, layer_size, inputs, outputs, exp_output, local_grad);

      for(int j = 0; j < layer_size[i]; j++)
         for(k = 0; k < layer_size[i - 1] + 1; k++){	
            weight_correction[i-1][k][j] += (param->learn_rate) * local_grad[i][j] * outputs[i-1][k];
         }
 
   }

   for(int i = 0; i < no_layers; i++)
      free(local_grad[i]);

   free(local_grad);

   free(exp_output);
	
}
//Function to classify the test dataset and print the accuracy scores
void mlp_classify(parameters* param, int* layer_size){
	int no_layers = param->no_hid + 2;
	double** inputs = (double**)calloc(no_layers, sizeof(double*));

	int i;
	for(i = 0; i < no_layers; i++)
	   inputs[i] = (double*)calloc(layer_size[i], sizeof(double));
    
    double** outputs = (double**)calloc(no_layers, sizeof(double*));

	for(i = 0; i < no_layers; i++)
	   outputs[i] = (double*)calloc(layer_size[i]+1, sizeof(double));

	double** final_output = (double**)calloc(param->no_rows_test, sizeof(double*));

	for(i = 0; i < param->no_rows_test; i++)
	   final_output[i] = (double*)calloc(param->out_size, sizeof(double));

	int test_sample;
	for(test_sample = 0; test_sample < param->no_rows_test; test_sample++){
	   printf("\nClassifying Test Sample %d\r", test_sample + 1);
       outputs[0][0] = 1.0;
       for(i = 0; i < param->no_cols - 1; i++){
          outputs[0][i+1] = (inputs[0][i] = param->data_test[test_sample][i]);
       }
       for(i = 1; i < no_layers - 1; i++){
          mat_mul(outputs[i-1], param->weight[i-1], inputs[i], layer_size[i-1]+1, layer_size[i]);

          switch(param->hid_act[i-1]){
             case 1: identity(layer_size[i], inputs[i], outputs[i]);
                     break;
             case 2: sigmoid(layer_size[i], inputs[i], outputs[i]);
                     break;
             case 3: tan_h(layer_size[i], inputs[i], outputs[i]);
                     break;
             case 4: relu(layer_size[i], inputs[i], outputs[i]);
                     break;    
             default: printf("Invalid activation function during forward prop\n");
                      exit(0);
                      break;

          }
       }


       mat_mul(outputs[no_layers-2], param->weight[no_layers-2], inputs[no_layers-1], layer_size[no_layers-2]+1, layer_size[no_layers-1]);

       switch(param->out_act){
          case 1: identity(layer_size[no_layers-1], inputs[no_layers-1], outputs[no_layers-1]);
                  break;
          case 2: sigmoid(layer_size[no_layers-1], inputs[no_layers-1], outputs[no_layers-1]);
                  break;
          case 3: tan_h(layer_size[no_layers-1], inputs[no_layers-1], outputs[no_layers-1]);
                  break;
          case 4: relu(layer_size[no_layers-1], inputs[no_layers-1], outputs[no_layers-1]);
                  break;
          default: printf("Error in hidden layer activation function\n");
                   exit(0);
                   break;
       }    
       
       for(i = 0; i < param->out_size;i++)
          final_output[test_sample][i] = outputs[no_layers-1][i+1];

    }


    if(param->out_size == 1){
       for(test_sample = 0; test_sample < param->no_rows_test; test_sample++){
          if(final_output[test_sample][0] < 0.5)
             final_output[test_sample][0] = 0;
          else
             final_output[test_sample][0] = 1;

       }
    }
    else{
       for(test_sample = 0; test_sample < param->no_rows_test; test_sample++){
          double max = 1;
          int max_class;
          for(i = 0; i < param->out_size; i++){
             if(final_output[test_sample][i] > max){
                max = final_output[test_sample][i];
                max_class = i+1;
             }
          }
          final_output[test_sample][0] = max_class;
       }
    }

    if(param->out_size == 1){
       int tp = 0, tn = 0, fp = 0, fn = 0;
       for(test_sample = 0; test_sample < param->no_rows_test; test_sample++){
          if(final_output[test_sample][0] == 0){
             if(param->data_test[test_sample][param->no_cols-1] == 0)
                ++tn;
              else
                ++fn;
          }
          else{
             if(param->data_test[test_sample][param->no_cols-1] == 1)
                ++tp;
             else
                ++fp;
          }
       }
       double accuracy = (double)(tp + tn)/param->no_rows_test;

       printf("\nAccuracy: %.2f\n\n", accuracy * 100);


    }
    else{
       int** conf_matrix = (int**)calloc(param->out_size, sizeof(int*));
       for(i = 0; i < param->out_size; i++)
          conf_matrix[i] = (int*)calloc(param->out_size, sizeof(int));

       int actual_class, predicted_class;
       for(test_sample = 0; test_sample < param->no_rows_test; test_sample++){
          actual_class = param->data_test[test_sample][param->no_cols - 1] - 1;
          predicted_class = final_output[test_sample][0] - 1;

          ++conf_matrix[actual_class][predicted_class];

       }


    
       double accuracy = 0.0;
       for (i = 0; i < param->out_size; i++)
           accuracy += conf_matrix[i][i];
       accuracy /= param->no_rows_test;

       printf("\nAccuracy: %.2lf\n\n", accuracy * 100);

       for (i = 0; i < param->out_size; i++)
           free(conf_matrix[i]);
       free(conf_matrix);
   }

   //Free memory
   for (i = 0; i < param->no_rows_test; i++)
       free(final_output[i]);

   free(final_output);

   for (i = 0; i < no_layers; i++)
       free(outputs[i]);

   free(outputs);

   for (i = 0; i < no_layers; i++)
       free(inputs[i]);

   free(inputs);
}
