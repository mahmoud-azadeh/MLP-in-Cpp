//#pragma once
#include <iostream>
#include <vector>
//#include <conio.h>
#include <string>
#include <random>
#include <ctime>
#include <cstdint>

#include "include/cmath.h"
#include "include/dataset.h"
#include "include/ann.h"

// g++ -O2 mlp.cpp -o  mlp
// ./mlp

int main()
{		
	// loading CFAR10 datasets
	std::cout << "\nloading datasets ...";

	unsigned int num_train = 5000;  // number of train samples
	unsigned int num_test  = 1000;  // number of test samples
	unsigned int num_val   = 1000;  // number of validation samples

	std::string dir = "./cifar-10-batches-bin/"; // the folder path of the datasets
	
	uiuc::cfar cfar10(dir, num_train, num_test, num_val);

	std::cout << "\n-------------"<< std::endl;
	std::cout << "train samples:" << cfar10.train_dataset.cardinality <<std::endl;	
	std::cout << "test samples:" << cfar10.test_dataset.cardinality <<std::endl;
	std::cout << "validation samples:" << cfar10.val_dataset.cardinality <<std::endl;
		
	// preprocessing;
	std::cout << "\n--------------" << std::endl;
	std::cout << "\n pre-processing ..."<< std::endl;
	ann cann;
	std::vector<std::vector<double>> X_train = cann.preprocessing<double, uint8_t>(cfar10.train_dataset.X);
	std::vector<std::vector<double>> X_val   = cann.preprocessing<double, uint8_t>(cfar10.val_dataset.X, true);
	std::vector<std::vector<double>> X_test  = cann.preprocessing<double, uint8_t>(cfar10.test_dataset.X, true);
	
	std::cout << "setting parameters: " << std::endl;
	
	unsigned int input_dim = 3 * 32 * 32; // input dimensions (features)
	std::vector<unsigned int> hidden_dims = { 100, 100 }; // number of hidden layers
	unsigned int num_classes = 10; // number of output classes

	double weight_scale = 0.01; // used in weights initialization
	double reg = 0.005;  // l2-norm regulization

    unsigned int epochs = 10;
	unsigned int batch_size = 200;
	double learning_rate_decay = 0.95;

	//ann cann; 
	cann.config.optim = "adam"; // "sgd"; "adam"
	cann.config.learning_rate = 0.002;
	cann.config.dropout_ratio = 0.75;	

    cann.initialization<double, double>(input_dim, hidden_dims, num_classes, weight_scale);
    cann.config.saveFlag = false; //cann.config.normalization = "";
	    
	// training
	std::cout << "\ntraining ..."<< std::endl;
	cann.train<double, double, uint8_t>(
        X_train, //
        cfar10.train_dataset.y, //
        epochs,
        batch_size,
        learning_rate_decay,
        reg);

	// # Predict on the validation set
	std::cout << "\n -------------" << std::endl;
	std::cout << "\nprediction ..."<< std::endl;
    double val_acc = cann.accuracy<double, uint8_t>(
        cfar10.val_dataset.y,
        cann.predict<uint8_t, double>(X_val));

	double test_acc = cann.accuracy<double, uint8_t>(
        cfar10.test_dataset.y,
        cann.predict<uint8_t, double>(X_test));

			
	std::cout <<"validation accuracy: " << val_acc << std::endl;
    std::cout <<"test accuracy: " << test_acc << std::endl;
	return 0;
}