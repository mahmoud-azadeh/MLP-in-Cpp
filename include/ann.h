#pragma once
#include <iostream>
#include <vector>
#include <cstdint>
#include <string>

#include "cmath.h"
#include "layers.h"
#include "optim.h"
#include "var.h"

class ann //: public uiuc::cmath
{

public:        

    struct configstruct config;
    std::vector<double> train_acc_history;

    template <typename T, typename U>
    T accuracy(std::vector<U> y_truth, std::vector<U> y_pred);    
    
    template <typename U, typename V>
    void initialization(
        unsigned int input_dim = 3 * 32 * 32,
        std::vector<unsigned int> hidden_dims = { 10, 20 },
        unsigned int num_classes = 10,
        U weight_scale = 1e-4);   

    template <typename T, typename U, typename V>
    std::vector<T> train(std::vector<std::vector<U>> X, std::vector<V> y,
        unsigned int num_epochs, unsigned int batch_size, double learning_rate_decay, double reg);
    template <typename T, typename U>
    std::vector <T> predict(std::vector<std::vector<U>> X);

    template <typename T, typename U>
    std::vector<std::vector<T>> preprocessing(std::vector<std::vector<U>> X, bool flag = false);
    
private:
    layers clayers;
    optim coptim;  
    uiuc::cmath calc;

    std::vector<double> mean_image;
    std::vector<std::vector <double>> scores;
    std::vector<std::vector <double>> dout;

    std::vector <struct paramstruct <double, double>> param;
    std::vector <struct paramstruct <double, double>> grad;
    std::vector <struct batchstruct <double, double>> batch;  

    template <typename S, typename T, typename U, typename V>
    void ann_forward(std::vector<std::vector <T>> X, std::vector <struct cacheLossStruct <S, S>> &cache);

    template <typename S, typename T, typename U, typename V>
    void ann_backward(std::vector<std::vector <T>> dout, std::vector <struct cacheLossStruct <S, S>> cache);

    double regularization(double loss, double reg = 0.0); 
    template <typename T, typename U>
    double loss(std::vector<std::vector <T>> X, std::vector <U>, double reg);

    template <typename T, typename U>
    struct paramstruct <double, double> numerical_gradient(
        std::vector<std::vector <T>> X,
        std::vector <U> y,        
        unsigned int layer,
        double reg = 0.0, 
        double h = 1e-5);

    template <typename T, typename U>
    T relative_error(std::vector<std::vector <U>> X, std::vector<std::vector <U>> Y);

    template <typename T, typename U>
    T relative_error(std::vector <U> x, std::vector <U> y);
};

/*
The function preprocessing normalizes the input images:
First, it normalizes the images values between 0, 1, then removes the mean-value from the images.
The mean-value of the train dataset is used for the validation dataset and test dataset.
*/
template <typename T, typename U>
std::vector<std::vector<T>> ann::preprocessing(std::vector<std::vector<U>> X, bool flag)
{    
    std::vector<std::vector<T>> nX = calc.divide<T, U, uint8_t>(X, 255);
    if (flag == false)
        // mean_image = mean<T, U>(X, 0);
        mean_image = calc.mean<T, T>(nX, 0);
    // return subtract<T, U, T>(X, mean_image);;
    return calc.subtract<T, T, T>(nX, mean_image);
}

/*
- creating the model defined by the input dimension, hidden layers, and number of output classes.
- initializing the parameters.
*/
template <typename U, typename V>
void ann::initialization(
    unsigned int input_dim,
    std::vector<unsigned int> hidden_dims,
    unsigned int num_classes,
    U weight_scale
)
{    
    // assigning the dimensions of the (input & hidden) layers
    std::vector<unsigned int> layers_dims(1 + hidden_dims.size(), 0);
    layers_dims[0] = input_dim;
    for (unsigned int i = 0; i < hidden_dims.size(); i++)
        layers_dims[i+1] = hidden_dims[i];    
    unsigned int num_layers = layers_dims.size();
   
    param.resize(num_layers); // defining the variables of the model (weights and biases)
    batch.resize(num_layers-1); // defining the variables for batch normalization

    // initializing the weights and biases of the model
    // initializing the parameters of batch normalzation layers  
    for (unsigned int i = 0; i < num_layers - 1; i++)
    {
        param[i].W = calc.randn<U>(layers_dims[i], layers_dims[i + 1], 0, weight_scale);
        param[i].b = calc.zeros<V>(layers_dims[i + 1]);
        
        if (config.normalization.compare("batchnorm") == 0)
        {            
            batch[i].beta = calc.zeros<V>(layers_dims[i + 1]);
            batch[i].running_mean = calc.zeros<V>(layers_dims[i + 1]);
           
            batch[i].gamma       = calc.ones<U>(layers_dims[i + 1]);
            batch[i].running_var = calc.ones<U>(layers_dims[i + 1]);
        } 

    } 

    param[num_layers-1].W = calc.randn<U>(layers_dims[num_layers-1], num_classes, 0, weight_scale);
    param[num_layers-1].b = calc.zeros<V>(num_classes);

    // config params.
    config.num_layers = num_layers;
    config.momentum = 0.9;    
    config.learning_rate = 1e-3;    
    config.epsilon = 1e-8;    
    config.decay_rate = 0.99;
    config.beta1 = 0.9;
    config.beta2 = 0.999;

    return;    
} 

template <typename S, typename T, typename U, typename V>
void ann::ann_forward(std::vector<std::vector <T>> X, std::vector <struct cacheLossStruct <S, S>> &cache)
{    
    unsigned int num_layers = config.num_layers;
    
    cache.resize(num_layers);
    
    // farward computation through all layers
    cache[0].X = calc.copy<S,T>(X);
    for (unsigned int layer = 0; layer + 1 < num_layers; layer++)
    {
        // performing affine computation: out = X * W + b
        cache[layer].out = clayers.affine<S, T, U, V>(cache[layer].X, param[layer].W, param[layer].b);

        if (config.normalization.compare("batchnorm") == 0)
        {
            cache[layer].bout  = clayers.batchnorm<S,S>(cache[layer].out, config, batch, cache, layer);
            cache[layer + 1].X = clayers.relu<S, S>(cache[layer].bout);
        }
        else 
            cache[layer + 1].X = clayers.relu<S, S>(cache[layer].out);

        // erforming dropout in train mode (not in inferene mode)
        if (config.dropout_ratio > 0 && config.dropout_ratio < 1 && config.mode.compare("train") == 0)
            cache[layer].mask = clayers.dropout<S>(cache[layer + 1].X, config);
    } 

    // last layer: scores = X * W + b
    scores = clayers.affine<S, T, U, V>(cache[num_layers - 1].X, 
                                        param[num_layers - 1].W, 
                                        param[num_layers - 1].b);
    
}

// performing backward gradients
template <typename S, typename T, typename U, typename V>
void ann::ann_backward(std::vector<std::vector <T>> dout, std::vector <struct cacheLossStruct <S, S>> cache)
{
    unsigned int num_layers = config.num_layers;
    if (num_layers < 2)
        return;
    grad.resize(num_layers);
   
    // computing the gradients for layer num_layers - 2
    cache[num_layers - 2].dout = clayers.affine_back<S, T, U, V>(cache[num_layers - 1].X,
                                                                param[num_layers - 1].W,
                                                                cache[num_layers - 1].dout,
                                                                grad[num_layers - 1].W,
                                                                grad[num_layers - 1].b);
    
    // computing the gradients for layer num_layers - 2 to 0
    for (int layer = num_layers-2; layer >= 0; layer--)
    {
        // performing dropout backward in the case of train mode  
        if (config.dropout_ratio > 0 && config.dropout_ratio < 1 && config.mode.compare("train") == 0)
            clayers.dropout_back(cache[layer].dout, cache[layer].mask);

        // in the case of batch normalization   
        if (config.normalization.compare("batchnorm") == 0)
        {
            clayers.relu_back<S, T>(cache[layer].dout, cache[layer].bout);
            cache[layer].dout = clayers.batchnorm_back<S>(config, batch, cache, layer);
        }
        else
            clayers.relu_back<S,T>(cache[layer].dout, cache[layer].out);       
        
        // computing the gradients dout, dW and dB of the corresponding layer        
        int layindx = layer - 1; if (layindx < 0) {layindx = 0;} // cache[0].dout is overwritten here!, which is not a problem.
        cache[layindx].dout = clayers.affine_back<S, T, U, V>(cache[layer].X,
            param[layer].W, cache[layer].dout, grad[layer].W, grad[layer].b);
        
    }           
}

// performing l2-norm regularization
double ann::regularization(double cost, double reg)
{
    for (unsigned int layer = 0; layer < config.num_layers; layer++)
    {        
        cost += 0.5 * reg * calc.sum<double, double>(calc.power<double, double>(param[layer].W));
        grad[layer].W = calc.add<double, double, double>(grad[layer].W, calc.dot<double, double, double>(reg, param[layer].W));
    }
    return cost;
} 

// computing loss of the model
template <typename T, typename U>
double ann::loss(std::vector<std::vector <T>> X, std::vector <U> y, double reg)
{
    std::vector <struct cacheLossStruct <double, double>> cache;

    // the forward computation gives the scorres and dout
    ann_forward<double, T, double, double>(X, cache);

    // a softmax fuction is used in last layer
    double cost = clayers.softmax_loss<double, U, double>(scores, y, dout);
    cache[config.num_layers - 1].dout = dout;
    
    ann_backward<double, double, double, double>(dout, cache);

    if (reg != 0)
        cost = regularization(cost, reg);

    return cost;
}

template <typename T, typename U>
std::vector <T> ann::predict(std::vector<std::vector<U>> X)
{    
    config.mode = "test";
    std::vector <struct cacheLossStruct <double, double>> cache;
    ann_forward<double, double, double, double>(X, cache);
    
    std::vector<T> y_pred(X.size(), 0);
    
    unsigned int row = scores.size(); unsigned int col = scores[0].size();
    double maxvalue = 0;	unsigned int maxarg;
    for (unsigned int i = 0; i < row; i++)
    {
        maxvalue = scores[i][0]; maxarg = 0;
        for (unsigned int j = 1; j < col; j++)
            if (maxvalue < scores[i][j])
            {
                maxvalue = scores[i][j];
                maxarg = j;
            }
        y_pred[i] = T(maxarg);
    }
    config.mode = "train";
    return y_pred;
}

template <typename T, typename U>
T ann::accuracy(std::vector<U> y_truth, std::vector<U> y_pred)
{
    unsigned int count = 0;
    for (unsigned int i = 0; i < y_truth.size(); i++)
        if (y_truth[i] == y_pred[i])
            count++;
    return T(count) / T(y_truth.size());
}


template <typename T, typename U, typename V>
std::vector<T> ann::train(std::vector<std::vector<U>> X, std::vector<V> y,
    unsigned int epochs, unsigned int batch_size, double learning_rate_decay, double reg)
{

    std::vector <struct cacheOptStruct <double, double>> cacheOpt;
    cacheOpt.resize(config.num_layers);    
    // initializing the optimization parameters
    for (unsigned int i = 0; i < config.num_layers; i++)
    {
        //    params["v" * key] = zeros(size(params[key]));
        cacheOpt[i].vW = calc.zeros<double>(size(param[i].W), size(param[i].W[0]));
        cacheOpt[i].vb = calc.zeros<double>(size(param[i].b));

        // params["m" * key] = zeros(size(params[key]));
        cacheOpt[i].mW = calc.zeros<double>(size(param[i].W), size(param[i].W[0]));
        cacheOpt[i].mb = calc.zeros<double>(size(param[i].b));

        if (i + 1 < config.num_layers && config.normalization.compare("batchnorm") == 0)
        {
            cacheOpt[i].vbeta = calc.zeros<double>(size(param[i].b));
            cacheOpt[i].mbeta = calc.zeros<double>(size(param[i].b));

            cacheOpt[i].vgamma = calc.zeros<double>(size(param[i].b));
            cacheOpt[i].mgamma = calc.zeros<double>(size(param[i].b));
        }
    } 
   
    unsigned int num_train = X.size();   
    unsigned int num_of_batches = int(round(0.5 + double(num_train) / double(batch_size)));
   
    std::vector<T> loss_history(epochs * num_of_batches, 0);    
    train_acc_history.resize(epochs, 0);

    double cost = 0;
    unsigned int hidx = 0;

    for (unsigned int epoch = 0; epoch < epochs; epoch++)
    {        
        unsigned int batch_index = 0;
        unsigned int lenStop     = 0;
        unsigned int dim_feature = X[0].size();
        double train_loss        = 0;

        for (unsigned int batchid = 0; batchid < num_of_batches; batchid++)
        {            
            lenStop = batch_index + batch_size;
            if ((lenStop > num_train) || (lenStop >= num_train - batch_size / 2))            
                lenStop = num_train; 

            std::vector<std::vector<U>> X_batch(lenStop- batch_index, std::vector<U>(X[0].size(), 0));
            std::vector<V> y_batch(lenStop - batch_index, 0);
            for (unsigned int i = batch_index; i < lenStop; i++)
            {
                for (unsigned int j = 0; j < dim_feature; j++)
                    X_batch[i- batch_index][j] = X[i][j];
                y_batch[i- batch_index] = y[i];
            }
            
            if (batch_index > num_train - batch_size / 2)
                break;
            batch_index += batch_size;            

            // performing the forward, backward (gradient) and loss computations
            cost = loss<U,V>(X_batch, y_batch, reg);
            
            loss_history[hidx++] = cost; // loss_history.push_back(cost);
            train_loss += cost;
            
            // updating the parameters by perfoming the corresponding optimization method
            if (config.optim.compare("sgd") == 0)
                coptim.sgd(param, grad, batch, config);
            else if (config.optim.compare("adam") == 0)
                coptim.adam(param, grad, batch, cacheOpt, config);
        } 

        std::cout << "epoch " << 1+epoch<<"/" << epochs << " -> train loss " << train_loss / num_of_batches << std::endl;
        //std::cout << epoch << ", learning rate: " << config.learning_rate << std::endl;
        
        // Decay learning rate        
        config.learning_rate *= learning_rate_decay;

        // # Check accuracy      
        //double train_accuracy = accuracy<double, V >(y, predict<V, U>(X));
        //train_acc_history[epoch] = train_accuracy;
        //std::cout << 1+epoch<<"/" << epochs << " -> train accuracy: " << train_accuracy << std::endl;

        
    }
    return loss_history;
}

template <typename T, typename U>
T ann::relative_error(std::vector<std::vector <U>> X, std::vector<std::vector <U>> Y)
{   
    unsigned int row = X.size();
    unsigned int col = X[0].size();
    T error = 0; T max_error = 0; T maxvalue = 0;  T z = 0;
   
    for (unsigned int i = 0; i < row; i++)
        for (unsigned int j = 0; j < col; j++)
        {
            z = abs(X[i][j]) + abs(Y[i][j]);
            if (maxvalue < z)
                maxvalue = z;

            error = abs(X[i][j] - Y[i][j]);
            if (max_error < error)
                max_error = error;
        }

    if (maxvalue < 1e-3)
        maxvalue = 1e-3;
   
    return max_error / maxvalue;
}

template <typename T, typename U>
T ann::relative_error(std::vector <U> x, std::vector <U> y)
{    
    unsigned int len = x.size();

    T error = 0; T max_error = 0; T maxvalue = 0;  T z = 0;
    for (unsigned int i = 0; i < len; i++)
    {
        z = abs(x[i]) + abs(y[i]);
        if (maxvalue < z)
            maxvalue = z;

        error = abs(x[i] - y[i]);
        if (max_error < error)
            max_error = error;
    }

    if (maxvalue < 1e-3)
        maxvalue = 1e-3;
    
    return max_error / maxvalue;
}

template <typename T, typename U>
struct paramstruct <double, double> ann::numerical_gradient(
    std::vector<std::vector <T>> X, 
    std::vector <U> y,
    unsigned int layer,
    double reg,
    double h
)
{
    struct paramstruct <double, double> numeric_grad;
    unsigned int row = param[layer].W.size();
    unsigned int col = param[layer].W[0].size();
    
    numeric_grad.W = calc.zeros<double>(row,col);
    numeric_grad.b = calc.zeros<double>(param[layer].b.size());

    double oldval = 0;
    double costl, costh;
    for (unsigned int i = 0; i < row; i++)
    {
        for (unsigned int j = 0; j < col; j++)
        {
            oldval = param[layer].W[i][j];

            param[layer].W[i][j] = oldval - h;
            costl = loss<T, U>(X, y, reg);

            param[layer].W[i][j] = oldval + h;
            costh = loss<T, U>(X, y, reg);

            numeric_grad.W[i][j] = 0.5 * (costh - costl) / h;
            param[layer].W[i][j]  = oldval;
        }
    }

    for (unsigned int i = 0; i < param[layer].b.size(); i++)
    {
        oldval = param[layer].b[i];

        param[layer].b[i] = oldval - h;
        costl = loss<T, U>(X, y, reg);
        
        param[layer].b[i] = oldval + h;
        costh = loss<T, U>(X, y, reg);
        
        numeric_grad.b[i] = 0.5*(costh - costl) / h;
        param[layer].b[i]  = oldval;
    }

    return numeric_grad;
}