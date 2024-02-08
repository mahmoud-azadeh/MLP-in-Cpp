#pragma once
#include <iostream>
#include <vector>
#include <cstdint>
#include <string>

#include "cmath.h"
#include "var.h"

class layers  //: public uiuc::cmath
{

public:
    template <typename S, typename T, typename U, typename V>
    std::vector<std::vector <S>> affine(std::vector<std::vector <T>> X,
        std::vector<std::vector <U>> W,
        std::vector <V> b);

    template <typename S, typename T>
    std::vector<std::vector <S>> relu(std::vector<std::vector <T>> X);

    template <typename S, typename T, typename U, typename V>
    std::vector<std::vector <S>> affine_back(std::vector<std::vector <T>> X,
        std::vector<std::vector <U>> W,
        std::vector<std::vector <V>> dout,
        std::vector<std::vector <S>>& dW,
        std::vector <S>& db);

    template <typename S, typename T>
    void relu_back(std::vector<std::vector <S>>& dout, std::vector<std::vector <T>> X);

    template <typename S, typename T>
    std::vector<std::vector <S>> batchnorm(std::vector<std::vector <T>> X,
        struct configstruct config,
        std::vector<struct batchstruct <S, S>>&batch,
        std::vector<struct cacheLossStruct <S, S>>&cache,
        unsigned int i);

    template <typename S, typename T>
    std::vector < std::vector <S>> batchnorm_back(
        struct configstruct config,
        std::vector<struct batchstruct <S, T>> &batch,
        std::vector<struct cacheLossStruct <S, T>> &cache,
        unsigned int i);

    template <typename T>
    std::vector< std::vector<T>> dropout(std::vector< std::vector<T>> & X, 
        struct configstruct config);

    template <typename T>
    void dropout_back(std::vector< std::vector<T>>& dout, std::vector< std::vector<T>> mask);
 

    template <typename T, typename U, typename V>
    double softmax_loss(std::vector< std::vector<T>> X, std::vector<U> y, std::vector<std::vector <V>> &dout);
private:
    uiuc::cmath calc;
};


template <typename T>
std::vector<std::vector<T>> layers::dropout(std::vector< std::vector<T>> & X, struct configstruct config)
{
    unsigned int row = X.size();
    unsigned int col = X[0].size();

    
    std::vector<std::vector<T>> mask(row, std::vector<T>(col, 0));
    
    int seed = -1;
    if (config.drop_seed != -1)         
        seed = config.drop_seed;   
    
    // if (config.mode.compare("train") == 0 && config.dropout_ratio > 0 && config.dropout_ratio <= 1)
    {        
        unsigned int i, j;
        mask = calc.rand<double>(row, col, 0, 1, seed);
        T inv_dropout_ratio = 1 / config.dropout_ratio;
        for (i = 0; i < row; i++)
            for (j = 0; j < col; j++)
            {
                if (mask[i][j] > config.dropout_ratio)
                {
                    mask[i][j] = 0;                   
                    X[i][j]    = 0; 
                }
                else
                {
                    mask[i][j] = inv_dropout_ratio;
                    X[i][j]   *= inv_dropout_ratio; 
                }
                
            }
    }
   
    return mask;
}


template <typename T>
void layers::dropout_back(std::vector< std::vector<T>> &dout, std::vector< std::vector<T>> mask)
{
   
    unsigned int row = dout.size();
    unsigned int col = dout[0].size();
    //std::vector<std::vector<T>> dx(row, std::vector<T>(col, 0));

    unsigned int i, j;
    for (i = 0; i < row; i++)
        for (j = 0; j < col; j++)
        {
            if (mask[i][j] > 0)                
                dout[i][j] *= mask[i][j]; // dx[i][j] = dout[i][j] * mask[i][j];
            else                
                dout[i][j] = 0; // dx[i][j] = 0;
        }
      
    // return dx
} 

// affine out = X * W + b
template <typename S, typename T, typename U, typename V>
std::vector<std::vector <S>> layers::affine(std::vector<std::vector <T>> X,
    std::vector<std::vector <U>> W,
    std::vector <V> b)
{
    std::vector<std::vector <S>> out = calc.add<S, S, V>(calc.multiply<S, T, U>(X, W), b, 1);
    return out;
}

template <typename S, typename T>
std::vector<std::vector <S>> layers::relu(std::vector<std::vector <T>> X)
{
    unsigned row = X.size();
    unsigned col = X[0].size();
    std::vector<std::vector <S>> out(row, std::vector <S>(col, 0));

    unsigned int i, j;
    for (i = 0; i < row; i++)
        for (j = 0; j < col; j++)
        {
            if (X[i][j] > 0)
                out[i][j] = X[i][j];
            else
                out[i][j] = 0;
        }

    return out;
}


template <typename S, typename T, typename U, typename V>
std::vector<std::vector <S>> layers::affine_back(std::vector<std::vector <T>> X,
    std::vector<std::vector <U>> W,
    std::vector<std::vector <V>> dout,
    std::vector<std::vector <S>>& dW,
    std::vector <S>& db)
{
    
    dW = calc.multiply<S, T, V>(calc.transpose<T>(X), dout);    
    db = calc.sum<S, V>(dout, 0);   
    std::vector<std::vector <S>> dx = calc.multiply<S, V, U>(dout, calc.transpose<T>(W));

    return dx;
}


template <typename S, typename T>
void layers::relu_back(std::vector<std::vector <S>>& dout, std::vector<std::vector <T>> X)
{
   
    unsigned row = X.size();
    unsigned col = X[0].size();

    unsigned int i, j;
    for (i = 0; i < row; i++)
        for (j = 0; j < col; j++)
            if (X[i][j] <= 0)
                dout[i][j] = 0;   
}

template <typename S, typename T>
std::vector < std::vector <S>> layers::batchnorm(std::vector < std::vector <T>> X,    
    struct configstruct config, 
    std::vector <struct batchstruct <S, S>> &batch,
    std::vector <struct cacheLossStruct <S, S>> &cache,
    unsigned int i)
{
    std::vector < std::vector <S>> out;
    
    if (config.mode.compare("train") == 0)
    {      
        cache[i].sample_mean = calc.mean<S, T>(X, 0);        
        cache[i].sample_var = calc.add<S, S, S>(calc.var<S, T>(X, 0), config.epsilon);
        cache[i].xk = calc.divide<S, S,S>(calc.subtract<S,T,S>(X, cache[i].sample_mean), calc.msqrt<S>(cache[i].sample_var));
        out = calc.add<S, S, S>(calc.dot<S, S, S>(cache[i].xk, batch[i].gamma), batch[i].beta);

        batch[i].running_mean = calc.add<S, S, S >(calc.dot<S, S, S>(config.momentum, batch[i].running_mean),
            calc.dot<S, S, S>(1 - config.momentum, cache[i].sample_mean));
        
        batch[i].running_var = calc.add<S, S, S >(calc.dot<S, S, S>(config.momentum, batch[i].running_var),
            calc.dot<S, S, S>(1 - config.momentum, cache[i].sample_var));
    }
    
    else if (config.mode.compare("test") == 0)
    {
        std::vector < std::vector <S>> xk = calc.divide<S, S>(calc.subtract<T, S>(X, batch[i].running_mean),
            calc.msqrt<S>(batch[i].running_var));

        out = calc.add<S, S, S>(calc.dot<S, S, S>(xk, batch[i].gamma), batch[i].beta);
    }
    else 
    { 
        // # raise ValueError('Invalid forward batchnorm mode "%s"' % mode)
        std::cout << " the mode for batch normalization should be 'train' or 'test'" << std::endl;
            
    } 

    return out;
}


template <typename S, typename T>
std::vector < std::vector <S>> layers::batchnorm_back(
    struct configstruct config,
    std::vector <struct batchstruct<S,T>>& batch,
    std::vector <struct cacheLossStruct<S,T>>& cache,
    unsigned int idx)
{
    
    unsigned int m = cache[idx].xk.size();
    batch[idx].dgamma = calc.sum<S, S>(calc.dot<S, S, S>(cache[idx].dout, cache[idx].xk), 0);
   
    batch[idx].dbeta = calc.sum<S, S>(cache[idx].dout, 0);
    std::vector < std::vector <S>> dxk = calc.dot<S, S, S>(cache[idx].dout, batch[idx].gamma);

    std::vector < std::vector <S>> dx = calc.subtract<S, S, S>(
        calc.subtract<S, S, S>(dxk,
            calc.divide<S, S, unsigned int>(calc.sum<S, S>(dxk, 0), m), 1),
        calc.dot<S, S, S>(cache[idx].xk,
            calc.divide<S, S, unsigned int>(calc.sum<S, S>(calc.dot<S, S, S>(dxk, cache[idx].xk), 0), m)));;
      
    unsigned int row = dx.size(); unsigned int col = dx[0].size(); unsigned int i, j; S sample_std;
    for (j = 0; j < col; j++)
        for (i = 0, sample_std = sqrt(cache[idx].sample_var[j]); i < row; i++)
            dx[i][j] /= sample_std;    
    
    return dx;

}

template <typename T, typename U, typename V>
double layers::softmax_loss(std::vector< std::vector<T>> X, std::vector<U> y, std::vector<std::vector <V>> &dout)
{
    unsigned int num_element = X.size();
    std::vector< std::vector<T>> softmax_scores = calc.subtract<double, double, double>(X, calc.max<double>(X, 1));

    std::vector < std::vector <double>> exp_scores = calc.mexp<double, double>(softmax_scores);

    std::vector< std::vector<double>> prob = calc.divide<double, double, double>(exp_scores, calc.sum<double, double>(exp_scores, 1));

    double cost = 0; for (unsigned int i = 0; i < num_element; i++) { cost -= log(prob[i][y[i]]); }; cost /= double(num_element);

    for (unsigned int i = 0; i < num_element; i++) { prob[i][y[i]] -= 1; }

    unsigned row = prob.size();
    unsigned col = prob[0].size();
    dout = calc.zeros<double>(row, col);
    unsigned int i, j;
    for (i = 0; i < row; i++)
        for (j = 0; j < col; j++)
            dout[i][j] = prob[i][j] / num_element;

    return cost;
} 