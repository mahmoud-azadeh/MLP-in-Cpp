#pragma once
#include <iostream>
#include <vector>
#include <cstdint>
#include <string>

template <typename S, typename T>
struct batchstruct // an structure including variables for batch normalization
{
    std::vector <S> gamma;
    std::vector <T> beta;

    std::vector <S> dgamma;
    std::vector <T> dbeta;

    std::vector <T> running_mean;
    std::vector <S> running_var;
};

template <typename S, typename T>
struct paramstruct // an structure including variables of the model (weights and biases)
{
    std::vector < std::vector <S>> W;
    std::vector <T> b;
};
template <typename S, typename T>
struct cacheBatchStruct // an structure including variables for batch normalization
{
    std::vector < std::vector <T>> xk;
    std::vector <T> sample_mean;
    std::vector <S> sample_var;
};


template <typename U, typename V>
struct cacheOptStruct // an structure including variables for optimization
{
    std::vector < std::vector <U>> vW;
    std::vector < std::vector <U>> mW;
    //std::vector < std::vector <U>> dW2;

    std::vector <V> vb;    
    std::vector <V> mb;    
    //std::vector <V> db2;

    std::vector <V> vbeta;
    std::vector <V> mbeta;
    //std::vector <V> dbeta2;

    std::vector <V> vgamma;
    std::vector <V> mgamma;
    //std::vector <V> dgamma2;
};

template <typename S, typename T>
struct cacheLossStruct // caching variables through backward computations
{
    std::vector < std::vector <S>> out;
    std::vector < std::vector <T>> X;

    std::vector < std::vector <S>> dout;
    std::vector < std::vector <T>> dX;

    // batch normalization
    std::vector < std::vector <T>> bout;
    std::vector < std::vector <T>> xk;
    std::vector <T> sample_mean;
    std::vector <S> sample_var;

    std::vector < std::vector <T>> mask;
};

struct configstruct
{
    unsigned int num_layers = 1;

    double epsilon = 1e-8;
    double momentum = 0.9;
    double learning_rate = 1e-3;
    double decay_rate = 0.99;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double time = 0.0;

    std::string optim = "sgd"; // "sgd"; "adam"
    std::string normalization = "batchnorm";
    std::string mode = "train";

    int drop_seed = -1;
    double dropout_ratio = 1;

    bool saveFlag = false;
};