#pragma once
#include <iostream>
#include <vector>
#include <cstdint>
#include <string>

#include "var.h"

class optim
{

public:

    template <typename S, typename T>
    void sgd(std::vector <struct paramstruct <S, T>>& param,
        std::vector <struct paramstruct <S, T>> grad,
        std::vector <struct batchstruct <S, T>>& batch,
        struct configstruct config);

    template <typename S, typename T>
    void adam(std::vector <struct paramstruct <S, T>>& param,
        std::vector <struct paramstruct <S, T>> grad,
        std::vector <struct batchstruct <S, T>>& batch,
        std::vector <struct cacheOptStruct <S, T>>& cacheOpt,
        struct configstruct config);
private:
    template <typename S, typename T, typename U, typename V>
    void update_adam(std::vector < std::vector <S>>& W,
        std::vector < std::vector <T>> dW,
        std::vector < std::vector <U>>& mW,
        std::vector < std::vector <V>>& vW,
        struct configstruct config);

    template <typename S, typename T, typename U, typename V>
    void update_adam(std::vector <S>& W,
        std::vector <T> dW,
        std::vector <U>& mW,
        std::vector <V>& vW,
        struct configstruct config);
};

template <typename S, typename T>
void optim::sgd(std::vector <struct paramstruct <S, T>>& param,
    std::vector <struct paramstruct <S, T>> grad,
    std::vector <struct batchstruct <S, T>>& batch,
    struct configstruct config)
{
    double learning_rate = config.learning_rate;
    
    for (unsigned int layer = 0; layer < config.num_layers; layer++)
    {
       
        unsigned int row = param[layer].W.size(); unsigned int col = param[layer].W[0].size();
        unsigned int i, j;
        for (i = 0; i < row; i++)
            for (j = 0; j < col; j++)
                param[layer].W[i][j] -= learning_rate * grad[layer].W[i][j];
        for (i = 0; i < param[layer].b.size(); i++)
            param[layer].b[i] -= learning_rate * grad[layer].b[i];
        if (layer + 1 < config.num_layers && config.normalization.compare("batchnorm") == 0)
        {
            for (i = 0; i < batch[layer].beta.size(); i++)
                batch[layer].beta[i] -= learning_rate * batch[layer].dbeta[i];
            for (i = 0; i < batch[layer].gamma.size(); i++)
                batch[layer].gamma[i] -= learning_rate * batch[layer].dgamma[i];
        }
    }
}

template <typename S, typename T, typename U, typename V>
void optim::update_adam(std::vector < std::vector <S>>& W,
    std::vector < std::vector <T>> dW,
    std::vector < std::vector <U>>& mW,
    std::vector < std::vector <V>>& vW,
    struct configstruct config)
{
    U t = config.time;

    U learning_rate = config.learning_rate;
    U beta1 = config.beta1;
    U beta2 = config.beta2;
    U epsilon = config.epsilon;
    U beta1t = (1 - pow(beta1, t));
    U beta2t = (1 - pow(beta2, t));
    U comp_beta1 = 1 - beta1;
    U comp_beta2 = 1 - beta2;

    unsigned int row = W.size(); unsigned int col = W[0].size();
    unsigned int i, j; U mt = 0; U vt = 0;
    for (i = 0; i < row; i++)
        for (j = 0; j < col; j++)
        {
            mW[i][j] = beta1 * mW[i][j] + comp_beta1 * dW[i][j];
            mt = mW[i][j] / beta1t;
            vW[i][j] = beta2 * vW[i][j] + comp_beta2 * pow(dW[i][j], 2);
            vt = vW[i][j] / beta2t;
            W[i][j] -= learning_rate * mt / (sqrt(vt) + epsilon);
        }
}
template <typename S, typename T, typename U, typename V>
void optim::update_adam(std::vector <S>& W,
    std::vector <T> dW,
    std::vector <U>& mW,
    std::vector <V>& vW,
    struct configstruct config)
{
    U t = config.time;

    U learning_rate = config.learning_rate;
    U beta1 = config.beta1;
    U beta2 = config.beta2;
    U epsilon = config.epsilon;
    U beta1t = (1 - pow(beta1, t));
    U beta2t = (1 - pow(beta2, t));
    U comp_beta1 = 1 - beta1;
    U comp_beta2 = 1 - beta2;


    unsigned int row = W.size();
    unsigned int i, j; U mt = 0; U vt = 0;

    for (i = 0; i < row; i++)
    {
        mW[i] = beta1 * mW[i] + comp_beta1 * dW[i];
        mt = mW[i] / beta1t;
        vW[i] = beta2 * vW[i] + comp_beta2 * pow(dW[i], 2);
        vt = vW[i] / beta2t;
        W[i] -= learning_rate * mt / (sqrt(vt) + epsilon);
    }
}

template <typename S, typename T>
void optim::adam(std::vector <struct paramstruct <S, T>>& param,
    std::vector <struct paramstruct <S, T>> grad,
    std::vector <struct batchstruct <S, T>>& batch,
    std::vector <struct cacheOptStruct <S, T>>& cacheOpt,
    struct configstruct config)
{
    config.time += 1;

    for (unsigned int layer = 0; layer < config.num_layers; layer++)
    {
        update_adam<S, S, S, S>(param[layer].W, grad[layer].W, cacheOpt[layer].mW, cacheOpt[layer].vW, config);

        update_adam<S, S, S, S>(param[layer].b, grad[layer].b, cacheOpt[layer].mb, cacheOpt[layer].vb, config);

        if (layer + 1 < config.num_layers && config.normalization.compare("batchnorm") == 0)
        {
            update_adam<S, S, S, S>(batch[layer].beta, batch[layer].dbeta, cacheOpt[layer].mbeta, cacheOpt[layer].vbeta, config);
            update_adam<S, S, S, S>(batch[layer].gamma, batch[layer].dgamma, cacheOpt[layer].mgamma, cacheOpt[layer].vgamma, config);
        }
    }
};
