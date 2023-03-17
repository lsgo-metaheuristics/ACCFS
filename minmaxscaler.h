//=======================================================================================
// Adaptive Cooperative Coevolutionary Feature Selector
//=======================================================================================
// Version     : v1.0
// Created on  : February 2023
//
// More details on the following paper:
//
// Adaptive Cooperative Coevolutionary Differential Evolution for Parallel Feature
// Selection in High - Dimensional Datasets
// Marjan Firouznia, Pietro Ruiu and Giuseppe A.Trunfio
//
//=======================================================================================

#ifndef MINMAXSCALER_H
#define MINMAXSCALER_H

#include <vector>

class MinMaxScaler {
public:
    // Constructor
    MinMaxScaler();

    // Fit the scaler to the input data
    void fit(const std::vector<std::vector<float>> &data);

    // Scale the input data using the computed minimum and maximum values
    std::vector<std::vector<float>> transform(const std::vector<std::vector<float>> &data) const;

private:
    std::vector<float> _min_vals;
    std::vector<float> _max_vals;
};

#endif // MINMAXSCALER_H

