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

#include <vector>
#include <algorithm>
#include <stdexcept>
#include "MinMaxScaler.h"


MinMaxScaler::MinMaxScaler() : _min_vals(), _max_vals() {};

    // Fit the scaler to the input data
void MinMaxScaler::fit(const std::vector<std::vector<float>>& data)
    {
        if (data.size() == 0)         
            throw std::invalid_argument("Input data is empty");
        
        const int n_cols = data[0].size();
        _min_vals.resize(n_cols);
        _max_vals.resize(n_cols);
        for (int j = 0; j < n_cols; ++j) 
        {
            float min_val = data[0][j];
            float max_val = data[0][j];
            for (int i = 1; i < data.size(); ++i) 
            {
                if (data[i][j] < min_val) 
                    min_val = data[i][j];
                
                if (data[i][j] > max_val) 
                    max_val = data[i][j];                
            }
            _min_vals[j] = min_val;
            _max_vals[j] = max_val;
        }
    }

    // Scale the input data using the computed minimum and maximum values
    std::vector<std::vector<float>> MinMaxScaler::transform(const std::vector<std::vector<float>>& data) const
    {
        if (data.size() == 0) 
            throw std::invalid_argument("Input data is empty");
        
        const int n_cols = data[0].size();
        std::vector<std::vector<float>> scaled_data(data.size(), std::vector<float>(n_cols));
        for (int i = 0; i < data.size(); ++i) 
        {
            for (int j = 0; j < n_cols; ++j) 
            {
                if (_max_vals[j] == _min_vals[j]) 
                    scaled_data[i][j] = 0.0;                
                else 
                    scaled_data[i][j] = (data[i][j] - _min_vals[j]) / (_max_vals[j] - _min_vals[j]);                
            }
        }
        return scaled_data;
    }


