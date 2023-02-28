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

#ifndef MATRIX_H
#define MATRIX_H

class matrix
{
private:
    float* data;
    int rows;
    int cols;

public:
    matrix(int numRows, int numCols);
    ~matrix();

    float* operator[](int rowIndex);
    int numRows() const;
    int numCols() const;
};

#endif
