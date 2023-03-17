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

#include "matrix.h"

matrix::matrix(int numRows, int numCols) : rows(numRows), cols(numCols) {
    data = new float[numRows * numCols];
    for (int i = 0; i < numRows * numCols; ++i) {
        data[i] = 0.0;
    }
}

matrix::~matrix() {
    delete[] data;
}

float *matrix::operator[](int rowIndex) {
    return data + rowIndex * cols;
}

int matrix::numRows() const {
    return rows;
}

int matrix::numCols() const {
    return cols;
}