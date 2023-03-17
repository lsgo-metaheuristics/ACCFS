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

#pragma once
#include "shade.h"
#include <vector>
#include <set>
#include <queue>
#include <map>
#include <random>
#include <functional>   
#include <numeric>      


#define MAX_NUM_OF_CYCLES 1000

class CCDE;

using namespace std;

typedef mt19937 RandomEngine;

class Decomposer
{
public:
	vector<unsigned> coordinates;
	CCDE& CCOptimizer;
	vector< SHADE* > optimizers;
	int sizeOfSubcomponents;
	bool applyRandomGrouping;
	unsigned individualsPerSubcomponent;
	unsigned numberOfSubcomponents;

	//Current population
	vector< vector<float> > population;
	//Final global best position and context vector
	vector<float> contextVector;
	//Fitnesses of population
	vector< tFitness > fitnessValues;
	vector<unsigned> sizes;
	vector<unsigned> baseCoordIndex;

	Decomposer(CCDE &_CCOptimizer, vector<unsigned> &_coordinates,
               unsigned _sizeOfSubcomponents, unsigned _individualsPerSubcomponent, bool RG,
               bool initContextVector);

	~Decomposer();

    void allocateOptimizers();

    void parallelLocalSearch(int maxIte, int maxParallelTrials, vector<float>& pfi, vector<unsigned>& coordinate_translator);
	void reinitPopulation();
	void setOptimizerCoordinates(unsigned index);
};