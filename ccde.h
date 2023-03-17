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

#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <cstdlib>
#include <random>
#include <algorithm>
#include <vector>
#include <set>
#include <list>
#include <map>
#include "shade.h"
#include "decomposer.h"
#include <numeric>

#define TH 0.5


#define __USE_MINGW_ANSI_STDIO 1


typedef tFitness(*FunctionCallback)(int d, float* x);

extern FunctionCallback functions[];
extern char* functionsNames[];
extern float functionsDomainBounds[];

using namespace std;


class ConvPlotPoint
{
public:
	unsigned nfe;
	tFitness f;
	//unsigned subcomponentSize;
	//unsigned individuals;
	unsigned numFeatures;
	unsigned numberOfSubcomponents;
	ConvPlotPoint(unsigned  _nfe, tFitness _f, unsigned _nfeat, unsigned _ns) :
		nfe(_nfe), f(_f), numFeatures(_nfeat), numberOfSubcomponents(_ns)
	{};
};




/**
	@brief Main CCDE class.
	This class represents multiple swarms of particles which operate on the grouped directions of the main search space.
*/
class CCDE
{
	FunctionCallback fitness = NULL;



public:
	///Create the CCDE object with the specified optimization parameters
	///@param pNum number of particles
	///@param D problem dimension
	///@param ite number of generations
	///@param
	CCDE();

	///Destroy the CCDE object
	~CCDE();

	///Perform the optimization
	float optimize(FunctionCallback _function, unsigned dim, float domain_min, float domain_max, unsigned int maxNumberOfEvaluations, unsigned sizeOfSubcomponents,
		unsigned individualsPerSubcomponent,
		vector<ConvPlotPoint>& convergence, int seed, unsigned numItePerCycle,
		vector<set<unsigned>> &decomposition,
		vector<float> pfi);
	void printResults();
	vector<unsigned> getFeatureFlags();
	tFitness computeFitnessValue(vector<float>& x);
	bool keepFeature(int gc, FunctionCallback fitness, vector<unsigned> counters, float best_fitness, int count_threshold);

	unsigned problemDimension;
	Decomposer* decomposer;
	unsigned numberOfEvaluations;
	unsigned ite;
	float lowerLimit;
	float upperLimit;
	float elapsedTime;
	vector<unsigned> subcomponentSizes;
	vector<unsigned> numIndividualsPerSubcomponents;
	vector<float> current_solution;
	tFitness current_best_fitness;
	vector<unsigned> coordinate_translator;
	vector<float> localSolutionToGlobalSolution(vector<float> x);
	clock_t trainingTime;
	RandomEngine local_eng;
	uniform_real_distribution<float> unifRandom;
};

