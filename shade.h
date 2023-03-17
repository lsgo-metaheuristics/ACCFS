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
#include <map>
#include <cstddef>

typedef float tFitness;

class Decomposer;

using namespace std;

typedef mt19937 RandomEngine;

class SHADE
{
	Decomposer& decomposer;

	struct doCompareIndividuals
	{
		doCompareIndividuals(const tFitness* _f) : f(_f) { }
		const tFitness* f;

		bool operator()(const int& i1, const int& i2)
		{
			return f[i1] < f[i2];
		}
	};

public:
	SHADE(unsigned _dimension, unsigned _numberOfIndividuals, Decomposer& _group, RandomEngine& _eng);

    void setCoordinates(unsigned* coordinates, unsigned numOfCoordinates);
	void update();
	void sortPopulation(vector<tFitness>& fitness, vector<int>& sortIndex);
	void evaluatePopulation(vector< vector<float> >& population, vector< tFitness >& fitness);
	int evaluateParents();

    int optimize(int iterations);
	void updateIndexOfBest();
	void loadIndividuals(vector< vector<float> >& population);
	void storeIndividuals(vector< vector<float> >& population);
	void updateIndividuals(vector< vector<float> >& population);

    void updateContextVectorMT();

    void handleBounds(vector<float>& child, vector<float>& parent);

    unsigned numThreads;
	unsigned nfe;
	vector<unsigned> coordinates;
	map<unsigned, unsigned> globalCoordToLocalCoord;
	unsigned dimension;
	vector< vector<float> > parents;
	vector< vector<float> > offsprings;
	vector<int> sortIndex;
	vector<float> SF;
	vector<float> CR;
	vector< tFitness > parentsFitness;
	vector< tFitness > offspringsFitness;
	tFitness bestFitness;
	unsigned indexOfBest;
	unsigned numberOfIndividuals;
	RandomEngine& eng;
	uniform_real_distribution<float> unifRandom;
	vector<float> success_sf;  // successful F values
	vector<float> success_cr;  // successful CR value
	vector<tFitness> dif_fitness;
	int archive_size;
	int memory_size;
	int memory_index;
	vector< vector<float> > archive;
	vector<float> memory_sf;
	vector<float> memory_cr;
	float SHADE_p;

};

