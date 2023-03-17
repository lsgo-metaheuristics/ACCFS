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

#include <cmath>
#include <fstream>
#include <algorithm>
#include <ctime>
#include "matrix.h"
#include "ccde.h"
#include "decomposer.h"
#include "shade.h"



using namespace std;
extern vector<matrix*> distances_buff;


//******************************************************************************************/
//
//
//
//******************************************************************************************/
CCDE::CCDE()
= default;

//******************************************************************************************/
//
//
//
//******************************************************************************************/
vector<float> CCDE::localSolutionToGlobalSolution(vector<float> x)
{
	vector<float> rx(problemDimension, 0.0);
	for (int i = 0; i < coordinate_translator.size(); ++i)
		rx[coordinate_translator[i]] = x[i];
	return rx;
}

//******************************************************************************************/
//
//
//
//******************************************************************************************/
tFitness CCDE::computeFitnessValue(vector<float>& x)
{
	vector<float> rx = localSolutionToGlobalSolution(x);

	tFitness f = fitness(problemDimension, &rx[0]);
	return f;
}


//******************************************************************************************/
//
//
//
//******************************************************************************************/
CCDE::~CCDE()
{
	delete decomposer;
}



/**
 * \brief Determines if a feature is to be kept or not based on the given parameters
 *
 * \param gc The feature to be considered
 * \param fitnessFunction The function used to determine the fitness of the feature
 * \param counters The vector of counters of the feature
 * \param best_fitness The best fitness score achieved so far
 * \param count_threshold The threshold value for the number of times the feature has been used
 *
 * \return A boolean value indicating whether the feature should be kept or not
 */
bool CCDE::keepFeature(int gc, FunctionCallback fitnessFunction, vector<unsigned> counters,
	float best_fitness, int count_threshold)
{
	if (counters[gc] < count_threshold)
		return true;
	else
	{
		vector<float> ts = current_solution;
		ts[gc] = 1;
		numberOfEvaluations++;
		if (best_fitness - fitnessFunction(ts.size(), &ts[0]) > 0.001)
			return true;
	}
	return false;
}



/**
 * \brief Use a Cooperative Coevolutionary Differential Evolution algorithm to select features.
 *
 * \param _function Function to optimize.
 * \param dim Number of problem dimensions.
 * \param _lowerLimit Lower limit of search space.
 * \param _upperLimit Upper limit of search space.
 * \param maxNumberOfEvaluations Max number of evaluations.
 * \param _sizeOfSubcomponents Size of subcomponents.
 * \param individualsPerSubcomponent Number of individuals per subcomponent.
 * \param convergence Vector of convergence plots.
 * \param seed Random seed.
 * \param numItePerCycle Number of iterations per cycle.
 * \param decomposition Vector of sets of coordinates.
 * \param pfi Performance Feature Identification.
 * \return Total elapsed time of the optimization process.
 */
float CCDE::optimize(FunctionCallback _function, unsigned dim, float _lowerLimit, float _upperLimit,
	unsigned int maxNumberOfEvaluations,
	unsigned _sizeOfSubcomponents,
	unsigned individualsPerSubcomponent,
	vector<ConvPlotPoint>& convergence,
	int seed,
	unsigned numItePerCycle,
	vector<set<unsigned>>& decomposition,
	vector<float> pfi)
{
	int LS_freq = 4;
	int LS_maxIte = 10;
	int LS_maxParallelTrials = 100;
	int dec_adapt_freq = 10;
	int update_dec_fs_counter = 10;
	int terminate_fs_counter = 25;
	int counter_threshold = 5;
	double pfi_threshold = 0.8;

	local_eng.seed(seed);

	current_solution.resize(dim);

	unsigned size_of_sub = _sizeOfSubcomponents;
	unsigned ind_per_sub = individualsPerSubcomponent;
	tFitness prevFitness;
	unsigned fit_not_improve_counter;

	lowerLimit = _lowerLimit;
	upperLimit = _upperLimit;

	numberOfEvaluations = 0;
	fitness = _function;
	problemDimension = dim;
	current_best_fitness = std::numeric_limits<tFitness>::infinity();
	std::cout << "Initializing population..." << endl;
	vector<unsigned> allCoordinates;
	for (unsigned i = 0; i < problemDimension; ++i)
	{
		allCoordinates.push_back(i);
		coordinate_translator.push_back(i);
	}
	bool randomGrouping = true;
	vector<unsigned> counters(problemDimension, 0);
	this->decomposer = new Decomposer(*this, allCoordinates, size_of_sub, ind_per_sub, randomGrouping, true);
	numberOfEvaluations++;
	std::cout << "Starting optimization..." << endl;
	unsigned cycle = 1;
	int nfe_pc = decomposer->numberOfSubcomponents * ind_per_sub;
	decomposer->allocateOptimizers();
	if (decomposer->optimizers[0]->dimension != decomposer->optimizers[decomposer->optimizers.size() - 1]->dimension)
	{
		std::cout << "Decomposition in " << decomposer->optimizers.size() - 1 << " subcomponents of size " << decomposer->optimizers[0]->dimension;
		std::cout << " plus a subcomponent of size " << decomposer->optimizers[decomposer->optimizers.size() - 1]->dimension << endl;
	}
	else
		std::cout << "Decomposition in " << decomposer->optimizers.size() << " subcomponents of size " << decomposer->optimizers[0]->dimension << endl;
	prevFitness = current_best_fitness;
	fit_not_improve_counter = 0;
	clock_t startTime = clock();

	while (numberOfEvaluations < maxNumberOfEvaluations && fit_not_improve_counter < terminate_fs_counter)
	{
		if (prevFitness - current_best_fitness > 0)
			fit_not_improve_counter = 0;
		else
			fit_not_improve_counter++;

		prevFitness = current_best_fitness;

		if (cycle % LS_freq == 0)
			decomposer->parallelLocalSearch(LS_maxIte, LS_maxParallelTrials, pfi, coordinate_translator);

		if (cycle % dec_adapt_freq == 0)
		{
			vector<unsigned> c_on;
			vector<float> ncv;

#pragma omp parallel for schedule(dynamic) 
			for (unsigned int lc : decomposer->coordinates)
			{
				//Gets the actual feature corresponding to i-th coordinate
				unsigned gc = coordinate_translator[lc];
				if (pfi[gc] > pfi_threshold || keepFeature(gc, fitness, counters, current_best_fitness, counter_threshold))
				{
#pragma omp critical
					{
						c_on.push_back(lc);
						ncv.push_back(decomposer->contextVector[lc]);
					}
				}
			}


			if (c_on.size() < decomposer->coordinates.size() || fit_not_improve_counter > update_dec_fs_counter)
			{
				cout << "New problem size: " << c_on.size() << endl;

				vector<unsigned> newCoordinates(c_on.size());
				vector<unsigned> new_coordinate_translator(c_on.size());

#pragma omp parallel for 
				for (unsigned i = 0; i < c_on.size(); ++i)
				{
					newCoordinates[i] = i;
					new_coordinate_translator[i] = coordinate_translator[c_on[i]];
				}

				coordinate_translator = new_coordinate_translator;

				delete this->decomposer;

				size_of_sub = max(min((unsigned)(2.0 * _sizeOfSubcomponents * newCoordinates.size() / dim), _sizeOfSubcomponents), (unsigned)50);
				ind_per_sub = min((int)((nfe_pc * size_of_sub) / c_on.size()), 25);
				cout << "Individuals =" << ind_per_sub << endl;

				this->decomposer = new Decomposer(*this, newCoordinates, size_of_sub, ind_per_sub, randomGrouping, false);
				decomposer->allocateOptimizers();

#pragma omp parallel for 
				for (int i = 0; i < decomposer->coordinates.size(); ++i)
					this->decomposer->population[0][i] = ncv[i];

				decomposer->contextVector = ncv;

				if (decomposer->sizes[0] != decomposer->sizes[decomposer->sizes.size() - 1])
				{
					cout << "New decomposition in " << decomposer->sizes.size() - 1 << " subcomponents of size " << decomposer->sizes[0];
					cout << " plus a subcomponent of size " << decomposer->sizes[decomposer->sizes.size() - 1] << endl;
				}
				else
					cout << "New Decomposition in " << decomposer->optimizers.size() << " subcomponents of size " << decomposer->sizes[0] << endl;
			}
		}

		if (decomposer->applyRandomGrouping)
			shuffle(decomposer->coordinates.begin(), decomposer->coordinates.end(), local_eng);


#pragma omp parallel for schedule(dynamic) reduction(+: numberOfEvaluations)
		for (int j = 0; j < decomposer->optimizers.size(); ++j)
		{
			decomposer->setOptimizerCoordinates(j);
			decomposer->optimizers[j]->updateIndividuals(decomposer->population);
			decomposer->optimizers[j]->evaluateParents();
			decomposer->optimizers[j]->optimize(numItePerCycle);
			decomposer->optimizers[j]->storeIndividuals(decomposer->population);
			numberOfEvaluations += decomposer->optimizers[j]->nfe;
			decomposer->optimizers[j]->nfe = 0;
		}


#pragma omp parallel for 
		for (auto& optimizer : decomposer->optimizers)
			optimizer->updateContextVectorMT();

#pragma omp parallel for 
		for (int i = 0; i < dim; ++i)
			std::fill(&current_solution[i], &current_solution[i + 1], 0);

#pragma omp parallel for 
		for (int i = 0; i < decomposer->coordinates.size(); ++i)
		{
			current_solution[coordinate_translator[i]] = decomposer->contextVector[i];

			unsigned lc = decomposer->coordinates[i];
			unsigned gc = coordinate_translator[lc];
			if (decomposer->contextVector[lc] < TH)
				counters[gc]++;
			else
				counters[gc] = 0;
		}

		cout << cycle << " " << "NOE=" << numberOfEvaluations << "  fitness=" << 1.0 - current_best_fitness << endl;
		vector<unsigned int> ff = getFeatureFlags();
		unsigned n_of_f = count(ff.begin(), ff.end(), 1);
		convergence.push_back(ConvPlotPoint(numberOfEvaluations, 1.0 - current_best_fitness, n_of_f, decomposer->numberOfSubcomponents));
		cycle++;
	}

	vector<unsigned int> ff = getFeatureFlags();
	unsigned n_of_f = count(ff.begin(), ff.end(), 1);
	cout << "NOE=" << numberOfEvaluations << "  " << 1.0 - current_best_fitness << " " << n_of_f << endl;
	clock_t stopTime = clock();
	elapsedTime = ((float)(stopTime - startTime)) / CLOCKS_PER_SEC;
	return elapsedTime;
}


//******************************************************************************************/
//
//
//
//******************************************************************************************/
vector<unsigned> CCDE::getFeatureFlags()
{
	vector< unsigned int> ff(this->problemDimension, 0);
	for (int i = 0; i < current_solution.size(); ++i)
		if (current_solution[i] >= TH)
			ff[i] = 1;
	return ff;
}



//******************************************************************************************/
//
//
//
//******************************************************************************************/
void CCDE::printResults()
{
	std::cout << endl
		<< "Final results:" << endl
		<< "-   Elapsed time = " << elapsedTime << " s" << endl
		<< std::scientific << "-   Optimum = " << 1.0 - current_best_fitness << endl;

	//for (unsigned int i = 0; i < this->problemDimension; ++i)
	//std::cout << globalBestPosition[i] << " ";
	std::cout << std::endl;
}
