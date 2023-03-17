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

#include "SHADE.h"
#include "CCDE.h"



SHADE::SHADE(unsigned _dimension, unsigned _numberOfIndividuals, Decomposer& _group, RandomEngine& _eng) :
	decomposer(_group), dimension(_dimension), numberOfIndividuals(_numberOfIndividuals), eng(_eng)
{
	nfe = 0;
	parentsFitness.resize(numberOfIndividuals, 0);
	offspringsFitness.resize(numberOfIndividuals, 0);
	SF.resize(numberOfIndividuals, 0);
	CR.resize(numberOfIndividuals, 0);
	success_sf.resize(numberOfIndividuals, 0);
	success_cr.resize(numberOfIndividuals, 0);
	indexOfBest = (unsigned)(numberOfIndividuals * unifRandom(eng));
	SHADE_p = 0.1;
	numThreads = 1;
}


void SHADE::setCoordinates(unsigned* _coordinates, unsigned numOfCoordinates)
{
	coordinates.clear();

	for (unsigned i = 0; i < numOfCoordinates; ++i)
	{
		coordinates.push_back(_coordinates[i]);
		globalCoordToLocalCoord[_coordinates[i]] = i;
	}
}


//******************************************************************************************/
//
//
//
//******************************************************************************************/
void SHADE::loadIndividuals(vector< vector<float> >& population)
{
	parents.clear();

	for (unsigned i = 0; i < numberOfIndividuals && i < population.size(); ++i)
	{
		vector< float > position;

		for (unsigned int coordinate : coordinates)
			position.push_back(population[i][coordinate]);

		parents.push_back(position);
	}
}

//******************************************************************************************/
//
//
//
//******************************************************************************************/
void SHADE::updateIndividuals(vector< vector<float> >& population)
{
	for (unsigned i = 0; i < numberOfIndividuals && i < population.size(); ++i)
		for (unsigned ld = 0; ld < coordinates.size(); ld++)
			parents[i][ld] = population[i][coordinates[ld]];
}

//******************************************************************************************/
//
//
//
//******************************************************************************************/
void SHADE::storeIndividuals(vector< vector<float> >& population)
{
	for (unsigned i = 0; i < numberOfIndividuals && i < population.size(); ++i)
		for (unsigned ld = 0; ld < coordinates.size(); ld++)
			population[i][coordinates[ld]] = parents[i][ld];
}


//******************************************************************************************/
//
//
//
//******************************************************************************************/
void SHADE::sortPopulation(vector<tFitness>& fitness, vector<int>& sortIndex)
{
	sortIndex.resize(fitness.size());

	for (unsigned int j = 0; j < fitness.size(); j++)
		sortIndex[j] = j;

	sort(&sortIndex[0], &sortIndex[0] + sortIndex.size(), doCompareIndividuals(&fitness[0]));
}

/**
 * \brief Handler function to ensure each child's value stays within the specified bounds defined by the parent.
 * \param child a reference to a float vector representing the child's candidate solution.
 * \param parent a reference to a float vector representing the parent candidate solution.
 * \details The function loops through the child vector and compares each element to the lower and upper limits determined by the optimizer. If the child's value is less than the lower limit, it sets the value to the average of the lower limit and parent's corresponding value. If the child's value is greater than the upper limit, it sets the value to the average of the upper limit and parent's corresponding value.
 */
void SHADE::handleBounds(vector<float>& child, vector<float>& parent)
{
	float l_min_region = decomposer.CCOptimizer.lowerLimit;
	float l_max_region = decomposer.CCOptimizer.upperLimit;

	for (int j = 0; j < dimension; j++)
	{
		if (child[j] < l_min_region)
		{
			child[j] = (l_min_region + parent[j]) / 2.0;
		}
		else if (child[j] > l_max_region)
		{
			child[j] = (l_max_region + parent[j]) / 2.0;
		}
	}
}


/**
 * \brief Update the population using the SHADE algorithm.
 *
 * This function sorts the population from best to worst, generates the CR and F values based on Gaussian and Cauchy distribution, generates the mutant vector, and evaluates the child population.
 * It then selects and saves the successful parameters. Finally, it updates the memories of the best performing vectors.
 *
 */
void SHADE::update()
{
	//Sort the population from best to worst
	sortPopulation(parentsFitness, sortIndex);

	//Generate the CR and F values based on Gaussian and Cauchy distribution, respectively	
	offsprings.resize(parents.size());

	for (int i = 0; i < parents.size(); i++)
	{
		//In each generation, CR_i and SF_i used by each individual i are generated by first selecting an index r_i randomly from [1, H] 
		int r_i = unifRandom(eng) * memory_size;
		float mu_sf = memory_sf[r_i];
		float mu_cr = memory_cr[r_i];
		cauchy_distribution<float> cauchy(mu_sf, 0.1);
		normal_distribution<float> gaussian(mu_cr, 0.1);

		//generate CR_i and repair its value
		if (mu_cr == -1)
		{
			CR[i] = 0;
		}
		else
		{
			CR[i] = gaussian(eng);
			if (CR[i] > 1)
				CR[i] = 1.0;
			else if (CR[i] < 0)
				CR[i] = 0.0;
		}

		//generate F_i and repair its value
		do {
			SF[i] = cauchy(eng);
		} while (SF[i] <= 0);
		if (SF[i] > 1) SF[i] = 1;


		unsigned r1, r2;
		//Generate the mutant vector
		//Randomly choose the p_best individual		
		unsigned p_index = sortIndex[unifRandom(eng) * parents.size() * SHADE_p];

		//Select parents randomly
		do
		{
			r1 = unifRandom(eng) * parents.size();
		} while (r1 == i);

		do
		{
			r2 = unifRandom(eng) * (parents.size() + archive.size());
		} while (r2 == i || r2 == r1);

		vector<float> child(dimension, 0);

		unsigned j_rnd = dimension * unifRandom(eng);

		if (r2 >= parents.size())
		{
			r2 -= parents.size();
			for (unsigned int j = 0; j < dimension; j++)
			{
				if (unifRandom(eng) < CR[i] || j == j_rnd)
					child[j] = parents[i][j] +
					SF[i] * (parents[p_index][j] - parents[i][j]) +
					SF[i] * (parents[r1][j] - archive[r2][j]);
				else
					child[j] = parents[i][j];
			}
		}
		else
		{
			for (unsigned int j = 0; j < dimension; j++)
			{
				if (unifRandom(eng) < CR[i] || j == j_rnd)
					child[j] = parents[i][j] +
					SF[i] * (parents[p_index][j] - parents[i][j]) +
					SF[i] * (parents[r1][j] - parents[r2][j]);
				else
					child[j] = parents[i][j];
			}
		}

		handleBounds(child, parents[i]);

		offsprings[i] = child;
	}

	//Evaluate the child population
	evaluatePopulation(offsprings, offspringsFitness);

	//Selection and save the successful parameters
	success_sf.clear();
	success_cr.clear();
	dif_fitness.clear();
	for (unsigned int i = 0; i < parents.size(); i++)
	{

		if (offspringsFitness[i] == parentsFitness[i])
		{
			parentsFitness[i] = offspringsFitness[i];
			for (unsigned int j = 0; j < dimension; j++)
				parents[i][j] = offsprings[i][j];
		}
		else
			if (offspringsFitness[i] < parentsFitness[i])
			{
				//parent vectors which were worse than the trial vectors are preserved			
				if (archive_size > 0)
				{
					if (archive.size() < archive_size)
					{
						archive.push_back(parents[i]);
					}
					else
					{
						//Whenever the size of the archive exceeds, randomly selected elements are deleted to make space for the newly inserted elements
						unsigned arch_pos = unifRandom(eng) * archive.size();
						for (int j = 0; j < dimension; j++)
							archive[arch_pos][j] = parents[i][j];
					}
				}

				dif_fitness.push_back(fabs(parentsFitness[i] - offspringsFitness[i]));

				//replace parent
				parentsFitness[i] = offspringsFitness[i];
				for (unsigned int j = 0; j < dimension; j++)
					parents[i][j] = offsprings[i][j];

				//Save the successful CR and SF values
				success_sf.push_back(SF[i]);
				success_cr.push_back(CR[i]);
			}
	}


	if (!success_cr.empty())
	{
		memory_sf[memory_index] = 0;
		memory_cr[memory_index] = 0;
		float temp_sum_sf = 0;
		float temp_sum_cr = 0;
		float sum = 0;

		for (int i = 0; i < success_cr.size(); i++)
			sum += dif_fitness[i];

		//weighted lehmer mean
		for (int i = 0; i < success_cr.size(); i++)
		{
			float weight = dif_fitness[i] / sum;

			memory_sf[memory_index] += weight * success_sf[i] * success_sf[i];
			temp_sum_sf += weight * success_sf[i];

			memory_cr[memory_index] += weight * success_cr[i] * success_cr[i];
			temp_sum_cr += weight * success_cr[i];
		}

		memory_sf[memory_index] /= temp_sum_sf;

		if (temp_sum_cr == 0 || memory_cr[memory_index] == -1)
			memory_cr[memory_index] = -1;
		else
			memory_cr[memory_index] /= temp_sum_cr;

		//increment the counter
		memory_index++;
		if (memory_index >= memory_size)
			memory_index = 0;
	}

}


/**
 * \brief Evaluates the fitness of the population using the SHADE algorithm
 *
 * \param population a vector of vectors representing the population
 * \param fitness a vector of tFitness containing the fitness values of the population
 */
void SHADE::evaluatePopulation(vector< vector<float> >& population, vector<tFitness>& fitness)
{
	fitness.resize(population.size());
	//#pragma omp parallel for num_threads(this->numThreads)
	for (int i = 0; i < population.size(); i++)
	{
		vector< float > xp(decomposer.coordinates.size());
		for (unsigned d = 0; d < decomposer.coordinates.size(); ++d)
			xp[d] = decomposer.contextVector[d];

		for (unsigned ld = 0; ld < dimension; ld++)
			xp[coordinates[ld]] = population[i][ld];

		fitness[i] = decomposer.CCOptimizer.computeFitnessValue(xp);
	}

	nfe += population.size();
}


//******************************************************************************************/
//
//
//
//******************************************************************************************/
int SHADE::evaluateParents()
{
	evaluatePopulation(parents, parentsFitness);
	return parents.size();
}


//******************************************************************************************/
//
//
//
//******************************************************************************************/
int SHADE::optimize(int iterations)
{
	//The archive is reset after a random grouping
	memory_size = parents.size() * 2;
	archive_size = parents.size() * 2;
	memory_index = 0;
	memory_sf.assign(memory_size, 0.5);
	memory_cr.assign(memory_size, 0.5);
	archive.clear();

	//Positions update
	for (int i = 0; i < iterations; ++i)
		update();

	updateIndexOfBest();

	return iterations * parents.size();
}



//******************************************************************************************/
//
//
//
//******************************************************************************************/
void SHADE::updateIndexOfBest()
{
	//find global best
	bestFitness = std::numeric_limits<float>::infinity();

	for (unsigned i = 0; i < parents.size(); ++i)
		if (parentsFitness[i] < bestFitness)
		{
			indexOfBest = i;
			bestFitness = parentsFitness[i];
		}
}


void SHADE::updateContextVectorMT()
{

	vector<float> cvt = decomposer.contextVector;

	for (unsigned ld = 0; ld < coordinates.size(); ld++)
		cvt[coordinates[ld]] = parents[indexOfBest][ld];
	tFitness nf = decomposer.CCOptimizer.computeFitnessValue(cvt);

#pragma omp critical 
	{
		if (nf < decomposer.CCOptimizer.current_best_fitness)
		{
			decomposer.CCOptimizer.current_best_fitness = nf;
			decomposer.contextVector = cvt;
			nfe++;
		}
	}
}


