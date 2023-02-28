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

#include "Decomposer.h"
#include "CCDE.h"
#include <random>


Decomposer::Decomposer(CCDE &_CCOptimizer,                       
                       vector<unsigned> &_coordinates,
					   vector<unsigned>& coordinate_translator,
					   vector<float> &pfi_global,
                       unsigned _sizeOfSubcomponents,
                       unsigned _individualsPerSubcomponent,                       
                       bool RG,					   					   
					   bool initContextVector) : CCOptimizer(_CCOptimizer), sizeOfSubcomponents(_sizeOfSubcomponents),
					   individualsPerSubcomponent(_individualsPerSubcomponent), applyRandomGrouping(RG) 
{
    		
	baseCoordIndex.clear();
	std::uniform_real_distribution<float> distribution(0.0, 1.0);	
    coordinates = _coordinates;    
	unsigned problem_dimension = coordinates.size();

	for (unsigned i = 0; i < individualsPerSubcomponent; ++i)
	{
		vector<float> ind;
		for (int k = 0; k < problem_dimension; ++k)
		{
			ind.push_back(CCOptimizer.lowerLimit + distribution(CCOptimizer.local_eng) * (CCOptimizer.upperLimit - CCOptimizer.lowerLimit));
		}
		population.push_back(ind);
	}
		
    unsigned d = 0, current_size = sizeOfSubcomponents;
    while ( d< problem_dimension)
    {
		if (d + sizeOfSubcomponents > coordinates.size())
			current_size = coordinates.size() - d;
					
		baseCoordIndex.push_back(d);
		sizes.push_back(current_size);
		 
		SHADE* optimizer = NULL;		
		optimizers.push_back(optimizer);

        d += current_size;
    }
	
	numberOfSubcomponents = sizes.size();	

	if (initContextVector)
	{
		contextVector.resize(problem_dimension);
		for (unsigned i = 0; i < problem_dimension; ++i)
			contextVector[i] = population[individualsPerSubcomponent / 2][i];		
	}

	//prepare shuffled coordinates
	if (applyRandomGrouping )
	{
		vector<unsigned> sc = coordinates;
		for (int i = 0; i < MAX_NUM_OF_CYCLES; ++i)
		{
			shuffledCoordinates.push_back(sc);
			shuffle(sc.begin(), sc.end(), CCOptimizer.local_eng);
		}
	}
}




Decomposer::Decomposer(CCDE& _CCOptimizer,
	vector<set<unsigned>> decomposition,	
	unsigned _individualsPerSubcomponent,	
	bool initContextVector	
	) : CCOptimizer(_CCOptimizer), 
	individualsPerSubcomponent(_individualsPerSubcomponent) 
{
		
	std::uniform_real_distribution<float> distribution(0.0, 1.0);	
	baseCoordIndex.clear();
	coordinates.clear();

	unsigned problem_dimension = 0;
	for (int i = 0; i < decomposition.size(); ++i)
	{
		problem_dimension += decomposition[i].size();
		for (int c: decomposition[i])
			coordinates.push_back(c);
	}
	

	for (unsigned i = 0; i < individualsPerSubcomponent; ++i)
	{
		vector<float> ind;
		for (int k = 0; k < problem_dimension; ++k)
			ind.push_back(CCOptimizer.lowerLimit + distribution(CCOptimizer.local_eng) * (CCOptimizer.upperLimit - CCOptimizer.lowerLimit));
		population.push_back(ind);
	}

	unsigned d = 0;
	for (int i = 0; i<decomposition.size(); ++i)
	{
		unsigned current_size = decomposition[i].size();
		baseCoordIndex.push_back(d);
		sizes.push_back(current_size);				
		SHADE* optimizer = NULL;		
		optimizers.push_back(optimizer);
		d += current_size;
	}
		
	numberOfSubcomponents = sizes.size();

	if (initContextVector)
	{
		contextVector.resize(problem_dimension);
		for (unsigned i = 0; i < problem_dimension; ++i)
			contextVector[i] = population[individualsPerSubcomponent / 2][i];
	}
}


void Decomposer::reinitPopulation()
{			
	std::uniform_real_distribution<float> distribution(0.0, 1.0);	
	for (int j = individualsPerSubcomponent/2; j < individualsPerSubcomponent; j++)
	{
		for (int k = 0; k < coordinates.size(); ++k)
			population[j][k] = CCOptimizer.lowerLimit + distribution(CCOptimizer.local_eng) * (CCOptimizer.upperLimit - CCOptimizer.lowerLimit);
	}
}


vector< SHADE* >  Decomposer::allocateOptimizers(vector<unsigned> &indexes, RandomEngine& eng)
{
	for (int i = 0; i < indexes.size(); ++i)
	{
		unsigned j = indexes[i];
		SHADE *optimizer = new SHADE(sizes[j], individualsPerSubcomponent, *this, eng);
		optimizer->setCoordinates(&coordinates[baseCoordIndex[j]], sizes[j]);
		optimizers[j] = optimizer;		
		optimizer->loadIndividuals(population);		
	}
	return optimizers;
}



void  Decomposer::allocateOptimizers()
{
	for (int j = 0; j < optimizers.size(); ++j)
	{	
		RandomEngine *rnd = new RandomEngine();
		rnd->seed(time(0));
		SHADE* optimizer = new SHADE(sizes[j], individualsPerSubcomponent, *this, *rnd);
		optimizer->setCoordinates(&coordinates[baseCoordIndex[j]], sizes[j]);
		optimizers[j] = optimizer;
		optimizer->loadIndividuals(population);
	}
}

void  Decomposer::setOptimizersCoordinatesAndEvaluatePopulation(vector<unsigned> &indexes)
{
	for (int i = 0; i < indexes.size(); ++i)
	{
		unsigned j = indexes[i];
		SHADE *optimizer = optimizers[j];		
		optimizer->setCoordinates(&coordinates[baseCoordIndex[j]], sizes[j]);
		optimizer->updateIndividuals(population);
		optimizer->evaluateParents();
		optimizer->updateIndexOfBest();
	}	
}


void  Decomposer::setOptimizersCoordinates(vector<unsigned> &indexes)
{
	for (int i = 0; i < indexes.size(); ++i)
	{
		unsigned j = indexes[i];
		SHADE *optimizer = optimizers[j];
		optimizer->setCoordinates(&coordinates[baseCoordIndex[j]], sizes[j]);
	}
}

void  Decomposer::setOptimizersShuffledCoordinates(vector<unsigned> &indexes, unsigned cycle)
{
	if (cycle >= MAX_NUM_OF_CYCLES - 1)
	{
		cout << "reached the maximum number of cycles" << endl;
		exit(1);
	}

	for (int i = 0; i < indexes.size(); ++i)
	{
		unsigned j = indexes[i];
		SHADE *optimizer = optimizers[j];
		optimizer->setCoordinates(&shuffledCoordinates[cycle][baseCoordIndex[j]], sizes[j]);		
	}
}

void  Decomposer::setOptimizerShuffledCoordinates(unsigned index, unsigned cycle)
{
	if (cycle >= MAX_NUM_OF_CYCLES - 1)
	{
		cout << "reached the maximum number of cycles" << endl;
		exit(1);
	}
	
	optimizers[index]->setCoordinates(&shuffledCoordinates[cycle][baseCoordIndex[index]], sizes[index]);
}


void Decomposer::setPopulation(vector< vector<float> > &_population)
{    
	population.clear();
	for (unsigned i = 0; i < individualsPerSubcomponent; ++i)
		population.push_back(_population[i]);
}



void Decomposer::setOptimizersCoordinatesAndEvaluatePopulation()
{
    unsigned d = 0, k = 0, size = sizeOfSubcomponents;
    while ( d<coordinates.size() )
    {
        if (d + size > coordinates.size())
            size = coordinates.size() - d;

        SHADE *optimizer = optimizers[k];

        optimizer->setCoordinates(&(coordinates[d]), size);

        optimizer->loadIndividuals(population);
        optimizer->evaluateParents();
        optimizer->updateIndexOfBest();

        d += size;
        k++;
    }

}

Decomposer::~Decomposer()
{
    for (unsigned i = 0; i < optimizers.size(); ++i)
        delete optimizers[i];
};



void Decomposer::setSeed(unsigned seed)
{
    CCOptimizer.local_eng.seed(seed);
}



void Decomposer::setCoordinates(vector<unsigned> &coordinates)
{
    this->coordinates = coordinates;
}

void Decomposer::updateContextVector(SHADE *optimizer)
{
    vector<float> v = optimizer->getCollaborator();		
	CCOptimizer.numberOfEvaluations++;
	for (unsigned ld = 0; ld<v.size(); ld++)
      contextVector[optimizer->coordinates[ld]] = v[ld];
}



void Decomposer::buildContextVector()
{    	
	for (unsigned j = 0; j<optimizers.size(); ++j)
    {
		optimizers[j]->nfe = 0;
		vector<float> v = optimizers[j]->getCollaborator();
        for (unsigned ld = 0; ld<v.size(); ld++)
            contextVector[optimizers[j]->coordinates[ld]] = v[ld];			
    }    	
}


void Decomposer::buildContextVectorMT(int numThreads)
{

#pragma omp parallel for num_threads(numThreads)
	for (int j = 0; j<optimizers.size(); ++j)
	{		
		vector<float> v = optimizers[j]->getCollaborator();
		for (unsigned ld = 0; ld<v.size(); ld++)
 		  contextVector[optimizers[j]->coordinates[ld]] = v[ld];					
	}	
}



void Decomposer::randomGrouping()
{
    if ( optimizers.size() && this->applyRandomGrouping )
    {
        shuffle(coordinates.begin(), coordinates.end(), CCOptimizer.local_eng);
        //setOptimizersCoordinatesAndEvaluatePopulation();

		unsigned base = 0;
        for (unsigned i = 0; i < optimizers.size(); ++i)
        {
			optimizers[i]->setCoordinates(&(coordinates[base]), optimizers[i]->dimension);
			base += optimizers[i]->dimension;
            optimizers[i]->loadIndividuals(population);			
            optimizers[i]->evaluateParents();			            
			CCOptimizer.numberOfEvaluations += optimizers[i]->nfe;
			optimizers[i]->nfe = 0;
        }
    }
}



void Decomposer::randomGroupingMT(int numThreads)
{
	if (optimizers.size() && this->applyRandomGrouping)
	{
		shuffle(coordinates.begin(), coordinates.end(), CCOptimizer.local_eng);
		//setOptimizersCoordinatesAndEvaluatePopulation();

		unsigned base = 0;
#pragma omp parallel for num_threads(numThreads)
		for (int i = 0; i < optimizers.size(); ++i)
		{			
			optimizers[i]->setCoordinates(&(coordinates[base]), optimizers[i]->dimension);
			base += optimizers[i]->dimension;
			optimizers[i]->loadIndividuals(population);
			optimizers[i]->evaluateParents();					
#pragma omp atomic
			CCOptimizer.numberOfEvaluations += optimizers[i]->nfe;
			optimizers[i]->nfe = 0;
		}
	}
}


void Decomposer::parallelLocalSearch(int maxIte, int maxParallelTrials,  vector<float>& pfig, vector<unsigned>& coordinate_translator, int numAvailableThreads)
{				
	cout << "Local search.....";
	int problem_dimension = contextVector.size();
	vector<float> pfi;	
	for (int k = 0; k < problem_dimension; ++k)
		pfi.push_back(pfig[coordinate_translator[k]]);

	vector<float> pfir(problem_dimension, 0);
	pfir[0] = pfi[0];
	for (int k = 1; k < problem_dimension; ++k)
		pfir[k] = pfir[k - 1] + pfi[k];	
	
	float fc = this->CCOptimizer.current_best_fitness;
	vector<float> yc = contextVector;
	std::uniform_real_distribution<float> distribution(0.0, 1.0);	
	int ite = 0;
	while (ite++ < maxIte)
	{
		vector<float> fy(maxParallelTrials, 0);
		vector<int> flip(maxParallelTrials, 0);

#pragma omp parallel for schedule(dynamic) num_threads(numAvailableThreads)
		for (int i = 0; i < maxParallelTrials; ++i)
		{
			vector<float> b(contextVector.size(), 0);						
			float v = distribution(CCOptimizer.local_eng)* pfir[pfir.size()-1];
			int c;
			for (c = 0; c < pfir.size()-1; ++c)
				if (v < pfir[c+1])
					break;			
			vector<float> y = yc;
			if (y[c] > TH) 
			{
				y[c] = TH - 0.2;
				flip[i] = -c;
			}
			else
			{
				y[c] = TH + 0.2;
				flip[i] = c;
			}			
			fy[i] = CCOptimizer.computeFitnessValue(y);
		}
		CCOptimizer.numberOfEvaluations += maxParallelTrials;
		int k = min_element(fy.begin(), fy.end()) - fy.begin();		
		if (fy[k] < fc)
		{			
			fc = fy[k];
			int c = flip[k];
			yc[abs(c)] = TH + 0.2*abs(c)/c;
		}
	}
	cout << "end" << endl;	
	if (fc < this->CCOptimizer.current_best_fitness)
	{
		cout << "From local search: " << 1 - CCOptimizer.current_best_fitness << " ===> " << 1 - fc << endl;
		contextVector = yc;
		CCOptimizer.current_best_fitness = fc;
	}
}
