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


Decomposer::Decomposer(CCDE &_CCOptimizer, vector<unsigned> &_coordinates,
                       unsigned _sizeOfSubcomponents, unsigned _individualsPerSubcomponent, bool RG,
                       bool initContextVector) : CCOptimizer(_CCOptimizer), sizeOfSubcomponents(_sizeOfSubcomponents),
                                                 individualsPerSubcomponent(_individualsPerSubcomponent),
                                                 applyRandomGrouping(RG) {

    baseCoordIndex.clear();
    std::uniform_real_distribution<float> distribution(0.0, 1.0);
    coordinates = _coordinates;
    unsigned problem_dimension = coordinates.size();

    for (unsigned i = 0; i < individualsPerSubcomponent; ++i) {
        vector<float> ind;
        for (int k = 0; k < problem_dimension; ++k) {
            ind.push_back(CCOptimizer.lowerLimit +
                          distribution(CCOptimizer.local_eng) * (CCOptimizer.upperLimit - CCOptimizer.lowerLimit));
        }
        population.push_back(ind);
    }

    unsigned d = 0, current_size = sizeOfSubcomponents;
    while (d < problem_dimension) {
        if (d + sizeOfSubcomponents > coordinates.size())
            current_size = coordinates.size() - d;

        baseCoordIndex.push_back(d);
        sizes.push_back(current_size);

        SHADE *optimizer = NULL;
        optimizers.push_back(optimizer);

        d += current_size;
    }

    numberOfSubcomponents = sizes.size();

    if (initContextVector) {
        contextVector.resize(problem_dimension);
        for (unsigned i = 0; i < problem_dimension; ++i)
            contextVector[i] = population[individualsPerSubcomponent / 2][i];
    }
}


void Decomposer::reinitPopulation() {
    std::uniform_real_distribution<float> distribution(0.0, 1.0);
    for (int j = individualsPerSubcomponent / 2; j < individualsPerSubcomponent; j++) {
        for (int k = 0; k < coordinates.size(); ++k)
            population[j][k] = CCOptimizer.lowerLimit +
                               distribution(CCOptimizer.local_eng) * (CCOptimizer.upperLimit - CCOptimizer.lowerLimit);
    }
}


void Decomposer::allocateOptimizers() {
    for (int j = 0; j < optimizers.size(); ++j) {
        auto *rnd = new RandomEngine();
        rnd->seed(time(0));
        auto *optimizer = new SHADE(sizes[j], individualsPerSubcomponent, *this, *rnd);
        optimizer->setCoordinates(&coordinates[baseCoordIndex[j]], sizes[j]);
        optimizers[j] = optimizer;
        optimizer->loadIndividuals(population);
    }
}


void Decomposer::setOptimizerCoordinates(unsigned index) {
    optimizers[index]->setCoordinates(&coordinates[baseCoordIndex[index]], sizes[index]);
}


Decomposer::~Decomposer() {
    for (auto & optimizer : optimizers)
        delete optimizer;
}


void Decomposer::parallelLocalSearch(int maxIte, int maxParallelTrials, vector<float> &pfig,
                                     vector<unsigned> &coordinate_translator) {
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
    while (ite++ < maxIte) {
        vector<float> fy(maxParallelTrials, 0);
        vector<int> flip(maxParallelTrials, 0);

#pragma omp parallel
        {
            std::mt19937 rng(std::random_device{}());
#pragma omp for schedule(dynamic)
            for (int i = 0; i < maxParallelTrials; ++i) {
                vector<float> b(contextVector.size(), 0);
                float v = distribution(rng) * pfir[pfir.size() - 1];
                int c;
                for (c = 0; c < pfir.size() - 1; ++c)
                    if (v < pfir[c + 1])
                        break;
                vector<float> y = yc;
                if (y[c] > TH) {
                    y[c] = TH - 0.2;
                    flip[i] = -c;
                } else {
                    y[c] = TH + 0.2;
                    flip[i] = c;
                }
                fy[i] = CCOptimizer.computeFitnessValue(y);
            }
        }
        CCOptimizer.numberOfEvaluations += maxParallelTrials;
        int k = min_element(fy.begin(), fy.end()) - fy.begin();
        if (fy[k] < fc) {
            fc = fy[k];
            int c = flip[k];
            yc[abs(c)] = TH + 0.2 * abs(c) / c;
        }
    }
    cout << "end" << endl;
    if (fc < this->CCOptimizer.current_best_fitness) {
        cout << "From local search: " << 1 - CCOptimizer.current_best_fitness << " ===> " << 1 - fc << endl;
        contextVector = yc;
        CCOptimizer.current_best_fitness = fc;
    }
}
