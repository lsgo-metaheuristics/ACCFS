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

#define _USE_MATH_DEFINES
#include "CCDE.h"
#include <thread>
#include <algorithm>    
#include <set>
#include <vector>
#include <random>
#include <map>
#include <omp.h>
#include <stdlib.h>     
#include <time.h>       
#include <iostream>      
#include <fstream>      
#include "KFoldCrossValidation.h"
#include "MinMaxScaler.h"
#include <cmath>
#include "matrix.h"


vector<matrix*> distances_buff;


using namespace std;
typedef vector<vector<float>> mat;

mat dataset_features;
vector<int> dataset_labels;

//Part of the original dataset used by the feature selection algorithnm
mat train_features;
vector<int> train_labels;
vector<mat> train_features_folds;
vector<vector<int>> train_labels_folds;
vector<mat> test_features_folds;
vector<vector<int>> test_labels_folds;

//Part of the original dataset used for evaluating the result of the feature selection algorithnm
mat evaluation_features;
vector<int> evaluation_labels;
vector<mat> evaluation_train_features_folds;
vector<vector<int>> evaluation_train_labels_folds;
vector<mat> evaluation_test_features_folds;
vector<vector<int>> evaluation_test_labels_folds;

float elapsedTimeFit = 0;
unsigned num_classes;
unsigned  problem_dimension = 1000;
string dataset_name = "";

enum  Dataset1
{
	LEUKEMIA_1 = 0, DLBCL, _9_TUMOR, BRAIN_TUMOR_1, PROSTATE_TUMOR, LEUKEMIA_2, BRAIN_TUMOR_2, LEUKEMIA_3,
	_11_TUMOR, LUNGC
};

enum Dataset2
{
	GrammaticalFacialExpression = 0, SemeionHandwrittenDigit, Isolet5, MultipleFeaturesDigit, HAPT, Har, UJIIndoorLoc, MadelonValid, OpticalRecognitionofHandWritten,
	ConnectionistBenchData, WDBC, LungCancer
};



int n_folds = 5;
int K = 3;
int K_EVAL = 3;
float w1 = 0.1;
float w2 = 0.1;
tFitness bestFitness = 0.0;
unsigned evaluations = 0;


/**
* \brief Reads a CSV dataset from a file and stores the features and labels in separate vectors.
*
* The CSV file should contain rows of float values terminated with an unsigned int label. The function
* reads each row and stores the feature values in a vector<float>, and the label in a vector<unsigned int>.
*
* \param filename The name of the CSV file to read.
* \param data A vector of vectors that will be filled with the feature values for each sample.
* \param labels A vector that will be filled with the corresponding labels for each sample.
*
* \throw std::runtime_error If the file could not be opened or read.
*
* \return The number of unique classes in the dataset.
*/
unsigned int read_csv(const std::string& filename, std::vector<std::vector<float>>& data, std::vector<int>& labels)
{
	std::ifstream file(filename);

	if (file)
	{
		std::string line;
		std::set<unsigned int> label_set; // Use set to keep track of unique labels
		while (std::getline(file, line))
		{
			std::vector<float> row;
			unsigned int label;
			std::stringstream ss(line);
			std::string value;
			while (std::getline(ss, value, ','))
			{
				if (!ss.eof())
					row.push_back(std::stod(value));
				else
				{
					label = std::stoi(value);
					label_set.insert(label); // Add label to set
				}
			}
			data.push_back(row);
			labels.push_back(label);
		}
		return label_set.size();
	}
	else
		throw std::runtime_error("Could not open file: " + filename);
}



/**
 * \brief Reads a tabular dataset from a file and stores the features and labels in separate vectors.
 *
 * The file should contain rows of double values separated by spaces, terminated with an unsigned int label.
 * The function reads each row and stores the feature values in a vector<double>, and the label in a vector<unsigned int>.
 *
 * \param filename The name of the file to read.
 * \param data A vector of vectors that will be filled with the feature values for each sample.
 * \param labels A vector that will be filled with the corresponding labels for each sample.
 *
 * \throw std::runtime_error If the file could not be opened or read.
 *
 * \return The number of unique classes in the dataset.
 */
unsigned int read_tabular_data(const std::string& filename, std::vector<std::vector<float>>& data, std::vector<int>& labels) 
{
	std::ifstream file(filename);

	if (file) {
		std::string line;
		std::set<unsigned int> label_set; // Use set to keep track of unique labels
		while (std::getline(file, line)) 
		{
			std::vector<float> row;
			unsigned int label;
			std::stringstream ss(line);
			std::string value;
			while (ss >> value) 
			{
				if (!ss.eof()) 
				{
					row.push_back(std::stod(value));
				}
				else 
				{
					label = std::stoi(value);
					label_set.insert(label); // Add label to set
				}
			}
			data.push_back(row);
			labels.push_back(label);
		}
		return label_set.size();
	}
	else 
	{
		throw std::runtime_error("Could not open file: " + filename);
	}
}


/**
 * \brief Stratified split of a dataset.
 *
 * Given a sample matrix, a label vector, and a split ratio, this function performs a stratified split
 * of the dataset into a training set and a test set, such that the proportion of samples from each class
 * is preserved in both sets. The training set contains a fraction of the samples given by the split ratio,
 * while the test set contains the remaining samples.
 *
 * \param features The feature vector of the dataset.
 * \param labels The label vector of the dataset.
 * \param split_ratio The split ratio between 0 and 1 indicating the fraction of samples to include in the
 * training set.
 * \param train_features Output parameter for the feature vector of the training set.
 * \param train_labels Output parameter for the label vector of the training set.
 * \param test_features Output parameter for the feature vector of the test set.
 * \param test_labels Output parameter for the label vector of the test set.
 *
 * \note The output vectors are cleared before being populated by the function.
 */
void stratified_split(const mat& samples, const std::vector<int>& labels, float split_ratio,
	mat& train_samples, std::vector<int>& train_labels,
	mat& test_samples, std::vector<int>& test_labels)
{
	// Count the number of samples for each class
	std::vector<int> class_count(*std::max_element(labels.begin(), labels.end()) + 1, 0);
	for (int label : labels)
		class_count[label]++;

	// Compute the number of samples to include in the train set for each class
	std::vector<int> train_class_count(class_count.size(), 0);
	for (size_t i = 0; i < class_count.size(); i++)
		train_class_count[i] = static_cast<int>(class_count[i] * split_ratio);

	// Initialize the train and test sets
	train_samples.clear();
	train_labels.clear();
	test_samples.clear();
	test_labels.clear();

	// Iterate over the samples and add them to the train or test set
	std::vector<int> train_count(class_count.size(), 0);
	for (size_t i = 0; i < samples.size(); i++)
	{
		int label = labels[i];
		if (train_count[label] < train_class_count[label])
		{
			// Add the sample to the train set
			train_features.push_back(samples[i]);
			train_labels.push_back(label);
			train_count[label]++;
		}
		else
		{
			// Add the sample to the test set
			test_samples.push_back(samples[i]);
			test_labels.push_back(label);
		}
	}
}



/**
* \brief Initializes the dataset for the specified problem.
*
* This function takes an input `Dataset1` enum, `d`, and initializes the
* dataset for the corresponding problem. The dataset is loaded from a .csv file
* based on the value of `d`. If the dataset is successfully loaded, the function
* splits the dataset into training and testing sets and stores the resulting
* matrices in the `train_data`, `train_labels`, `test_data`, and `test_labels`
* matrices. The function also initializes the `num_classes` variable based on the
* unique values in the combined `train_labels` and `test_labels` matrices.
*
* If the input `Dataset1` enum is not recognized, the function outputs an error
* message and exits with a status code of 1. If the .csv file for the specified
* dataset is not found, the function outputs an error message and exits with a
* status code of 1.
*
* \param d The `Dataset1` enum specifying the problem to be initialized.
*/
void initDataset1(Dataset1 d)
{
	string filename;

	dataset_features.clear();
	dataset_labels.clear();
	train_features.clear();
	train_labels.clear();
	evaluation_labels.clear();
	evaluation_features.clear();
	test_features_folds.clear();
	test_labels_folds.clear();
	train_features_folds.clear();
	train_labels_folds.clear();
	evaluation_test_features_folds.clear();
	evaluation_test_labels_folds.clear();
	evaluation_train_features_folds.clear();
	evaluation_train_labels_folds.clear();

	switch (d) {
	case LEUKEMIA_1:
		dataset_name = "LEUKEMIA_1";
		filename = "dataset1/leukemia_1.csv";
		break;
	case DLBCL:
		dataset_name = "DLBCL";
		filename = "dataset1/DLBCL.csv";
		break;
	case _9_TUMOR:
		dataset_name = "9_TUMOR";
		filename = "dataset1/9_tumor.csv";
		break;
	case BRAIN_TUMOR_1:
		dataset_name = "BRAIN_TUMOR_1";
		filename = "dataset1/brain_tumor_1.csv";
		break;
	case PROSTATE_TUMOR:
		dataset_name = "PROSTATE_TUMOR";
		filename = "dataset1/prostate_tumor_1.csv";
		break;
	case LEUKEMIA_2:
		dataset_name = "LEUKEMIA_2";
		filename = "dataset1/leukemia_2.csv";
		break;
	case BRAIN_TUMOR_2:
		dataset_name = "BRAIN_TUMOR_2";
		filename = "dataset1/brain_tumor_2.csv";
		break;
	case LEUKEMIA_3:
		dataset_name = "LEUKEMIA_3";
		filename = "dataset1/leukemia_3.csv";
		break;
	case _11_TUMOR:
		dataset_name = "_11_TUMOR";
		filename = "dataset1/11_tumor.csv";
		break;
	case LUNGC:
		dataset_name = "LUNGC";
		filename = "dataset1/lungc.csv";
		break;
	default:
		cout << "Unable to find problem" << endl;
		exit(1);
	}

	num_classes = read_csv(filename, dataset_features, dataset_labels);

	stratified_split(dataset_features, dataset_labels, 0.9,
		train_features, train_labels,
		evaluation_features, evaluation_labels);

	if (evaluation_features.size() < dataset_features.size() * 0.1)
	{
		stratified_split(dataset_features, dataset_labels, 0.85,
			train_features, train_labels, evaluation_features, evaluation_labels);

		if (evaluation_features.size() < dataset_features.size() * 0.1)
			stratified_split(dataset_features, dataset_labels, 0.80,
				train_features, train_labels, evaluation_features, evaluation_labels);
	}


	vector<int> indices;
	for (int i = 0; i < train_features.size(); ++i)
		indices.push_back(i);

	KFoldCrossValidation<int> kFoldCrossValidation = KFoldCrossValidation<int>(indices, n_folds, time(0));

	for (int i = 0; i < n_folds; ++i)
	{
		vector<int> testSample = kFoldCrossValidation.getTestFold(i);
		vector<int> trainSample = kFoldCrossValidation.getTrainFold(i);

		mat train_data_f;
		for (int i : trainSample) train_data_f.push_back(train_features[i]);

		mat test_data_f;
		for (int i : testSample) test_data_f.push_back(train_features[i]);

		vector<int> train_labels_f;
		for (int i : trainSample) train_labels_f.push_back(train_labels[i]);

		vector<int> test_labels_f;
		for (int i : testSample) test_labels_f.push_back(train_labels[i]);

		test_features_folds.push_back(test_data_f);
		test_labels_folds.push_back(test_labels_f);
		train_features_folds.push_back(train_data_f);
		train_labels_folds.push_back(train_labels_f);
	}

	indices.clear();
	for (int i = 0; i < evaluation_features.size(); ++i)
		indices.push_back(i);

	kFoldCrossValidation = KFoldCrossValidation<int>(indices, n_folds, time(0));

	for (int i = 0; i < n_folds; ++i)
	{
		vector<int> testSample = kFoldCrossValidation.getTestFold(i);
		vector<int> trainSample = kFoldCrossValidation.getTrainFold(i);

		mat train_data_f;
		for (int i : trainSample) train_data_f.push_back(evaluation_features[i]);

		mat test_data_f;
		for (int i : testSample) test_data_f.push_back(evaluation_features[i]);

		vector<int> train_labels_f;
		for (int i : trainSample) train_labels_f.push_back(evaluation_labels[i]);

		vector<int> test_labels_f;
		for (int i : testSample) test_labels_f.push_back(evaluation_labels[i]);

		evaluation_test_features_folds.push_back(test_data_f);
		evaluation_test_labels_folds.push_back(test_labels_f);
		evaluation_train_features_folds.push_back(train_data_f);
		evaluation_train_labels_folds.push_back(train_labels_f);
	}

	problem_dimension = train_features[0].size();
}


/**
* \brief Initializes the specified dataset and loads its training and testing data.
* \details	The function load the dataset and converts to zero-based indices the labels.
*			The dataset is already split into train and test.
*			Further k-fold split is performed for the purposes of the algorithm.
*			The scaler is fitted to the data and the data is transformed using the scaler.
*
* \param d - Enumeration value specifying which dataset to initialize.
*/
void initDataset2(Dataset2 d)
{
	string trainingdata_filename;
	string testingdata_filename;

	train_features.clear();
	train_labels.clear();
	evaluation_features.clear();
	evaluation_labels.clear();
	test_features_folds.clear();
	test_labels_folds.clear();
	train_features_folds.clear();
	train_labels_folds.clear();
	evaluation_test_features_folds.clear();
	evaluation_test_labels_folds.clear();
	evaluation_train_features_folds.clear();
	evaluation_train_labels_folds.clear();

	switch (d) {
	case GrammaticalFacialExpression:
		dataset_name = "grammatical facial expression";
		trainingdata_filename = "dataset2/trainingdata/grammatical_facial_expression01.txt";
		testingdata_filename = "dataset2/testingdata/grammatical_facial_expression01.txt";
		break;
	case SemeionHandwrittenDigit:
		dataset_name = "SemeionHandwrittenDigit";
		trainingdata_filename = "dataset2/trainingdata/SemeionHandwrittenDigit.txt";
		testingdata_filename = "dataset2/testingdata/SemeionHandwrittenDigit.txt";
		break;
	case Isolet5:
		dataset_name = "isolet5";
		trainingdata_filename = "dataset2/trainingdata/isolet5.txt";
		testingdata_filename = "dataset2/testingdata/isolet5.txt";
		break;
	case MultipleFeaturesDigit:
		dataset_name = "MultipleFeaturesDigit";
		trainingdata_filename = "dataset2/trainingdata/MultipleFeaturesDigit.txt";
		testingdata_filename = "dataset2/testingdata/MultipleFeaturesDigit.txt";
		break;
	case HAPT:
		dataset_name = "HAPTDataSet";
		trainingdata_filename = "dataset2/trainingdata/HAPTDataSet.txt";
		testingdata_filename = "dataset2/testingdata/HAPTDataSet.txt";
		break;
	case Har:
		dataset_name = "har";
		trainingdata_filename = "dataset2/trainingdata/har.txt";
		testingdata_filename = "dataset2/testingdata/har.txt";
		break;
	case UJIIndoorLoc:
		dataset_name = "UJIIndoorLoc";
		trainingdata_filename = "dataset2/trainingdata/UJIIndoorLoc_training.txt";
		testingdata_filename = "dataset2/testingdata/UJIIndoorLoc_validation.txt";
		break;
	default:
		dataset_name = "Invalid dataset";
		trainingdata_filename = "";
		testingdata_filename = "";
		break;
	}

	mat train_data_0;
	mat test_data_0;

	num_classes = max(read_tabular_data(trainingdata_filename, train_features, train_labels),
		read_tabular_data(testingdata_filename, evaluation_features, evaluation_labels));

	vector<int> all_labels(train_labels);
	std::copy(evaluation_labels.begin(), evaluation_labels.end(), std::back_inserter(all_labels));
	int min_label = *min_element(all_labels.begin(), all_labels.end());

	std::for_each(train_labels.begin(), train_labels.end(), [min_label](int& elem) {	elem -= min_label;	});
	std::for_each(evaluation_labels.begin(), evaluation_labels.end(), [min_label](int& elem) {	elem -= min_label;	});

	mat all_features(train_features);
	std::copy(evaluation_features.begin(), evaluation_features.end(), std::back_inserter(all_features));

	MinMaxScaler scaler;
	scaler.fit(all_features);
	scaler.transform(train_features);
	scaler.transform(evaluation_features);

	vector<int> indices;
	for (int i = 0; i < train_features.size(); ++i)	indices.push_back(i);

	KFoldCrossValidation<int> kFoldCrossValidation = KFoldCrossValidation<int>(indices, n_folds, time(0));
	for (int i = 0; i < n_folds; ++i)
	{
		vector<int> testSample = kFoldCrossValidation.getTestFold(i);
		vector<int> trainSample = kFoldCrossValidation.getTrainFold(i);

		mat train_data_f;
		for (int i : trainSample) train_data_f.push_back(train_features[i]);

		mat test_data_f;
		for (int i : testSample) test_data_f.push_back(train_features[i]);

		vector<int> train_labels_f;
		for (int i : trainSample) train_labels_f.push_back(train_labels[i]);

		vector<int> test_labels_f;
		for (int i : testSample) test_labels_f.push_back(train_labels[i]);

		test_features_folds.push_back(test_data_f);
		test_labels_folds.push_back(test_labels_f);
		train_features_folds.push_back(train_data_f);
		train_labels_folds.push_back(train_labels_f);
	}


	indices.clear();
	for (int i = 0; i < evaluation_features.size(); ++i) indices.push_back(i);

	kFoldCrossValidation = KFoldCrossValidation<int>(indices, n_folds, time(0));
	for (int i = 0; i < n_folds; ++i)
	{
		vector<int> testSample = kFoldCrossValidation.getTestFold(i);
		vector<int> trainSample = kFoldCrossValidation.getTrainFold(i);

		mat train_data_f;
		for (int i : trainSample) train_data_f.push_back(evaluation_features[i]);

		mat test_data_f;
		for (int i : testSample) test_data_f.push_back(evaluation_features[i]);

		vector<int> train_labels_f;
		for (int i : trainSample) train_labels_f.push_back(evaluation_labels[i]);

		vector<int> test_labels_f;
		for (int i : testSample) test_labels_f.push_back(evaluation_labels[i]);


		evaluation_test_features_folds.push_back(test_data_f);
		evaluation_test_labels_folds.push_back(test_labels_f);
		evaluation_train_features_folds.push_back(train_data_f);
		evaluation_train_labels_folds.push_back(train_labels_f);
	}

	problem_dimension = train_features[0].size();
}



/**
 * Computes the Euclidean distance between a reference point and a query point.
 *
 * \param ref          refence points by rows
 * \param query        query points by rows
 * \param ref_index    index to the reference point to consider
 * \param query_index  index to the query point to consider
 * \param active_features_flag vector of 0/1 for deactivate/activate the corresponding feature
 * \return computed distance
 */
float compute_distance(vector<vector<float>>& ref,
	vector<vector<float>>& query,
	int   ref_index,
	int   query_index,
	vector<int> &active_features_flag)
{
	float sum = 0.0;
	int dim = ref[0].size();
	int* pf = &(active_features_flag[0]);
	float* rd = &(ref[ref_index][0]);
	float* qd = &(query[query_index][0]);	
	for (int d = 0; d < dim; ++d)
	{						
		float x1 = *rd++;
		float x2 = *qd++;		
		if (*pf++ > 0)
		{
			float diff = x1 - x2;
			sum += diff * diff;
		}
	}	
	return sqrtf(sum);
}


/**
 * Computes the Euclidean distance between a reference point and a query point.
 *
 * \param ref          refence points by rows
 * \param query        query points by rows
 * \param ref_index    index to the reference point to consider
 * \param query_index  index to the query point to consider
  * \return computed distance
 */
float compute_distance(vector<vector<float>>& ref,
	vector<vector<float>>& query,
	int   ref_index,
	int   query_index)
{
	float sum = 0.0;
	int dim = ref[0].size();
	float* rd = &(ref[ref_index][0]);
	float* qd = &(query[query_index][0]);
	for (int d = 0; d < dim; ++d)
	{
		float diff = *rd++ - *qd++;
		sum += diff * diff;
	}
	return sqrtf(sum);
}



/**
 * Gathers at the beginning of the `dist` array the k smallest values and their
 * respective index (in the initial array) in the `index` array. After this call,
 * only the k-smallest distances and indexes are available.
 *
 * \param dist    vector containing the `length` distances
 * \param index   vector containing the index of the k smallest distances
 * \param k       number of smallest distances to locate
 */
void  k_insertion_sort(int length, float* dist, vector<int> &index, int k)
{
	// Initialise the first index
	index[0] = 0;

	// Go through all points
	for (int i = 1; i < length; ++i)
	{
		// Store current distance and associated index
		float curr_dist = dist[i];
		int   curr_index = i;

		// Skip the current value if its index is >= k and if it's higher the k-th already sorted smallest value
		if (i >= k && curr_dist >= dist[k - 1])
			continue;

		// Shift values (and indexes) higher that the current distance to the right
		int j = min(i, k - 1);
		while (j > 0 && dist[j - 1] > curr_dist)
		{
			dist[j] = dist[j - 1];
			index[j] = index[j - 1];
			--j;
		}

		// Write the current distance and index at their position
		dist[j] = curr_dist;
		index[j] = curr_index;
	}
}



/*
 * \brief For each input query point, locates the k-NN (indexes and distances) among the reference points.
 *
 * \param ref        refence points
 * \param query      query points
 * \param k          number of neighbors to consider
 * \param knn_dist   output array containing the query_nb x k distances
 * \param knn_index  output array containing the query_nb x k indexes
 * \param active_features_flag vector of 0/1 for deactivate/activate the corresponding feature
 */
void knn(vector<vector<float>>& ref,
	vector<vector<float>>& query,
	int k,
	vector<vector<float>>& knn_dist,
	vector<vector<int>>& knn_index,
	vector<int> &active_features_flag)
{
	int tid = omp_get_thread_num();

	matrix& distances = *distances_buff[tid];

	//compute all distances	
	if (&ref == &query )
	{
		for (int i = 0; i < query.size(); ++i)
			for (int j = 0; j < ref.size(); ++j)
				if (i == j)
					distances[i][j] = 0.0;
				else
					if (i > j)
						distances[i][j] = distances[j][i];
					else
						distances[i][j] = compute_distance(ref, query, j, i, active_features_flag);
	}
	else
	{
		for (int i = 0; i < query.size(); ++i)
			for (int j = 0; j < ref.size(); ++j)
				distances[i][j] = compute_distance(ref, query, j, i, active_features_flag);
	}

	knn_dist.resize(query.size());
	knn_index.resize(query.size());
	//Process one query point at a time
	for (int i = 0; i < query.size(); ++i)
	{
		vector<int> index(ref.size(), 0);
		std::iota(std::begin(index), std::end(index), 0);
	    k_insertion_sort(ref.size(), distances[i], index, k);
		index.resize(k);
		knn_dist[i].resize(k);
		for (int j = 0; j < k; ++j)
			knn_dist[i][j] = distances[i][j];
		knn_index[i] = index;
	}
}




/**
* \brief	Calculates the balanced accuracy of k - nearest neighbors classification.
* \details	The balanced accuracy is the average of the accuracy per class,
*			where accuracy for a class is defined as the number of correctly classified instances of that class
*			divided by the total number of instances of that class in the test data.
*			If the train data is empty, the function returns 0.
* \param train_data The training data as a matrix of features.
* \param train_labels The labels corresponding to the training data.
* \param test_data The testing data as a matrix of features.
* \param test_labels The labels corresponding to the testing data.
* \param k The number of neighbors to consider.
* \param active_features_flag vector of flags indicating active features
* \return Returns the balanced accuracy of the k - nearest neighbors classification.
*/
tFitness knn_balanced_accuracy(int K, mat& train_data, vector<int>& train_labels,
	mat& test_data, vector<int>& test_labels,
	vector<int> &active_features_flag)
{

	if (train_data.size() == 0)
		return 0.0;

	int k = K;
	bool loocv = false;
	if (&train_data == &test_data && &train_labels == &test_labels)
	{
		loocv = true; //use leave-one-out cv	
		k = k + 1;
	}		
	vector<vector<float>> knn_dist;
	vector<vector<int>> knn_index;
	knn(train_data, test_data, k, knn_dist, knn_index, active_features_flag);


	vector<int> total(num_classes, 0);
	vector<int> correct(num_classes, 0);
	mat confusion_matrix(num_classes, std::vector<float>(num_classes, 0));	
	for (int i = 0; i < test_data.size(); ++i)
	{
		unsigned actual_lab = test_labels[i];
		vector<int> counts(num_classes, 0);
		for (int j = 0; j < k; ++j)
		{
			unsigned neigh_index = knn_index[i][j];
			if (loocv && neigh_index == i) continue;			
			unsigned neigh_label = train_labels[neigh_index];
			counts[neigh_label]++;
		}
		auto max_it = std::max_element(counts.begin(), counts.end());
		unsigned predicted_lab = std::distance(counts.begin(), max_it);
		if (predicted_lab == actual_lab)
		{
			correct[actual_lab]++;			
			confusion_matrix[predicted_lab][actual_lab]++;
		}
		total[actual_lab]++;
	}

	float cvAcc = 0.0;
	int actual_num_classes = 0;
	for (int i = 0; i < num_classes; ++i)
	{
		if (total[i] > 0)
		{
			cvAcc += float(correct[i]) / float(total[i]);
			actual_num_classes++;
		}
	}
	cvAcc /= actual_num_classes;

	return cvAcc;
}




/**
 * \brief Computes the accuracy of KNN classification using the given train and test data
 *
 * \param train_data Matrix containing the training data
 * \param train_labels Row vector containing the labels for the training data
 * \param test_data Matrix containing the test data
 * \param test_labels Row vector containing the labels for the test data
 * \param k Number of nearest neighbors to consider
 * \param active_features_flag vector of flags indicating active features
 *
 * \return The accuracy of the KNN classification as a fraction of correctly classified instances
 */
tFitness knn_accuracy(int K, mat& train_data, vector<int>& train_labels,
	mat& test_data, vector<int>& test_labels,
	vector<int> &active_features_flag)
{
	int k = K;
	bool loocv = false;
	if (&train_data == &test_data && &train_labels == &test_labels)
	{
		loocv = true; //use leave-one-out cv	
		k = k + 1;
	}
	
	vector<vector<float>> knn_dist;
	vector<vector<int>> knn_index;
	knn(train_data, test_data, k, knn_dist, knn_index, active_features_flag);

	vector<int> total(num_classes, 0);
	vector<int> correct(num_classes, 0);
	mat confusion_matrix(num_classes, std::vector<float>(num_classes, 0));
	unsigned total_correct = 0;

	for (int i = 0; i < test_data.size(); ++i)
	{
		unsigned actual_lab = test_labels[i];
		vector<int> counts(num_classes, 0);
		for (int j = 0; j < k; ++j)
		{
			unsigned neigh_index = knn_index[i][j];
			if (loocv && neigh_index == i) continue;			
			unsigned neigh_label = train_labels[neigh_index];
			counts[neigh_label]++;
		}

		// Get the index of the maximum element
		auto max_it = std::max_element(counts.begin(), counts.end());
		unsigned predicted_lab = std::distance(counts.begin(), max_it);		
		if (predicted_lab == actual_lab)
		{
			total_correct++;
			confusion_matrix[predicted_lab][actual_lab]++;
		}
	}	
	return ((float)total_correct) / test_data.size();
}




tFitness evaluate_fitness_on_dataset(int dim, float* x,
	mat& train_dataset,
	vector<int>& train_labels,
	mat& test_dataset,
	vector<int>& test_labels)
{
	++evaluations;
	tFitness f = 0;
	int k = K;
	bool loocv = false;

	if (&test_dataset == &train_dataset && &test_labels == &train_labels)
	{
		loocv = true; //use leave-one-out cv	
		k = k + 1;
	}

	//Set the flags of active features using the threshold 	
	vector<int> active_features_flag(dim, 1);
	int realDim = dim;
	for (int i = 0; i < dim; ++i)
		if (x[i] < TH)
		{
			active_features_flag[i] = 0.0;
			realDim--;
		}
	
	vector<vector<float>> knn_dist;
	vector<vector<int>> knn_index;
	knn(train_dataset, test_dataset, k, knn_dist, knn_index, active_features_flag);

	//Computes metrics	
	float dist_same_label = 0, dist_diff_label = 0;
	unsigned n_same_label = 0, n_diff_label = 0;
	vector<unsigned> total(num_classes, 0);
	vector<int> correct(num_classes, 0);	

	for (int i = 0; i < test_dataset.size(); ++i)
	{
		unsigned actual_lab = test_labels[i];
		vector<int> count_neigh_labels(num_classes, 0);
		for (int j = 0; j < k; ++j)
		{
			unsigned neigh_index = knn_index[i][j];

			if (loocv && neigh_index == i) continue;

			float neigh_dist = knn_dist[i][j];
			unsigned neigh_label = train_labels[neigh_index];
			count_neigh_labels[neigh_label]++;
			if (neigh_label == actual_lab)
			{
				dist_same_label += neigh_dist;
				n_same_label++;
			}
			else
			{
				dist_diff_label += neigh_dist;
				n_diff_label++;
			}
		}

		// Get the index of the maximum element
		auto max_it = std::max_element(count_neigh_labels.begin(), count_neigh_labels.end());
		unsigned predicted_lab = std::distance(count_neigh_labels.begin(), max_it);

		if (predicted_lab == actual_lab)
		{
			correct[actual_lab]++;			
		}
		total[actual_lab]++;
	}

	// if the number of neighbours with a different label is zero 
	// then use the maximum possible distance in a space with n_rows features
	// (note that all features are normalized in [0, 1]
	if (n_diff_label)
		dist_diff_label /= n_diff_label;
	else
		dist_diff_label = sqrt(train_dataset.size());

	if (n_same_label)
		dist_same_label /= n_same_label;
	else
		dist_same_label = sqrt(train_dataset.size());

	float cvAcc = 0.0;
	int actual_num_classes = 0;
	for (int i = 0; i < num_classes; ++i)
	{
		if (total[i] > 0)
		{
			cvAcc += float(correct[i]) / float(total[i]);
			actual_num_classes++;
		}
	}
	cvAcc /= actual_num_classes;
	float srdim = sqrt(realDim);
	f = (1.0 - w1 - w2) * cvAcc + w1 * (1 - dist_same_label / srdim)  + w2 * dist_diff_label / srdim;	
	return 1.0 - f;
}



/**
 * \brief Evaluates the fitness of a given solution on training and testing datasets.
 * \param dim The dimension of the solution.
 * \param x The solution to be evaluated.
 * \return The fitness of the solution.
 */
tFitness evaluate_on_dataset(int dim, float* x)
{
	//Uses leave-one-out cross validation
	return evaluate_fitness_on_dataset(dim, x, train_features, train_labels, train_features, train_labels);
}


/**
 * \brief Evaluates the fitness of a given solution on the training dataset with k-fold cross-validation.
 * \param dim The dimension of the solution.
 * \param x The solution to be evaluated.
 * \return The average fitness across all folds of cross-validation.
 */
tFitness evaluate_on_dataset_cv(int dim, float* x)
{
	float f = 0.0;
	for (int fold = 0; fold < train_features_folds.size(); fold++)
	{
		f += evaluate_fitness_on_dataset(dim, x, train_features_folds[fold], train_labels_folds[fold],
			test_features_folds[fold], test_labels_folds[fold]);
	}
	return f / train_features_folds.size();
}

float evaluate_full_features()
{
	int K = K_EVAL;
	cout << "==================== EVALUATION ON KNN (K=" << K << ") ====================" << endl;
	cout << "Training and testing with the full set of " << problem_dimension << " features" << endl;
	cout << "Training with " << train_features.size() << " observations" << endl;
	cout << "Testing with  " << train_features.size() << " observations" << endl;
	vector<int> aff = vector<int>(problem_dimension, 1);
	float cvAccFull = knn_accuracy(K, train_features, train_labels, train_features, train_labels, aff);
	cout << "Accuracy: " << cvAccFull << endl;
	return cvAccFull;
}


/**
* \brief Evaluates a solution for feature selection using KNN classification algorithm (k-fold cv version)
*
*		This function evaluates the performance of a feature selection solution using the KNN classification algorithm.
*		It computes the accuracy of the solution on the full set of features, the optimized set of features for the training set,
*		and the optimized set of features for the test set. It then writes the elapsed time, number of function evaluations,
*		number of active features, and the accuracies of the three evaluations to a file.
*       This version is only reasonably useful when the evaluation set is big enough
*
*		\param x A vector of doubles representing the feature selection solution
*		\param fName A char pointer to the filename of the file to write the results to
*		\param etime A float representing the elapsed time
*		\param nfe An integer representing the number of function evaluations
*/
void evaluate_solution_fold(vector<float> x, char* fName, float etime, int nfe)
{
	FILE* fileOpt;
	fopen_s(&fileOpt, fName, "at");

	int nf = x.size();
	vector<int> active_features_flag(x.size(), 1);

	cout << "==================== EVALUATION ON KNN (K=" << K << ") ====================" << endl;

	cout << "Training and testing with the full set of " << problem_dimension << " features" << endl;
	float cvAccFull = 0;
	for (int fold = 0; fold < evaluation_train_features_folds.size(); ++fold)
	{
		cout << "Training with " << evaluation_train_features_folds[fold].size() << " observations" << endl;
		cout << "Testing with  " << evaluation_test_features_folds[fold].size() << " observations" << endl;
		cvAccFull += knn_accuracy(K, evaluation_train_features_folds[fold], evaluation_train_labels_folds[fold],
			evaluation_test_features_folds[fold], evaluation_test_labels_folds[fold],
			active_features_flag);
	}
	cvAccFull /= evaluation_train_features_folds.size();
	cout << "Accuracy: " << cvAccFull << endl;


	for (int i = 0; i < x.size(); ++i)
		if (x[i] < TH)
		{
			active_features_flag[i] = 0.0;
			nf--;
		}

	cout << endl << "TRAIN SET: Training and testing with an optimized set of " << nf << " features" << endl;
	float cvAccTrain = 0;
	for (int fold = 0; fold < evaluation_train_features_folds.size(); ++fold)
	{
		cout << "Training with " << train_features_folds[fold].size() << " observations" << endl;
		cout << "Testing with  " << test_features_folds.size() << " observations" << endl;
		cvAccTrain += knn_accuracy(K, train_features_folds[fold], train_labels_folds[fold],
			test_features_folds[fold], test_labels_folds[fold],
			active_features_flag);
	}
	cvAccTrain /= evaluation_train_features_folds.size();
	cout << "Accuracy on train set: " << cvAccTrain << endl;

	cout << endl << "TEST SET: Training and testing with an optimized set of " << nf << " features" << endl;
	float cvAccTest = 0;
	for (int fold = 0; fold < evaluation_train_features_folds.size(); ++fold)
	{
		cout << "Training with " << evaluation_train_features_folds[fold].size() << " observations" << endl;
		cout << "Testing with  " << evaluation_test_features_folds[fold].size() << " observations" << endl;
		cvAccTest += knn_accuracy(K, evaluation_train_features_folds[fold], evaluation_train_labels_folds[fold],
			evaluation_test_features_folds[fold], evaluation_test_labels_folds[fold],
			active_features_flag);
	}
	cvAccTest /= evaluation_train_features_folds.size();
	cout << "Accuracy on test set: " << cvAccTest << endl;
	fprintf(fileOpt, "%.2lf; %d; %d; %.4lf; %.4lf; %.4lf\n", etime, nfe, nf, cvAccFull, cvAccTrain, cvAccTest);

	fclose(fileOpt);
}



/**
* \brief Evaluates a solution for feature selection using KNN classification algorithm
*
*		This function evaluates the performance of a feature selection solution using the KNN classification algorithm.
*		It computes the accuracy of the solution on the full set of features, the optimized set of features for the training set,
*		and the optimized set of features for the test set. It then writes the elapsed time, number of function evaluations,
*		number of active features, and the accuracies of the three evaluations to a file.
*
*		\param x A vector of doubles representing the feature selection solution
*		\param fName A char pointer to the filename of the file to write the results to
*		\param etime A float representing the elapsed time
*		\param nfe An integer representing the number of function evaluations
*/
void evaluate_solution(vector<float> x, char* fName, float etime, int nfe)
{	
	srand(time(nullptr));
	vector<size_t> predictions;

	FILE* fileOpt;
	fopen_s(&fileOpt, fName, "at");

	int nf = x.size();
	vector<int> active_features_flag(x.size(), 1);

	int seed = rand() % 100;
	cout << "seed =" << seed << endl;
	int K = K_EVAL;
	cout << "==================== EVALUATION ON KNN (K=" << K << ") ====================" << endl;
	cout << "Training and testing with the full set of " << problem_dimension << " features" << endl;
	cout << "Training with " << train_features.size() << " observations" << endl;
	cout << "Testing with  " << evaluation_features.size() << " observations" << endl;
	float cvAccFull = knn_accuracy(K, train_features, train_labels, evaluation_features, evaluation_labels, active_features_flag);
	cout << "Accuracy: " << cvAccFull << endl;

	for (int i = 0; i < x.size(); ++i)
		if (x[i] < TH)
		{
			active_features_flag[i] = 0.0;
			nf--;
		}

	cout << endl << "TRAIN SET: Training and testing with an optimized set of " << nf << " features" << endl;
	cout << "Training with " << train_features.size() << " observations" << endl;
	cout << "Testing with  " << train_features.size() << " observations" << endl;
	float cvAccTrain = knn_accuracy(K, train_features, train_labels, train_features, train_labels, active_features_flag);
	cout << "Accuracy on train set: " << cvAccTrain << endl;

	cout << endl << "TEST SET: Training and testing with an optimized set of " << nf << " features" << endl;
	cout << "Training with " << train_features.size() << " observations" << endl;
	cout << "Testing with  " << evaluation_features.size() << " observations" << endl;
	float cvAccTest = knn_accuracy(K, train_features, train_labels, evaluation_features, evaluation_labels, active_features_flag);
	cout << "Accuracy on test set: " << cvAccTest << endl;

	fprintf(fileOpt, "%.2lf; %d; %d; %.4lf; %.4lf; %.4lf\n", etime, nfe, nf, cvAccFull, cvAccTrain, cvAccTest);
	fclose(fileOpt);
}




/**
 * \brief Calculates the permutation feature importance for the model.
 *
 * This function calculates the permutation feature importance by randomly permuting
 * the values of a single feature and observing the change in the model's performance.
 *
 * \return A vector of doubles representing the permutation feature importances for each feature.
 */
vector<float> permutationFeatureImportance()
{
	cout << "Computing permutation feature importance....";
	vector<float> feature_imp;
	
	//fitness with all features
	vector<float> x(problem_dimension, 1);
	tFitness f0 = evaluate_on_dataset(problem_dimension, &x[0]);
	//evaluate_fitness_on_dataset(dim, x, train_features, train_labels, train_features, train_labels);
	auto rng = std::default_random_engine{};
	
	for (int i = 0; i < problem_dimension; ++i)
	{
		//permutate randomly i-th feature on dataset
		vector<float> feature_i_train0(train_features.size(), 0.0);
		vector<float> feature_i_train(train_features.size(), 0.0);
		for (int j = 0; j < train_features.size(); ++j)
			feature_i_train0[j] = feature_i_train[j] = train_features[j][i];

		shuffle(feature_i_train.begin(), feature_i_train.end(), rng);

		for (int j = 0; j < train_features.size(); ++j)
			train_features[j][i] = feature_i_train[j];

		tFitness fi = evaluate_on_dataset(problem_dimension, &x[0]);
		//evaluate_fitness_on_dataset(dim, x, train_features, train_labels, train_features, train_labels);

		feature_imp.push_back(fi - f0);

		//set back the i-th feature values		
		for (int j = 0; j < train_features.size(); ++j)
			train_features[j][i] = feature_i_train0[j];
	}

	float min_pfi = *min_element(feature_imp.begin(), feature_imp.end());
	for (int k = 0; k < problem_dimension; ++k)
		feature_imp[k] -= min_pfi;
	float max_pfi = *max_element(feature_imp.begin(), feature_imp.end());
	for (int k = 0; k < problem_dimension; ++k)
	{
		feature_imp[k] /= max_pfi;
	}
	cout << "done" << endl;
	return feature_imp;
}



int toInt(string s)
{
	try 
	{
		size_t pos;
		return stoi(s, &pos);		
	}
	catch (std::invalid_argument const& ex) {
		std::cerr << "Invalid number: " << s << '\n';
	}
	catch (std::out_of_range const& ex) {
		std::cerr << "Number out of range: " << s << '\n';
	}
}

void optimization(int argc, char* argv[])
{		
	unsigned numItePerCycle = 5;
	unsigned numRep = 1;
	unsigned numThreads = 12;
	unsigned numberOfEvaluations = 1.0E06;
	unsigned int sizeOfSubcomponents = 100;	
	int numOfIndividuals = 15; 
	sizeOfSubcomponents = 100;
	int d_index = BRAIN_TUMOR_1;
	d_index = MultipleFeaturesDigit;
	int suite = 1;	
	
	if (argc == 4)
	{
		suite = toInt(argv[1]); //Test suite in {1,2}
		d_index = toInt(argv[2]); //dataset in {1,2,...}
		numThreads = toInt(argv[3]); //num threads		
	}
			
	if( suite==1 )
		initDataset1((Dataset1)d_index);
	else
	if (suite == 2)
		initDataset2((Dataset2)d_index);
	else
	{
		cout << "Unspecified test suite";
		exit(1);
	}

	cout << "DATSET: " << dataset_name << endl;
	cout << "Train samples = " << train_features.size() << endl;
	cout << "Test samples = " << evaluation_features.size() << endl;	

	dataset_name.append("_");
	
	for (int i = 0; i < numThreads; ++i)
		distances_buff.push_back(new matrix(train_features.size(), train_features.size()));

	vector<int> seeds;
	unsigned maxNumRep = 100;
	for (unsigned i = 0; i < maxNumRep; ++i)
		seeds.push_back(i);

	cout << "Number of features = " << problem_dimension << endl;
	cout << "Number of labels = " << num_classes << endl;
	cout << "Using " << numThreads << " threads" << endl;
	cout << "Number of iterations per cycle = " << numItePerCycle << endl;
	cout << "Number of individuals per subpopulation = " << numOfIndividuals << endl;

	char fNameFullFeatures[256];
	char fNameOptimizedFeatures[256];
	FILE* file;
	sprintf_s(fNameFullFeatures, "results_full_features_%s.csv", dataset_name.c_str());
	sprintf_s(fNameOptimizedFeatures, "results_optimized_features_%s.csv", dataset_name.c_str());
	fopen_s(&file, fNameFullFeatures, "wt"); fclose(file);
	fopen_s(&file, fNameOptimizedFeatures, "wt"); fclose(file);
	
	vector< vector<ConvPlotPoint> > convergences;
	
	//left empty in this implementation /apply random grouping)
	vector<set<unsigned>> decomposition;

	for (unsigned k = 0; k < numRep; ++k)
	{
		vector<ConvPlotPoint> convergence;
		CCDE ccde;
		int seed = seeds[k];

		vector<float> pfi = permutationFeatureImportance();

		elapsedTimeFit = 0;
		clock_t startTimeCycle = clock();

		ccde.optimize(evaluate_on_dataset_cv, problem_dimension, 0.0, 1.0, numberOfEvaluations,
			sizeOfSubcomponents, numOfIndividuals, convergence, seed, numItePerCycle,
			numThreads, decomposition, pfi);

		clock_t stopTimeCycle = clock();
		float elapsedTimeCycle = ((float)(stopTimeCycle - startTimeCycle)) / CLOCKS_PER_SEC;
		cout << "Time spent in optimization = " << elapsedTimeCycle << endl;

		if (convergence.size() > 0)
			convergences.push_back(convergence);

		evaluate_solution(ccde.current_solution, fNameOptimizedFeatures, ccde.elapsedTime, ccde.numberOfEvaluations);

		if (numRep > 1)
			initDataset1((Dataset1)d_index);
	}	

	if (convergences.size() == 0) return;
	if (convergences[0].size() == 0) return;
	char fName[256];
	sprintf_s(fName, "convplot_%s_dec%d_popsize%d.csv", dataset_name.c_str(), sizeOfSubcomponents, numOfIndividuals);

	fopen_s(&file, fName, "wt");
	vector<ConvPlotPoint> averageConvergence;

	int maxSize = 0;
	int idOfMaxSize = 0;
	for (unsigned q = 0; q < convergences.size(); ++q)
		if (convergences[q].size() > maxSize)
		{
			maxSize = convergences[q].size();
			idOfMaxSize = q;
		}

	for (unsigned q = 0; q < maxSize; ++q)
	{
		tFitness f = 0.0;
		float nfeatures = 0;
		float nsubs = 0;
		vector<bool> found(numRep, false);
		for (unsigned k = 0; k < numRep; ++k)
		{
			if (q < convergences[k].size())
			{
				fprintf(file, "%d; %.8e; %d; %d;", convergences[k][q].nfe, convergences[k][q].f, convergences[k][q].numFeatures, convergences[k][q].numberOfSubcomponents);
				if (!found[k])
					if (convergences[k][q].nfe > numberOfEvaluations / 2)
					{
						found[k] = true;
					}
			}
			else
			{
				fprintf(file, "; ; ; ;");
			}


			//trova il più vicino
			int closerIndex = 0;
			float dist = 1.0E10;
			for (int j = 0; j < convergences[k].size(); j++)
				if (fabs((float)(((int)convergences[k][j].nfe) - ((int)convergences[idOfMaxSize][q].nfe))) < dist)
				{
					dist = fabs((float)(((int)convergences[k][j].nfe) - ((int)convergences[idOfMaxSize][q].nfe)));
					closerIndex = j;
				}
			f += convergences[k][closerIndex].f;
			nfeatures += convergences[k][closerIndex].numFeatures;
			nsubs += convergences[k][closerIndex].numberOfSubcomponents;
		}

		fprintf(file, "%d;", convergences[idOfMaxSize][q].nfe);
		fprintf(file, "%.8e;", f / numRep);
		fprintf(file, "%.8e;", nfeatures / numRep);
		fprintf(file, "%.8e;", nsubs / numRep);
		fprintf(file, "\n");

		averageConvergence.push_back(ConvPlotPoint(convergences[idOfMaxSize][q].nfe, f / numRep, nfeatures / numRep, nsubs / numRep));
	}

	char fNameAverageConvergences[256];
	sprintf_s(fNameAverageConvergences, "avg_convplot_%s_dec%d_popsize%d.csv", dataset_name.c_str(), sizeOfSubcomponents, numOfIndividuals);


	fopen_s(&file, fNameAverageConvergences, "wt");
	for (unsigned q = 0; q < maxSize; ++q)
	{
		if (q < averageConvergence.size())
			fprintf(file, "%d; %.3e; %d; %d;", averageConvergence[q].nfe, averageConvergence[q].f, averageConvergence[q].numFeatures, averageConvergence[q].numberOfSubcomponents);
		else
			fprintf(file, "; ; ");
		fprintf(file, "\n");
	}
	fclose(file);

	cout << "Time spent in fitness = " << elapsedTimeFit << endl;
}



int main(int argc, char* argv[])
{
	optimization(argc, argv);	
}

