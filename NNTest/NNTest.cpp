// NNTest.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "NNetwork.h"
#include "NNMatrix.h"
#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"

void convert_data(std::vector<std::pair<std::vector<double>, unsigned int>>& train_data, std::vector<std::pair<std::vector<double>, unsigned int>>& valid_data, mnist_data* data, unsigned int count)
{
	train_data.clear();
	std::pair<std::vector<double>, unsigned int> tData = std::make_pair(std::vector<double>(28 * 28), 0);
	train_data = std::vector<std::pair<std::vector<double>, unsigned int>>(50000, tData);
	valid_data = std::vector<std::pair<std::vector<double>, unsigned int>>(100, tData);
	for (int k = 0; k < 50100; k++)
	{
		for (int i = 0; i < 28; i++)
		{
			for (int j = 0; j < 28; j++)
			{
				if (k < 50000)
					train_data[k].first[i * 28 + j] = data[k].data[i][j];
				else
					valid_data[k-50000].first[i * 28 + j] = data[k].data[i][j];
			}
		}
		if (k < 50000)
			train_data[k].second = data[k].label;
		else
			valid_data[k - 50000].second = data[k].label;
	}

}

int _tmain(int argc, _TCHAR* argv[])
{
	
	mnist_data *data;
	unsigned int cnt;
	int ret;

	if (ret = mnist_load("../data/train-images.idx3-ubyte", "../data/train-labels.idx1-ubyte", &data, &cnt)) {
		printf("An error occured: %d\n", ret);
	}
	else {
		std::vector<std::pair<std::vector<double>, unsigned int>> train_data;
		std::vector<std::pair<std::vector<double>, unsigned int>> valid_data;
		printf("image count: %d\n", cnt);
		convert_data(train_data, valid_data, data, cnt);
		free(data);
		std::vector<std::size_t> layer = { 28 * 28, 30, 10 };
		NNetwork network(layer, ActivationFunction::Sigmoid, CostFunction::Quadratic);
		network.train(train_data, valid_data, 10, 10, 3.0);

	}
	
	

	int a;
	std::cin >> a;
	return 0;
}

