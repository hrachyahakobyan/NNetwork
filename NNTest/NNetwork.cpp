#include "stdafx.h"
#include "NNetwork.h"

std::default_random_engine gen;
std::normal_distribution<double> distribution;

double random()
{
	return distribution(gen);
}

double NNetwork::log(double a)
{
	if (a <= 0)
		return 0;
	return std::log(a);
}

/* Activation functions and their derivatives*/

double NNetwork::sigmoid(double a)
{
	return 1 / (1 + std::exp(-a));
}

double NNetwork::sigmoid_prime(double a)
{
	return sigmoid(a)*(1 - sigmoid(a));
}

/*Cost functions and their derivatives */

double NNetwork::cost_quad(const Matrix& a, const int y)
{
	return 0.5 * (a - double(y)).norm(false);
}

double NNetwork::cost_cross(const Matrix& a, const int y)
{
	double sum = 0;
	for (std::size_t i = 0; i < a.cols(); i++)
		sum += -y * log(a(0, i)) - (1 - y) * log(1 - a(0, i));
	return sum;
}

NNetwork::Matrix NNetwork::cost_deriv_quad(const Matrix& z, const Matrix& a, const int y)
{
	return (a- y) * applyFunction(z, active_prime_);
}

NNetwork::Matrix NNetwork::cost_deriv_cross(const Matrix& z, const Matrix& a, const int y)
{
	return (a - double(y));
}

// Applies the given transformation to each element of the matrix
NNMatrix<double> NNetwork::applyFunction(const NNMatrix<double>& matrix, Active_Func f)
{
	NNMatrix<double> out(matrix.rows(), matrix.cols());
	for (std::size_t i = 0; i < matrix.rows(); i++)
		for (std::size_t j = 0; j < matrix.cols(); j++)
			out(i, j) = (this->*f)(matrix(i, j));
	return out;
}


NNetwork::NNetwork(const std::vector<std::size_t>& layers, ActivationFunctionType activType, CostType costType, RegularizationType regType) : layers_(layers),
nabla_b_back_(layers.size() - 1), nabla_w_back_(layers.size() - 1), activations_(layers.size()), zs_(layers.size() - 1)
{
	// Prepare the biases. The first layer is assumed not to have biases as it is the input layer
	// Biases are Nx1 vectors, where N is the number of neurons in the given layer
	for (std::size_t i = 1; i < layers.size(); i++)
	{
		biases_.push_back(NNMatrix<double>(layers[i], 1, &random));
	}
	// Prepare the weight matrices. A weight matrice is a NxM matrix of doubles, where
	// N is the number of the neurons in the NEXT layer and M is the number of neurons in the
	// PREVIOUS layer. There are L-1 such matrices, where L is the number of layers
	for (std::size_t i = 0; i < layers.size() - 1; i++)
	{
		weights_.push_back(NNMatrix<double>(layers[i + 1], layers[i], &random));
	}

	activations_[0] = NNMatrix<double>(layers_[0], 1);

	// Prepare the activation and its derivative
	switch (activType)
	{
		case ActivationFunctionType::Sigmoid:
		{
			active_prime_ = &NNetwork::sigmoid_prime;
			active_ = &NNetwork::sigmoid;
		}
		break;
		default:
			break;
	}

	switch (costType)
	{
		case CostType::CrossEntropy:
		{
			cost_ = &NNetwork::cost_cross;
			cost_deriv_ = &NNetwork::cost_deriv_cross;
		}
		break;
		case CostType::Quadratic:
		{
			cost_ = &NNetwork::cost_quad;
			cost_deriv_ = &NNetwork::cost_deriv_quad;
		}
		break;
		default:
			break;

	}

}

NNetwork::~NNetwork()
{

}


// Prepares the input for the layer given the input from the previous layer, the biases and the weight matrix.
NNMatrix<double> NNetwork::prepare_input(const NNMatrix<double>& input, const NNMatrix<double>& biases, const NNMatrix<double>& weights)
{
	return (weights.dot(input) + biases);
}

// Activates the given later given the input from the previous layer.
// Step1. prepares the input
// Step2. activates the input by applying the activation function
NNMatrix<double> NNetwork::activate(std::size_t layer, const NNMatrix<double>& input)
{
	// The layers are indexed starting from 0
	NNMatrix<double> prepared_input = prepare_input(input, biases_[layer - 1], weights_[layer - 1]);
	return applyFunction(prepared_input, active_);
}


// Feed the given input to the neural network and return the output from the last layer
NNMatrix<double> NNetwork::feed_forward(const NNMatrix<double>& input)
{
	NNMatrix<double> output = input;
	for (std::size_t layer = 0; layer < biases_.size(); layer++)
	{
		output = activate(layer + 1, output);
	}
	return output;
}


// The main function of training using SGD. Divides the train data into separata mini batches
// of size batch_size and feeds to update_mini_batch
void NNetwork::train(std::vector<TrainInput>& trainData,const std::vector<TrainInput>& val_data,std::size_t epochs, std::size_t batch_size, double eta)
{
	for (std::size_t epoch = 0; epoch < epochs; epoch++)
	{
		std::cout << "Epoch " << epoch << std::endl;
		std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
		std::vector<std::vector<TrainInput>> mini_batches;
		break_trainData(trainData, batch_size, mini_batches);
		std::vector<std::vector<TrainInput>>::const_iterator it;
		for (std::size_t mb = 0; mb < mini_batches.size(); mb++)
		{
			std::cout << "Mini batch " << mb << std::endl;
			std::chrono::high_resolution_clock::time_point tt1 = std::chrono::high_resolution_clock::now();
			update_mini_batch(mini_batches[mb], eta);
			std::chrono::high_resolution_clock::time_point tt2 = std::chrono::high_resolution_clock::now();
			double dur = std::chrono::duration_cast<std::chrono::microseconds>(tt2 - tt1).count();
			std::cout << "Time " << dur / 1000000 << '\n';
		}
		if (val_data.size() > 0)
		{
			int correct = 0;
			evaluate(val_data, correct);
			std::cout << "Epoch: " << epoch << correct << " out of " << val_data.size() << std::endl;
		}
		std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
		double dur = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		std::cout << "Time " << dur / 1000000 << '\n';

	}
}

void NNetwork::evaluate(const std::vector<TrainInput>& val_data, int& correct)
{
	correct = 0;
	for (std::size_t i = 0; i < val_data.size(); i++)
	{
		NNMatrix<double> result = feed_forward(NNMatrix<double>(val_data[i].first.size(), 1, val_data[i].first));
		std::pair<std::size_t, std::size_t> approx_result = result.max_index();
		if (approx_result.first == val_data[i].second)
			correct++;
	}
}

void NNetwork::break_trainData(std::vector<TrainInput>& trainData, std::size_t batch_size, std::vector<std::vector<TrainInput>>& mini_batches)
{
	std::random_shuffle(trainData.begin(), trainData.end());
	std::size_t mini_batches_count = trainData.size() / batch_size;
	mini_batches = std::vector<std::vector<TrainInput>>(mini_batches_count, std::vector<TrainInput>(batch_size));
	std::size_t j = 0;
	for (std::size_t i = 0; i < mini_batches_count; i++)
	{
		for (std::size_t j = 0; j < batch_size; j++)
		{
			mini_batches[i][j] = trainData[i * batch_size + j];
		}
	}
}

void NNetwork::update_mini_batch(const std::vector<TrainInput>& trainData, double eta)
{
	auto nabla_b = biases_;
	auto nabla_w = weights_;

	for (std::size_t i = 0; i < trainData.size(); i++)
	{
		std::cout << "Train data " << i << std::endl;
		std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
		back_prop(trainData[i]);
		std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
		double dur = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		std::cout << "Back_prop time " << dur / 1000000 << '\n';
		std::cout << "Matrix time " << NNMatrix<double>::opTime << '\n';
		NNMatrix<double>::opTime = 0;
		for (std::size_t i = 0; i < nabla_b_back_.size(); i++)
		{
			nabla_b[i] += nabla_b_back_[i];
			nabla_w[i] += nabla_w_back_[i];
		}
	}

	for (std::size_t i = 0; i < nabla_b.size(); i++)
	{
		weights_[i] = weights_[i] - nabla_w[i] * (double(eta / trainData.size()));
		biases_[i] = biases_[i] - nabla_b[i] * (double(eta / trainData.size()));
	}
}


// Back propagation algorithm. Given a single input X and output Y, updates the layer by layer
// derivatives of biases and weights of the cost function
void NNetwork::back_prop(const TrainInput& trainInput)
{
	//std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
	// The activation of the first layer of the neurons, i.e. the input
	activations_[0].assign(trainInput.first);
	// Feeding forward
	for (std::size_t layer = 0; layer < biases_.size(); layer++)
	{
		// The prepared input of the current layer
		//std::chrono::high_resolution_clock::time_point tt1 = std::chrono::high_resolution_clock::now();
		zs_[layer] = prepare_input(activations_[layer], biases_[layer], weights_[layer]);
		//std::chrono::high_resolution_clock::time_point tt2 = std::chrono::high_resolution_clock::now();
		//double durt = std::chrono::duration_cast<std::chrono::microseconds>(tt2 - tt1).count();
		//std::cout << "Back_prop prepare input time " << durt / 1000000 << '\n';
		// Apply transformation to the preapred input
		//std::chrono::high_resolution_clock::time_point tt3 = std::chrono::high_resolution_clock::now();
		activations_[layer + 1] = applyFunction(zs_[layer], active_);;
		//std::chrono::high_resolution_clock::time_point tt4 = std::chrono::high_resolution_clock::now();
		//double durt2 = std::chrono::duration_cast<std::chrono::microseconds>(tt4 - tt3).count();
		//std::cout << "Back_prop apply function time " << durt2 / 1000000 << '\n';

	}
	//std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
	//double dur = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
	//std::cout << "Back_prop feed forward " << dur / 1000000 << '\n';

	// Backward passj
	//std::chrono::high_resolution_clock::time_point t3 = std::chrono::high_resolution_clock::now();

	NNMatrix<double> delta = cost_derivative(activations_.back(), trainInput.second) * applyFunction(zs_.back(), active_prime_);
	nabla_b_back_.back() = delta;
	nabla_w_back_.back() = delta.dot(activations_[activations_.size() - 2].transpose());
	for (int l = layers_.size() - 2; l >= 1; l--)
	{
		//auto sp = applyFunction(zs_[l - 1], active_prime_);
		delta = ((weights_[l].transpose()).dot(delta)) * applyFunction(zs_[l - 1], active_prime_);
		nabla_b_back_[l - 1] = delta;
		nabla_w_back_[l - 1] = delta.dot(activations_[l - 1].transpose());
	}
	//std::chrono::high_resolution_clock::time_point t4 = std::chrono::high_resolution_clock::now();
	//double dur2 = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
	//std::cout << "Back_prop back prop " << dur2 / 1000000 << '\n';

}

NNMatrix<double> NNetwork::cost_derivative(const NNMatrix<double>& output_activation, int y)
{
	return (output_activation - double(y));
}





