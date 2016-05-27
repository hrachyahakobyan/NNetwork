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


NNetwork::NNetwork(const std::vector<std::size_t>& layers, ActivationFunctionType activType, CostType costType, RegularizationType regType) : layers_(layers)
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
	NNMatrix<double> biases = biases_[layer - 1];
	NNMatrix<double> weights = weights_[layer - 1];
	NNMatrix<double> prepared_input = prepare_input(input, biases, weights);
	NNMatrix<double> activated = applyFunction(prepared_input, active_);
	return activated;
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
		std::vector<std::vector<TrainInput>> mini_batches;
		break_trainData(trainData, batch_size, mini_batches);
		std::vector<std::vector<TrainInput>>::const_iterator it;
		for (std::size_t mb = 0; mb < mini_batches.size(); mb++)
		{
			std::cout << "Mini batch " << mb << std::endl;
			update_mini_batch(mini_batches[mb], eta);
		}
		if (val_data.size() > 0)
		{
			int correct = 0;
			evaluate(val_data, correct);
			std::cout << "Epoch: " << epoch << correct << " out of " << val_data.size() << std::endl;
		}
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
		auto delta_nablas = back_prop(trainData[i]);
		for (std::size_t i = 0; i < delta_nablas.first.size(); i++)
		{
			nabla_b[i] += delta_nablas.first[i];
			nabla_w[i] += delta_nablas.second[i];
		}
	}

	for (std::size_t i = 0; i < nabla_b.size(); i++)
	{
		weights_[i] = weights_[i] - nabla_w[i] * (double(eta / trainData.size()));
		biases_[i] = biases_[i] - nabla_b[i] * (double(eta / trainData.size()));
	}
}


// Back propagation algorithm. Given a single input X and output Y, returns the layer by layer
// derivatives of biases and weights of the cost function
std::pair<std::vector<NNMatrix<double>>, std::vector<NNMatrix<double>>> NNetwork::back_prop(const TrainInput& trainInput)
{
	auto nabla_b = std::vector<NNMatrix<double>>(biases_.size());
	auto nabla_w = std::vector<NNMatrix<double>>(weights_.size());

	// The activation of the first layer of the neurons, i.e. the input
	NNMatrix<double> activation(trainInput.first.size(), 1, trainInput.first);
	// The vector of all activations of layers
	std::vector<NNMatrix<double>> activations(layers_.size());
	activations[0] = activation;

	// The vector storing all the prepared inputs, i.e.
	// the inputs passed through the prepare_input method
	std::vector<NNMatrix<double>> zs(layers_.size() - 1);
	

	// Feeding forward
	for (std::size_t layer = 0; layer < biases_.size(); layer++)
	{
		// The prepared input of the current layer
		auto z = prepare_input(activation, biases_[layer], weights_[layer]);
		zs[layer] = z;

		// Apply transformation to the preapred input
		activation = applyFunction(z, active_);
		activations[layer + 1] = activation;
	}

	// Backward passj
	NNMatrix<double> delta = cost_derivative(activations.back(), trainInput.second) * applyFunction(zs.back(), active_prime_);
	nabla_b.back() = delta;
	nabla_w.back() = delta.dot(activations[activations.size() - 2].transpose());

	for (int l = layers_.size() - 2; l >= 1; l--)
	{
		auto z = zs[l - 1];
		auto sp = applyFunction(z, active_prime_);
		delta = ((weights_[l].transpose()).dot(delta)) * sp;
		nabla_b[l - 1] = delta;
		nabla_w[l - 1] = delta.dot(activations[l - 1].transpose());
	}

	return std::make_pair(nabla_b, nabla_w);
}

NNMatrix<double> NNetwork::cost_derivative(const NNMatrix<double>& output_activation, int y)
{
	auto copy(output_activation);
	return (copy - double(y));
}





