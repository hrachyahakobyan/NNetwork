#include "stdafx.h"
#include "NNetwork.h"

#define TIME_EPOCHS true
#define TIME_MINI_BATCHES true
#define TIME_TRAIN_DATA true
#define TIME_BB true
#define TIME_BB_FF true // feedforward loop in backpropagation
#define TIME_BB_BB_LOOP true // backpropagation loop in backpropagation
#define TIME_BB_BB_PRELOOP true
#define TIME_NABLA_LOOP true 

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

double NNetwork::cost_quad(const NVector& a, const int y)
{
	return (a.array() - y).matrix().squaredNorm() * 0.5;
}

double NNetwork::cost_cross(const NVector& a, const int y)
{
	double sum = 0;
	for (std::size_t i = 0; i < a.cols(); i++)
		sum += -y * log(a(0, i)) - (1 - y) * log(1 - a(0, i));
	return sum;
}

NNetwork::NVector NNetwork::cost_deriv_quad(const NVector& z, const NVector& a, const int y)
{
	return (a.array() - y).matrix().cwiseProduct(applyFunction(z, active_prime_));
}

NNetwork::NVector NNetwork::cost_deriv_cross(const NVector& z, const NVector& a, const int y)
{
	return (a.array() - y);
}

// Applies the given transformation to each element of the matrix
NNetwork::NVector NNetwork::applyFunction(const NVector& matrix, Active_Func f)
{
	NVector out = matrix;
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
		biases_.push_back(NVector::Random(layers[i]));
	}
	// Prepare the weight matrices. A weight matrice is a NxM matrix of doubles, where
	// N is the number of the neurons in the NEXT layer and M is the number of neurons in the
	// PREVIOUS layer. There are L-1 such matrices, where L is the number of layers
	for (std::size_t i = 0; i < layers.size() - 1; i++)
	{
		weights_.push_back(NMatrix::Random(layers[i + 1], layers[i]));
	}

	activations_[0] = NVector(layers_[0]);

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
NNetwork::NVector NNetwork::prepare_input(const NVector& input, const NVector& biases, const NMatrix& weights)
{
	return (weights*input + biases);
}

// Activates the given later given the input from the previous layer.
// Step1. prepares the input
// Step2. activates the input by applying the activation function
NNetwork::NVector NNetwork::activate(std::size_t layer, const NVector& input)
{
	// The layers are indexed starting from 0
	NVector prepared_input = prepare_input(input, biases_[layer - 1], weights_[layer - 1]);
	return applyFunction(prepared_input, active_);
}

// Feed the given input to the neural network and return the output from the last layer
NNetwork::NVector NNetwork::feed_forward(const NVector& input)
{
	NVector output = input;
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
		std::chrono::high_resolution_clock::time_point t1Epoch = std::chrono::high_resolution_clock::now();
		std::vector<std::vector<TrainInput>> mini_batches;
		break_trainData(trainData, batch_size, mini_batches);
		std::vector<std::vector<TrainInput>>::const_iterator it;
		for (std::size_t mb = 0; mb < mini_batches.size(); mb++)
		{
			std::cout << "Mini batch " << mb << std::endl;
			std::chrono::high_resolution_clock::time_point t1Batch = std::chrono::high_resolution_clock::now();
			update_mini_batch(mini_batches[mb], eta);
			if (TIME_MINI_BATCHES)
			{
				std::chrono::high_resolution_clock::time_point t2Batch = std::chrono::high_resolution_clock::now();
				double dur = std::chrono::duration_cast<std::chrono::microseconds>(t2Batch - t1Batch).count();
				std::cout << "Mini-batch time " << dur / 1000000 << '\n';
			}
		}
		if (val_data.size() > 0)
		{
			int correct = 0;
			evaluate(val_data, correct);
			std::cout << "Epoch: " << epoch << correct << " out of " << val_data.size() << std::endl;
		}
		if (TIME_EPOCHS)
		{
			std::chrono::high_resolution_clock::time_point t2Epoch = std::chrono::high_resolution_clock::now();
			double dur = std::chrono::duration_cast<std::chrono::microseconds>(t2Epoch - t1Epoch).count();
			std::cout << "Epoch time " << dur / 1000000 << '\n';
		}

	}
}

void NNetwork::evaluate(const std::vector<TrainInput>& val_data, int& correct)
{
	correct = 0;
	for (std::size_t i = 0; i < val_data.size(); i++)
	{
		//NNMatrix<double> result = feed_forward(NNMatrix<double>(val_data[i].first.size(), 1, val_data[i].first));
		//std::pair<std::size_t, std::size_t> approx_result = result.max_index();
		//if (approx_result.first == val_data[i].second)
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
		std::chrono::high_resolution_clock::time_point tBegin = std::chrono::high_resolution_clock::now();
		back_prop(trainData[i]);
		if (TIME_BB)
		{
			std::chrono::high_resolution_clock::time_point tBBEnd = std::chrono::high_resolution_clock::now();
			double dur = std::chrono::duration_cast<std::chrono::microseconds>(tBBEnd - tBegin).count();
			std::cout << "BB time " << dur / 1000000 << '\n';
		}
		std::chrono::high_resolution_clock::time_point tNablaBegin = std::chrono::high_resolution_clock::now();
		for (std::size_t i = 0; i < nabla_b_back_.size(); i++)
		{
			nabla_b[i] += nabla_b_back_[i];
			nabla_w[i] += nabla_w_back_[i];
		}
		std::chrono::high_resolution_clock::time_point tEnd = std::chrono::high_resolution_clock::now();
		if (TIME_TRAIN_DATA)
		{
			double dur = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tBegin).count();
			std::cout << "Train data time " << dur / 1000000 << '\n';
		}
		if (TIME_NABLA_LOOP)
		{
			double dur = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tNablaBegin).count();
			std::cout << "Nabla loop time " << dur / 1000000 << '\n';
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
	std::chrono::high_resolution_clock::time_point tFFBegin = std::chrono::high_resolution_clock::now();
	// The activation of the first layer of the neurons, i.e. the input
	std::vector<double> co(trainInput.first);
	double* ptr = &(co[0]);
	activations_[0] = Eigen::Map<Eigen::VectorXd>(ptr, co.size());
	// Feeding forward
	for (std::size_t layer = 0; layer < biases_.size(); layer++)
	{
		zs_[layer] = prepare_input(activations_[layer], biases_[layer], weights_[layer]);
		activations_[layer + 1] = applyFunction(zs_[layer], active_);;
	}
	if (TIME_BB_FF)
	{
		std::chrono::high_resolution_clock::time_point tFFEnd = std::chrono::high_resolution_clock::now();
		double dur = std::chrono::duration_cast<std::chrono::microseconds>(tFFEnd - tFFBegin).count();
		std::cout << "BB FF time " << dur / 1000000 << '\n';
	}
	// Backward pass
	std::chrono::high_resolution_clock::time_point tBBPreloopBegin = std::chrono::high_resolution_clock::now();
	NVector delta = cost_derivative(activations_.back(), trainInput.second).cwiseProduct(applyFunction(zs_.back(), active_prime_));
	nabla_b_back_.back() = delta;
	nabla_w_back_.back().noalias() = delta * (activations_[activations_.size() - 2].transpose());
	if (TIME_BB_BB_PRELOOP)
	{
		std::chrono::high_resolution_clock::time_point tBBPreloopEnd = std::chrono::high_resolution_clock::now();
		double dur2 = std::chrono::duration_cast<std::chrono::microseconds>(tBBPreloopEnd - tBBPreloopBegin).count();
		std::cout << "BB preloop time " << dur2 / 1000000 << '\n';
	}

	std::chrono::high_resolution_clock::time_point tBBLoopBegin = std::chrono::high_resolution_clock::now();

	for (int l = layers_.size() - 2; l >= 1; l--)
	{
		delta = (weights_[l].transpose() * (delta)).cwiseProduct(applyFunction(zs_[l - 1], active_prime_));
		nabla_b_back_[l - 1] = delta;
		nabla_w_back_[l - 1].noalias() = delta * (activations_[l - 1].transpose());
	}
	if (TIME_BB_BB_LOOP)
	{
		std::chrono::high_resolution_clock::time_point tBBLoopEnd = std::chrono::high_resolution_clock::now();
		double dur = std::chrono::duration_cast<std::chrono::microseconds>(tBBLoopEnd - tBBLoopBegin).count();
		std::cout << "BB loop time " << dur / 1000000 << '\n';
	}
}

NNetwork::NVector NNetwork::cost_derivative(const NVector& output_activation, int y)
{
	return (output_activation.array() - y);
}





