#include "stdafx.h"
//#include "NNetwork.h"
//
//double sigmoid(double);
//double sigmoid_prime(double);
//double tangent(double);
//
//NNetwork::NNetwork(const std::vector<int>& sizes, ActivFuncType activType) : sizes_(sizes), num_layer_(sizes.size())
//{
//	RandomManager* rand = RandomManager::sharedManager();
//	/* The first (input) layer does not have biases*/
//	for (int layer = 1; layer < num_layer_ ; layer++)
//	{
//		biases_.push_back(Vector(sizes[layer], rand->gauss_rand(0, 1)));
//	}
//	/*There are num_layer_ - 1 weight matrices */
//	weights_ = std::vector<Matrix>(num_layer_ - 1);
//	for (int layer = 0; layer < num_layer_ - 1; layer++)
//	{
//		Matrix wMatrix(sizes_[layer + 1]);
//		/*a matrix of weights of dimensions SIZE(layer) x SIZE(previous_layer)*/
//		for (auto it1 = wMatrix.begin(); it1 != wMatrix.end(); ++it1)
//		{
//			wMatrix.push_back(Vector(sizes_[layer], rand->gauss_rand(0, 1)));
//		}
//		weights_.push_back(wMatrix);
//	}
//
//	switch (activType)
//	{
//	case ActivFuncType::Sigmoid:
//	{
//		active_prime_ = &sigmoid_prime;
//		active_ = &sigmoid;
//	}
//		break;
//	case ActivFuncType::Tangent:
//		active_ = &tangent;
//		break;
//	default:
//		break;
//	}
//}
//
//
//NNetwork::~NNetwork()
//{
//
//}
//
//NNetwork::Vector NNetwork::feed_forward(const Vector& in)
//{
//	int size = int(biases_.size());
//	Vector out = in;
//	for (int i = 1; i < size; i++)
//	{
//		out = activate(i, out);
//	}
//	return out;
//}
//
//NNetwork::Vector NNetwork::convolve(const Vector& input, const Matrix& weights, const Vector& biases)
//{
//	Vector output(biases.size());
//	for (int i = 0; i < int(output.size()); i++)
//	{
//		double sum = 0;
//		for (int j = 0; j < int(input.size()); j++)
//		{
//			sum += input[j] * weights[i][j];
//		}
//		sum += biases[i];
//		output[i] = sum;
//	}
//	return output;
//}
//
//NNetwork::Vector NNetwork::activate(int layer, const Vector& input)
//{
//	Vector convolved = convolve(input, weights_[layer - 1], biases_[layer]);
//	Vector applied = apply(convolved, active_);
//	return applied;
//}
//
//
//std::pair<NNetwork::Matrix, std::vector<NNetwork::Matrix>> NNetwork::backprop(const TData& tdata)
//{
//	auto nabla_b(biases_);
//	auto nabla_w(weights_);
//
//	Vector activation(tdata.first);
//	std::vector<Vector> activations;
//	activations.push_back(activation);
//
//	std::vector<Vector> zs;
//
//	/*Forward pass*/
//
//	for (int i = 0; i < int(biases_.size()); i++)
//	{
//		Vector z = convolve(activation, weights_[i], biases_[i]);
//		zs.push_back(z);
//
//		activation = apply(z, active_);
//		activations.push_back(activation);
//	}
//
//	/*Backward pass*/
//	Vector delta = elem_prod(cost_derivative(activations.back(), tdata.second), apply(zs.back(), active_prime_));
//
//	nabla_b.back() = delta;
//
//
//
//
//
//}
//
//NNetwork::Vector NNetwork::cost_derivative(const Vector& output_activation, const int y)
//{
//	Vector out(output_activation.size());
//	for (int i = 0; i < int(out.size()); i++)
//	{
//		out[i] = output_activation[i] - y;
//	}
//	return out;
//}
//
//NNetwork::Vector NNetwork::elem_prod(const Vector& a, const Vector& b)
//{
//	Vector c(a.size());
//	for (int i = 0; i < int(c.size()); i++)
//	{
//		c[i] = a[i] * b[i];
//	}
//	return c;
//}
//
//double sigmoid(double a)
//{
//	return 1.0 / (1.0 + std::exp(-a));
//}
//
//double sigmoid_prime(double a)
//{
//	return sigmoid(a) * (1 - sigmoid(a));
//}
//
//double tangent(double a)
//{
//	return a;
//}
//
//
