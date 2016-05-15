//#pragma once
//
//class NNetwork
//{
//public:
//	typedef std::vector<double> Vector;
//	typedef std::vector<Vector> Matrix;
//	typedef std::pair<Vector, int> TData;
//	enum class ActivFuncType{Sigmoid, Tangent};
//public:
//	void sgd(const std::vector<TData>& train_data, int epochs, int mbatch_size, double eta);
//	NNetwork(const std::vector<int>& sizes, ActivFuncType activType);
//	~NNetwork();
//private:
//
//	int num_layer_;
//	std::vector<int> sizes_;
//	Matrix biases_;
//	std::vector<Matrix> weights_;
//private:
//	typedef double(*Active_Func)(double);
//	Active_Func active_;
//	Active_Func active_prime_;
//
//	template<typename T>
//	T apply(const T& container, Active_Func func);
//
//	Vector feed_forward(const Vector& in);
//	Vector convolve(const Vector& input, const Matrix& weights, const Vector& biases);
//	Vector activate(int layer, const Vector& input);
//	Vector elem_prod(const Vector& a, const Vector& b);
//
//	void update_mini_batch(const std::vector<TData>& mini_batch, double eta);
//	std::pair<Matrix, std::vector<Matrix>> backprop(const TData& tdata);
//	int evaluate(const std::vector<TData>& test_data);
//	Vector cost_derivative(const Vector& output_activations, const int y);
//
//};
//
//
//template<>
//NNetwork::Vector NNetwork::apply(const Vector& container, Active_Func func)
//{
//	Vector out(container);
//	for (int i = 0; i < int(out.size()); i++)
//	{
//		out[i] = func(container[i]);
//	}
//	return out;
//}
//
//template<>
//NNetwork::Matrix NNetwork::apply(const Matrix& container, Active_Func func)
//{
//	Matrix out(container);
//	for (auto it = container.begin(); it != container.end(); ++it)
//	{
//		//*it = apply(*it, func);
//	}
//	return out;
//}
//
