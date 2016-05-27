#include "NNMatrix.h"

class NNetwork
{
public:
	enum class ActivationFunctionType{Sigmoid};
	enum class CostType{Quadratic, CrossEntropy};
	enum class RegularizationType{None, L1, L2};
public:
	NNetwork(const std::vector<std::size_t>& layers, ActivationFunctionType activType, CostType costType, RegularizationType regType = RegularizationType::None);
	~NNetwork(); 

	typedef std::pair<std::vector<double>, unsigned int> TrainInput;
	void train(std::vector<TrainInput>& trainData,const std::vector<TrainInput>& val_data, std::size_t epochs, std::size_t batch_size, double eta);
	void evaluate(const std::vector<TrainInput>& val_data, int& correct);
private:
	typedef NNMatrix<double> Matrix;
	std::vector<std::size_t> layers_;
	std::vector<Matrix> biases_;
	std::vector<Matrix> weights_;
private:
	typedef double(NNetwork::*Active_Func)(double);
	Active_Func active_;
	Active_Func active_prime_;

	typedef double(NNetwork::*Cost_Func)(const Matrix& a, const int y);
	typedef Matrix(NNetwork::*Cost_Deriv_Func)(const Matrix& z, const Matrix& a, const int y);
	Cost_Func cost_;
	Cost_Deriv_Func cost_deriv_;
private:
	Matrix prepare_input(const Matrix& input, const Matrix& biases, const Matrix& weights);
	Matrix activate(std::size_t layer, const Matrix& input);
	Matrix feed_forward(const Matrix& input);
	Matrix cost_derivative(const Matrix& output_activation, int y);
	std::pair<std::vector<Matrix>, std::vector<Matrix>> back_prop(const TrainInput& trainInput);
	void update_mini_batch(const std::vector<TrainInput>& trainData, double eta);
	void break_trainData(std::vector<TrainInput>& trainData, std::size_t batch_size, std::vector<std::vector<TrainInput>>& mini_batches);
	
private:
	double log(double a);

	 double sigmoid(double a);
	 double sigmoid_prime(double a);

	 double cost_quad(const Matrix& a, const int y);
	 double cost_cross(const Matrix& a, const int y);

	 Matrix cost_deriv_quad(const Matrix& z, const Matrix& a, const int y);
	 Matrix cost_deriv_cross(const Matrix& z, const Matrix& a, const int y);

	 Matrix applyFunction(const Matrix& matrix, Active_Func f);
};