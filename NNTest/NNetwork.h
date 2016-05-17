#include "NNMatrix.h"

class NNetwork
{
public:
	enum class ActivationFunctionType{Sigmoid};
public:
	NNetwork(const std::vector<std::size_t>& layers, ActivationFunctionType activType);
	~NNetwork(); 

	typedef std::pair<std::vector<double>, unsigned int> TrainInput;
	void train(std::vector<TrainInput>& trainData,const std::vector<TrainInput>& val_data, std::size_t epochs, std::size_t batch_size, double eta);
	void evaluate(const std::vector<TrainInput>& val_data, int& correct);
private:
	std::vector<std::size_t> layers_;
	std::vector<NNMatrix<double>> biases_;
	std::vector<NNMatrix<double>> weights_;
private:
	typedef double(*Active_Func)(double);
	Active_Func active_;
	Active_Func active_prime_;
private:
	NNMatrix<double> prepare_input(const NNMatrix<double>& input, const NNMatrix<double>& biases, const NNMatrix<double>& weights);
	NNMatrix<double> activate(std::size_t layer, const NNMatrix<double>& input);
	NNMatrix<double> feed_forward(const NNMatrix<double>& input);
	NNMatrix<double> applyFunction(const NNMatrix<double>& matrix, Active_Func f);
	NNMatrix<double> cost_derivative(const NNMatrix<double>& output_activation, int y);
	std::pair<std::vector<NNMatrix<double>>, std::vector<NNMatrix<double>>> back_prop(const TrainInput& trainInput);
	void update_mini_batch(const std::vector<TrainInput>& trainData, double eta);
	void break_trainData(std::vector<TrainInput>& trainData, std::size_t batch_size, std::vector<std::vector<TrainInput>>& mini_batches);

};