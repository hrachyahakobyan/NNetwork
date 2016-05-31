enum class ActivationFunction{
	Sigmoid,
	Softmax
};

enum class CostFunction{
	Quadratic,
	CrossEntropy,
	LogLikelyhood
};

enum class Regularization{
	None,
	L1,
	L2
};

class NNetwork
{
public:
	NNetwork(const std::vector<std::size_t>& layers, ActivationFunction activType, CostFunction costType, Regularization regType = Regularization::None);
	~NNetwork(); 

	typedef std::pair<std::vector<double>, unsigned int> TrainInput;
	void train(std::vector<TrainInput>& trainData,const std::vector<TrainInput>& val_data, std::size_t epochs, std::size_t batch_size, double eta);
	void evaluate(const std::vector<TrainInput>& val_data, int& correct);
private:
	typedef Eigen::VectorXd NVector;
	typedef Eigen::MatrixXd NMatrix;

	std::vector<std::size_t> layers_;
	std::vector<NVector> biases_;
	std::vector<NMatrix> weights_;
	std::vector<NVector> nabla_b_back_;
	std::vector<NMatrix> nabla_w_back_;
	std::vector<NVector> activations_;
	std::vector<NVector> zs_;

private:
	typedef double(NNetwork::*Active_Func)(double);
	Active_Func active_;
	Active_Func active_prime_;

	typedef double(NNetwork::*Cost_Func)(const NVector& a, const int y);
	typedef NVector(NNetwork::*Cost_Deriv_Func)(const NVector& z, const NVector& a, const int y);

	Cost_Func cost_;
	Cost_Deriv_Func cost_deriv_;

private:
	NVector prepare_input(const NVector& input, const NVector& biases, const NMatrix& weights);
	NVector activate(std::size_t layer, const NVector& input);
	NVector feed_forward(const NVector& input);
	void back_prop(const TrainInput& trainInput);
	void update_mini_batch(const std::vector<TrainInput>& trainData, double eta);
	void break_trainData(std::vector<TrainInput>& trainData, std::size_t batch_size, std::vector<std::vector<TrainInput>>& mini_batches);
	
private:
	double log(double a);

	 double sigmoid(double a);
	 double sigmoid_prime(double a);

	 double cost_quad(const NVector& a, const int y);
	 double cost_cross(const NVector& a, const int y);

	 NVector cost_deriv_quad(const NVector& z, const NVector& a, const int y);
	 NVector cost_deriv_cross(const NVector& z, const NVector& a, const int y);

	 NVector applyFunction(const NVector& matrix, Active_Func f);
};