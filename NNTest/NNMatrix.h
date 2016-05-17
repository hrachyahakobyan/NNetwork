#pragma once

template<typename T>
class NNMatrix
{
public:
	NNMatrix();
	NNMatrix(std::size_t rows, std::size_t cols);
	NNMatrix(std::size_t rows, std::size_t cols, T val);
	NNMatrix(std::size_t rows, std::size_t cols, const std::vector<T>& vec);
    NNMatrix(std::size_t rows, std::size_t cols, T(*f)());
	~NNMatrix();
	void print();
public:
	T& operator()(std::size_t row, std::size_t col);
	const T& operator()(std::size_t row, std::size_t col) const;

	NNMatrix<T> operator+(const NNMatrix<T> &m);
	NNMatrix<T> operator-(const NNMatrix<T> &m);
	NNMatrix<T> operator*(const NNMatrix<T> &m);
	NNMatrix<T> operator/(const NNMatrix<T> &m);
	NNMatrix<T>& operator+=(const NNMatrix<T> &m);
	NNMatrix<T>& operator-=(const NNMatrix<T> &m);
	NNMatrix<T>& operator*=(const NNMatrix<T> &m);
	NNMatrix<T>& operator/=(const NNMatrix<T> &m);

	NNMatrix<T> operator+(const T val);
	NNMatrix<T> operator-(const T val);
	NNMatrix<T> operator*(const T val);
	NNMatrix<T> operator/(const T val);
	NNMatrix<T>& operator+=(const T val);
	NNMatrix<T>& operator-=(const T val);
	NNMatrix<T>& operator*=(const T val);
	NNMatrix<T>& operator/=(const T val);

	bool operator==(const NNMatrix<T> &other) const;
	bool operator!=(const NNMatrix<T> &other) const;

public:
	std::size_t rows() const;
	std::size_t cols() const;
	std::size_t size() const;
public:
	void apply(T(*f)(T));
	std::pair<std::size_t, std::size_t> max_index() const;
	std::pair<std::size_t, std::size_t> min_index() const;
public:
	NNMatrix<T> row(std::size_t row) const;
	NNMatrix<T> col(std::size_t col) const;
	NNMatrix<T> transpose() const;
	NNMatrix<T> dot(const NNMatrix<T>& m2) const;
private:
	typedef std::vector<T> V;
	typedef std::vector<V> M;
	std::size_t rows_;
	std::size_t cols_;
	M m_;
	void assert_rows(const NNMatrix<T>& other);
	void assert_cols(const NNMatrix<T>& other);
	void assert_dims(const NNMatrix<T>& other);
};

/*Constructors*/

template<typename T>
NNMatrix<T>::NNMatrix() : rows_(0), cols_(0)
{

}

template<typename T>
NNMatrix<T>::NNMatrix(std::size_t rows, std::size_t cols) : rows_(rows), cols_(cols), m_(rows_, V(cols_))
{
}

template<typename T>
NNMatrix<T>::~NNMatrix()
{
}

template<typename T>
NNMatrix<T>::NNMatrix(std::size_t rows, std::size_t cols, T val) : rows_(rows), cols_(cols), m_(rows_, V(cols_, val))
{
}

template<typename T>
NNMatrix<T>::NNMatrix(std::size_t rows, std::size_t cols, const std::vector<T>& vec) : rows_(rows), cols_(cols), m_(rows_, vec)
{
}

template<typename T>
NNMatrix<T>::NNMatrix(std::size_t rows, std::size_t cols, T(*f)()) : rows_(rows), cols_(cols), m_(rows_, V(cols_))
{
	for (std::size_t i = 0; i < rows_; i++)
	{
		for (std::size_t j = 0; j < cols_; j++)
		{
			m_[i][j] = f();
		}
	}
}


/*Access operators*/
template<typename T>
T& NNMatrix<T>::operator()(std::size_t row, std::size_t col)
{
	assert(row >= 0 && row < rows() && "Row out of bounds");
	assert(col >= 0 && col < cols() && "Column out of bounds");
	return m_[row][col];
}

template<typename T>
const T& NNMatrix<T>::operator()(std::size_t row, std::size_t col) const
{
	assert(row >= 0 && row < rows() && "Row out of bounds");
	assert(col >= 0 && col < cols() && "Column out of bounds");
	return m_[row][col];
}

/*Dimensions*/
template<typename T>
std::size_t NNMatrix<T>::rows() const
{
	return rows_;
}

template<typename T>
std::size_t NNMatrix<T>::cols() const
{
	return cols_;
}

template<typename T>
std::size_t NNMatrix<T>::size() const
{
	return rows() * cols();
}

/*Algebraic operations*/
template<typename T>
NNMatrix<T>  NNMatrix<T>::row(std::size_t row) const
{
	assert(row >= 0 && row < rows() && "Row out of bounds");
	return NNMatrix<T>(1, cols_, m_[row]);
}

template<typename T>
NNMatrix<T>  NNMatrix<T>::col(std::size_t col) const
{
	assert(col >= 0 && col < cols() && "Col out of bounds");
	NNMatrix<T> mat(rows_, 1);
	for (int j = 0; j < rows_; j++)
		mat(j, 0) = m_[j][0];
	return mat;
}

template<typename T>
NNMatrix<T>  NNMatrix<T>::transpose() const
{
	NNMatrix<T> mat(cols_, rows_);
	for (std::size_t i = 0; i < cols_; i++)
		for (std::size_t j = 0; j < rows_; j++)
			mat(i, j) = m_[j][i];
	return mat;
}

template<typename T>
NNMatrix<T> NNMatrix<T>::dot(const NNMatrix<T>& m1) const
{
	assert(cols_ == m1.rows() && "Dot product: incompatible dimensions");
	NNMatrix<T> out(rows_, m1.cols());
	for (std::size_t i = 0; i < rows_; i++)
	{
		for (std::size_t j = 0; j < m1.cols(); j++)
		{
			T sum = T{};
			for (std::size_t k = 0; k < cols_; k++)
			{
				sum = sum + (m_[i][k] * m1(k, j));
			}
			out(i, j) = sum;
		}
	}
	return out;
}


/*Operators*/
template<typename T>
bool NNMatrix<T>::operator==(const NNMatrix<T> &other) const
{
	assert_dims(other);
	for (int i = 0; i < rows_; i++)
		for (int j = 0; j < cols_; j++)
		{
			if (m_[i][j] != other(i, j))
				return false;
		}
	return true;
}

template<typename T>
bool NNMatrix<T>::operator!=(const NNMatrix<T> &other) const
{
	assert_dims(other);
	return !((*this) == other);
}

template<typename T>
NNMatrix<T> NNMatrix<T>::operator+(const NNMatrix<T> &m)
{
	assert_dims(m);
	NNMatrix<T> out(*this);
	out += m;
	return out;
}

template<typename T>
NNMatrix<T> NNMatrix<T>::operator-(const NNMatrix<T> &m)
{
	assert_dims(m);
	NNMatrix<T> out(*this);
	out -= m;
	return out;
}

template<typename T>
NNMatrix<T> NNMatrix<T>::operator*(const NNMatrix<T> &m)
{
	assert_dims(m);
	NNMatrix<T> out(*this);
	out *= m;
	return out;
}

template<typename T>
NNMatrix<T> NNMatrix<T>::operator/(const NNMatrix<T> &m)
{
	assert_dims(m);
	NNMatrix<T> out(*this);
	out /= m;
	return out;
}

template<typename T>
NNMatrix<T>& NNMatrix<T>::operator+=(const NNMatrix<T> &m)
{
	assert_dims(m);
	for (std::size_t i = 0; i < rows(); i++)
		for (std::size_t j = 0; j < cols(); j++)
			m_[i][j] += m(i, j);
	return *this;
}

template<typename T>
NNMatrix<T>& NNMatrix<T>::operator-=(const NNMatrix<T> &m)
{
	assert_dims(m);
	for (std::size_t i = 0; i < rows(); i++)
		for (std::size_t j = 0; j < cols(); j++)
			m_[i][j] -= m(i, j);
	return *this;
}

template<typename T>
NNMatrix<T>& NNMatrix<T>::operator*=(const NNMatrix<T> &m)
{
	assert_dims(m);
	for (std::size_t i = 0; i < rows(); i++)
		for (std::size_t j = 0; j < cols(); j++)
			m_[i][j] *= m(i, j);
	return *this;
}

template<typename T>
NNMatrix<T>& NNMatrix<T>::operator/=(const NNMatrix<T> &m)
{
	assert_dims(m);
	for (std::size_t i = 0; i < rows(); i++)
		for (std::size_t j = 0; j < cols(); j++)
			m_[i][j] /= m(i, j);
	return *this;
}

template<typename T>
NNMatrix<T>& NNMatrix<T>::operator+=(const T val)
{
	for (std::size_t i = 0; i < rows(); i++)
		for (std::size_t j = 0; j < cols(); j++)
			m_[i][j] += val;
	return *this;
}

template<typename T>
NNMatrix<T>& NNMatrix<T>::operator-=(const T val)
{
	*this += (-val);
	return *this;
}

template<typename T>
NNMatrix<T>& NNMatrix<T>::operator*=(const T val)
{
	for (std::size_t i = 0; i < rows(); i++)
		for (std::size_t j = 0; j < cols(); j++)
			m_[i][j] *= val;
	return *this;
}

template<typename T>
NNMatrix<T>& NNMatrix<T>::operator/=(const T val)
{
	*this *= (1 / val);
	return *this;
}


template<typename T>
NNMatrix<T> NNMatrix<T>::operator+(const T val)
{
	NNMatrix<T> out(*this);
	out += val;
	return out;
}

template<typename T>
NNMatrix<T> NNMatrix<T>::operator-(const T val)
{
	NNMatrix<T> out(*this);
	out -= val;
	return out;
}

template<typename T>
NNMatrix<T> NNMatrix<T>::operator*(const T val)
{
	NNMatrix<T> out(*this);
	out *= val;
	return out;
}

template<typename T>
NNMatrix<T> NNMatrix<T>::operator/(const T val)
{
	NNMatrix<T> out(*this);
	out /= val;
	return out;
}

/*Assertions*/
template<typename T>
void NNMatrix<T>::assert_rows(const NNMatrix<T>& other)
{
	assert(rows_ == other.rows() && "Error: incompatible rows");
}

template<typename T>
void NNMatrix<T>::assert_cols(const NNMatrix<T>& other)
{
	assert(cols_ == other.cols() && "Error: incompatible columns");
}

template<typename T>
void NNMatrix<T>::assert_dims(const NNMatrix<T>& other)
{
	assert_rows(other);
	assert_cols(other);
}

/*Apply*/
template<typename T>
void NNMatrix<T>::apply(T(*f)(T))
{
	for (std::size_t i = 0; i < rows(); i++)
		for (std::size_t j = 0; j < cols(); j++)
			m_[i][j] = f(m_[i][j]);
}


template<typename T>
std::pair<std::size_t, std::size_t> NNMatrix<T>::max_index() const
{
	if (size() == 0)
		return std::make_pair(0,0);
	T max = m_[0][0];
	std::pair<std::size_t, std::size_t> max_id(0,0);
	for (std::size_t i = 0; i < rows_; i++)
	{
		for (std::size_t j = 0; j < cols_; j++)
		{
			if (m_[i][j] > max)
			{
				max = m_[i][j];
				max_id.first = i;
				max_id.second = j;
			}
		}
	}
	return max_id;
}

template<typename T>
std::pair<std::size_t, std::size_t> NNMatrix<T>::min_index() const
{
	if (size() == 0)
		return std::make_pair<0, 0>;
	T min = m_[0][0];
	std::pair<std::size_t, std::size_t> min_id(0, 0);
	for (std::size_t i = 0; i < rows_; i++)
	{
		for (std::size_t j = 0; j < cols_; j++)
		{
			if (m_[i][j] < min)
			{
				min = m_[i][j];
				min_id.first = i;
				min_id.second = j;
			}
		}
	}
	return min_id;
}

/*Print*/

template<typename T>
void NNMatrix<T>::print()
{
	for (auto outIt = m_.begin(); outIt != m_.end(); ++outIt)
	{
		for (auto inIt = (*outIt).begin(); inIt != (*outIt).end(); ++inIt)
		{
			std::cout << *inIt << " ";
		}
		std::cout << '\n';
	}
}