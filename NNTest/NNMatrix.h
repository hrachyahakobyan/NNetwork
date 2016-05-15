#pragma once

template<std::size_t R, std::size_t C, typename T>
class NNMatrix
{
public:
	NNMatrix();
	explicit NNMatrix(const T val);
	explicit NNMatrix(const std::vector<T>& vec);
	explicit NNMatrix(T(*f)());
	~NNMatrix();
	void print();
public:
	T& operator()(std::size_t row, std::size_t col);
	const T& operator()(std::size_t row, std::size_t col) const;

	NNMatrix<R, C, T> operator+(const NNMatrix<R, C, T> &m);
	NNMatrix<R, C, T> operator-(const NNMatrix<R, C, T> &m);
	NNMatrix<R, C, T> operator*(const NNMatrix<R, C, T> &m);
	NNMatrix<R, C, T> operator/(const NNMatrix<R, C, T> &m);
	NNMatrix<R, C, T>& operator+=(const NNMatrix<R, C, T> &m);
	NNMatrix<R, C, T>& operator-=(const NNMatrix<R, C, T> &m);
	NNMatrix<R, C, T>& operator*=(const NNMatrix<R, C, T> &m);
	NNMatrix<R, C, T>& operator/=(const NNMatrix<R, C, T> &m);

	NNMatrix<R, C, T> operator+(const T val);
	NNMatrix<R, C, T> operator-(const T val);
	NNMatrix<R, C, T> operator*(const T val);
	NNMatrix<R, C, T> operator/(const T val);
	NNMatrix<R, C, T>& operator+=(const T val);
	NNMatrix<R, C, T>& operator-=(const T val);
	NNMatrix<R, C, T>& operator*=(const T val);
	NNMatrix<R, C, T>& operator/=(const T val);

	bool operator==(const NNMatrix<R, C, T> &other) const;
	bool operator!=(const NNMatrix<R, C, T> &other) const;

public:
	std::size_t rows() const;
	std::size_t cols() const;
	std::size_t size() const;
public:
	void apply(T(*f)(T));
public:
	NNMatrix<1, C, T> row(std::size_t row) const;
	NNMatrix<R, 1, T> col(std::size_t col) const;
	NNMatrix<C, R, T> transpose() const;

	template<std::size_t C2>
	NNMatrix<R, C2, T> dot(const NNMatrix<C, C2, T>& m2) const;
private:
	static_assert(R > 0, "Number of Rows must be greater than 0.");
	static_assert(C > 0, "Number of Columns must be greater than 0.");
	typedef std::vector<T> V;
	typedef std::vector<V> M;
	M m_;
};

/*Constructors*/

template<std::size_t R, std::size_t C, typename T>
NNMatrix<R, C, T>::NNMatrix() : m_(R, V(C))
{
}

template<std::size_t R, std::size_t C, typename T>
NNMatrix<R, C, T>::~NNMatrix()
{
}

template<std::size_t R, std::size_t C, typename T>
NNMatrix<R, C, T>::NNMatrix(const T val) : m_(R, V(C, val))
{
}

template<std::size_t R, std::size_t C, typename T>
NNMatrix<R, C, T>::NNMatrix(const std::vector<T>& vec) : m_(R, vec)
{
}

template<std::size_t R, std::size_t C, typename T>
NNMatrix<R, C, T>::NNMatrix(T(*f)()) : m_(R, V(C))
{
	for (int i = 0; i < R; i++)
	{
		for (int j = 0; j < C; j++)
		{
			m_[i][j] = f();
		}
	}
}


/*Access operators*/
template<std::size_t R, std::size_t C, typename T>
T& NNMatrix<R, C, T>::operator()(std::size_t row, std::size_t col)
{
	assert(row >= 0 && row < rows() && "Row out of bounds");
	assert(col >= 0 && col < cols() && "Column out of bounds");
	return m_[row][col];
}

template<std::size_t R, std::size_t C, typename T>
const T& NNMatrix<R, C, T>::operator()(std::size_t row, std::size_t col) const
{
	assert(row >= 0 && row < rows() && "Row out of bounds");
	assert(col >= 0 && col < cols() && "Column out of bounds");
	return m_[row][col];
}

/*Dimensions*/
template<std::size_t R, std::size_t C, typename T>
std::size_t NNMatrix<R, C, T>::rows() const
{
	return R;
}

template<std::size_t R, std::size_t C, typename T>
std::size_t NNMatrix<R, C, T>::cols() const
{
	return C;
}

template<std::size_t R, std::size_t C, typename T>
std::size_t NNMatrix<R, C, T>::size() const
{
	return rows() * cols();
}

/*Algebraic operations*/
template<std::size_t R, std::size_t C, typename T>
NNMatrix<1, C, T>  NNMatrix<R, C, T>::row(std::size_t row) const
{
	assert(row >= 0 && row < rows() && "Row out of bounds");
	return NNMatrix<1, R, T>(m_[row]);
}

template<std::size_t R, std::size_t C, typename T>
NNMatrix<R, 1, T>  NNMatrix<R, C, T>::col(std::size_t col) const
{
	assert(col >= 0 && col < cols() && "Col out of bounds");
	NNMatrix<R, 1, T> mat;
	for (int j = 0; j < R; j++)
		mat(j, 0) = m_[j][0];
	return mat;
}

template<std::size_t R, std::size_t C, typename T>
NNMatrix<C, R, T>  NNMatrix<R, C, T>::transpose() const
{
	NNMatrix<C, R, T> mat;
	for (std::size_t i = 0; i < C; i++)
		for (std::size_t j = 0; j < R; j++)
			mat(i, j) = m_[j][i];
	return mat;
}

template<std::size_t R, std::size_t C, typename T>
template<std::size_t C2>
NNMatrix<R, C2, T> NNMatrix<R, C, T>::dot(const NNMatrix<C, C2, T>& m1) const
{
	NNMatrix<R, C2, T> out;
	for (int i = 0; i < R; i++)
	{
		for (int j = 0; j < C2; j++)
		{
			T sum = T{};
			for (int k = 0; k < C; k++)
			{
				sum = sum + (m_[i][k] * m1(k, j));
			}
			out(i, j) = sum;
		}
	}
	return out;
}


/*Operators*/
template<std::size_t R, std::size_t C, typename T>
bool NNMatrix<R, C, T>::operator==(const NNMatrix<R, C, T> &other) const
{
	for (int i = 0; i < R; i++)
		for (int j = 0; j < C; j++)
		{
			if (m_[i][j] != other(i, j))
				return false;
		}
	return true;
}

template<std::size_t R, std::size_t C, typename T>
bool NNMatrix<R, C, T>::operator!=(const NNMatrix<R, C, T> &other) const
{
	return !((*this) == other);
}

template<std::size_t R, std::size_t C, typename T>
NNMatrix<R, C, T> NNMatrix<R, C, T>::operator+(const NNMatrix<R, C, T> &m)
{
	NNMatrix<R, C, T> out(*this);
	out += m;
	return out;
}

template<std::size_t R, std::size_t C, typename T>
NNMatrix<R, C, T> NNMatrix<R, C, T>::operator-(const NNMatrix<R, C, T> &m)
{
	NNMatrix<R, C, T> out(*this);
	out -= m;
	return out;
}

template<std::size_t R, std::size_t C, typename T>
NNMatrix<R, C, T> NNMatrix<R, C, T>::operator*(const NNMatrix<R, C, T> &m)
{
	NNMatrix<R, C, T> out(*this);
	out *= m;
	return out;
}

template<std::size_t R, std::size_t C, typename T>
NNMatrix<R, C, T> NNMatrix<R, C, T>::operator/(const NNMatrix<R, C, T> &m)
{
	NNMatrix<R, C, T> out(*this);
	out /= m;
	return out;
}

template<std::size_t R, std::size_t C, typename T>
NNMatrix<R, C, T>& NNMatrix<R, C, T>::operator+=(const NNMatrix<R, C, T> &m)
{
	for (std::size_t i = 0; i < rows(); i++)
		for (std::size_t j = 0; j < cols(); j++)
			m_[i][j] += m(i, j);
	return *this;
}

template<std::size_t R, std::size_t C, typename T>
NNMatrix<R, C, T>& NNMatrix<R, C, T>::operator-=(const NNMatrix<R, C, T> &m)
{
	for (std::size_t i = 0; i < rows(); i++)
		for (std::size_t j = 0; j < cols(); j++)
			m_[i][j] -= m(i, j);
	return *this;
}

template<std::size_t R, std::size_t C, typename T>
NNMatrix<R, C, T>& NNMatrix<R, C, T>::operator*=(const NNMatrix<R, C, T> &m)
{
	for (std::size_t i = 0; i < rows(); i++)
		for (std::size_t j = 0; j < cols(); j++)
			m_[i][j] *= m(i, j);
	return *this;
}

template<std::size_t R, std::size_t C, typename T>
NNMatrix<R, C, T>& NNMatrix<R, C, T>::operator/=(const NNMatrix<R, C, T> &m)
{
	for (std::size_t i = 0; i < rows(); i++)
		for (std::size_t j = 0; j < cols(); j++)
			m_[i][j] /= m(i, j);
	return *this;
}

template<std::size_t R, std::size_t C, typename T>
NNMatrix<R, C, T>& NNMatrix<R, C, T>::operator+=(const T val)
{
	for (std::size_t i = 0; i < rows(); i++)
		for (std::size_t j = 0; j < cols(); j++)
			m_[i][j] += val;
	return *this;
}

template<std::size_t R, std::size_t C, typename T>
NNMatrix<R, C, T>& NNMatrix<R, C, T>::operator-=(const T val)
{
	*this += (-val);
	return *this;
}

template<std::size_t R, std::size_t C, typename T>
NNMatrix<R, C, T>& NNMatrix<R, C, T>::operator*=(const T val)
{
	for (std::size_t i = 0; i < rows(); i++)
		for (std::size_t j = 0; j < cols(); j++)
			m_[i][j] *= val;
	return *this;
}

template<std::size_t R, std::size_t C, typename T>
NNMatrix<R, C, T>& NNMatrix<R, C, T>::operator/=(const T val)
{
	*this *= (1 / val);
	return *this;
}


template<std::size_t R, std::size_t C, typename T>
NNMatrix<R, C, T> NNMatrix<R, C, T>::operator+(const T val)
{
	NNMatrix<R, C, T> out(*this);
	out += val;
	return out;
}

template<std::size_t R, std::size_t C, typename T>
NNMatrix<R, C, T> NNMatrix<R, C, T>::operator-(const T val)
{
	NNMatrix<R, C, T> out(*this);
	out -= val;
	return out;
}

template<std::size_t R, std::size_t C, typename T>
NNMatrix<R, C, T> NNMatrix<R, C, T>::operator*(const T val)
{
	NNMatrix<R, C, T> out(*this);
	out *= val;
	return out;
}

template<std::size_t R, std::size_t C, typename T>
NNMatrix<R, C, T> NNMatrix<R, C, T>::operator/(const T val)
{
	NNMatrix<R, C, T> out(*this);
	out /= val;
	return out;
}


/*Apply*/
template<std::size_t R, std::size_t C, typename T>
void NNMatrix<R, C, T>::apply(T(*f)(T))
{
	for (std::size_t i = 0; i < rows(); i++)
		for (std::size_t j = 0; j < cols(); j++)
			m_[i][j] = f(m_[i][j]);
}

/*Print*/

template<std::size_t R, std::size_t C, typename T>
void NNMatrix<R, C, T>::print()
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