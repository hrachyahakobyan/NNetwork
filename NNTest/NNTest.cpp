// NNTest.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "NNMatrix.h"

int _tmain(int argc, _TCHAR* argv[])
{
	NNMatrix<double> m(10, 10, 4);
	NNMatrix<double> m2(10, 10, 3);
	auto m3 = m.dot(m2);
	m3.print();
	int a;
	std::cin >> a;
	return 0;
}

