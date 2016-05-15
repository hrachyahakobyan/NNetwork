#include "stdafx.h"
#include "RandomManager.h"
#include <time.h>
#define EPSILON 0.0001

RandomManager* RandomManager::sharedRandomManager = NULL;

RandomManager::RandomManager()
{
	srand(time(NULL));
}


RandomManager::~RandomManager()
{
}


double RandomManager::gauss_rand(double mean, double stddev)
{
	std::default_random_engine gen;
	std::normal_distribution<double> distribution(mean, stddev);
	return distribution(gen);
}


int RandomManager::random(int a, int b)
{
	if (a > b)
	{
		a += b;
		b = a - b;
		a -= b;
	}
	return rand() % (b - a + 1) + a;
}
std::vector<int> RandomManager::random(int a, int b, int n)
{
	std::vector<int> vec;
	for (int i = 0; i < n; i++)
		vec.push_back(random(a, b));
	return vec;
}

int RandomManager::random_index(std::vector<double> p)
{
	if (p.size() < 2)
		return 0;
	std::vector<double> intervals(p.size(), p[0]);
	for (int i = 1; i < int(p.size()); i++)
	{
		intervals[i] = intervals[i - 1] + p[i];
	}
	double r = ((double)rand() / RAND_MAX);
	for (int i = 0; i < int(intervals.size()); i++)
	{
		if (leq(r, intervals[i]))
			return i;
	}
	return 0;
}

std::vector<int> RandomManager::pruefer(int n)
{
	return random(1, n, n - 2);
}

bool RandomManager::event(double p)
{
	double r = ((double)rand() / RAND_MAX);
	return bool(r <= p);
}


bool RandomManager::leq(double a, double b)
{
	return (a < b || eq(a, b));
}

bool RandomManager::eq(double a, double b)
{
	return std::abs(a - b) < EPSILON;
}

bool RandomManager::beq(double a, double b)
{
	return (a > b || eq(a, b));
}































































