#include <iostream>
#include <string.h>
#include <stack>

#include "Utils.cuh"
#include "Solutions.h"

//void solveCPU(std::string p, std::string t, int* dp)
//{
//	int n = p.size();
//	int m = t.size();
//
//	for (int i = 0; i <= n; i++)
//		dp[i][0] = i;
//	
//	for (int j = 1; j <= m; j++)
//		dp[0][j] = j;
//
//	for (int i = 1; i <= n; i++)
//	{
//		for (int j = 1; j <= m; j++)
//			if (t[j - 1] == p[i - 1])
//				dp[i][j] = dp[i - 1][j - 1];
//			else
//				dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
//	}
//}

void solveCPU(std::string p, std::string t, int * dp)
{
	int n = p.size();
	int m = t.size();

	for (int i = 0; i <= n; i++)
	{
		dp[flatIndex(i, 0, (m + 1))] = i;
	}

	for (int j = 1; j <= m; j++)
	{
		dp[flatIndex(0, j, (m + 1))] = j;
	}

	for (int i = 1; i <= n; i++)
	{
		for (int j = 1; j <= m; j++)
			if (t[j - 1] == p[i - 1])
				dp[flatIndex(i, j, (m + 1))] = dp[flatIndex(i - 1, j - 1, (m + 1))];
			else
				dp[flatIndex(i, j, (m + 1))] = 1 + min(dp[flatIndex(i - 1, j, (m + 1))],
					dp[flatIndex(i, j - 1, (m + 1))], dp[flatIndex(i - 1, j - 1, (m + 1))]);
	}
}


std::stack<std::string> getTransformations(std::string p, std::string t, int* dp)
{
	std::stack<std::string> order;
	int i = p.size();
	int j = t.size();

	int m = t.size();

	while (i != 0 || j != 0)
	{
		if (i == 0) 
		{
			order.push((std::string)"ADD " + t[j - 1]);
			j--;
		}
		else if (j == 0)
		{
			order.push((std::string)"DELETE " + p[i - 1]);
			i--;
		}
		else if (t[j - 1] == p[i - 1])
		{
			order.push((std::string)"PROCEED");
			i--;
			j--;
		}
		else
		{
			int minVal = min(dp[flatIndex(i - 1, j, (m + 1))],
				dp[flatIndex(i, j - 1, (m + 1))], dp[flatIndex(i - 1, j - 1, (m + 1))]);

			if (dp[flatIndex(i - 1, j, (m + 1))] == minVal)
			{
				order.push((std::string)"DELETE " + p[i - 1]);
				i--;
			}
			else if (dp[flatIndex(i, j - 1, (m + 1))] == minVal)
			{
				order.push((std::string)"ADD " + t[j - 1]);
				j--;
			}
			else
			{
				order.push((std::string)"SUBSTITUTE " + p[i - 1] + " WITH " + t[j - 1]);
				i--;
				j--;
			}
		}
	}

	return order;
}

void printDpOLD(int dp[N][N], std::string p, std::string t, int n, int m)
{
	std::cout << "    ";
	for (int i = 0; i < m; i++)
		std::cout << t[i] << " ";
	std::cout << std::endl;
	for (int i = 0; i <= n; i++)
	{
		for (int j = 0; j <= m; j++)
		{
			if (i == 0 && j == 0)
				std::cout << "  ";
			else if (j == 0)
				std::cout << p[i - 1] << " ";

			std::cout << dp[i][j] << " ";
		}
		std::cout << std::endl;
	}
}

void printDp(int* dp, std::string p, std::string t, int n, int m)
{
	
	std::cout << "    ";
	for (int i = 0; i < m; i++)
		std::cout << t[i] << " ";
	std::cout << std::endl;
	for (int i = 0; i <= n; i++)
	{
		for (int j = 0; j <= m; j++)
		{
			if (i == 0 && j == 0)
				std::cout << "  ";
			else if (j == 0)
				std::cout << p[i - 1] << " ";

			std::cout << dp[flatIndex(i, j, (m + 1))] << " ";
		}
		std::cout << std::endl;
	}
}

void printX(int* X, std::string p, std::string t, int m)
{
	for (int i = 0; i < AL; i++)
	{
		std::cout << char(i + 65) << " ";
		for (int j = 0; j <= m; j++)
		{
			std::cout << X[flatIndex(i, j, (m + 1))] << " ";
		}
		std::cout << std::endl;
	}
}