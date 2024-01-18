#include <iostream>
#include <string.h>
#include <stack>

#include "Utils.h"
#include "Solutions.h"

void solveCPU(std::string p, std::string t, int dp[N][M])
{
	int n = p.size();
	int m = t.size();

	for (int i = 0; i <= n; i++)
		dp[i][0] = i;

	for (int j = 1; j <= m; j++)
		dp[0][j] = j;

	for (int i = 1; i <= n; i++)
	{
		for (int j = 1; j <= m; j++)
			if (t[j - 1] == p[i - 1])
				dp[i][j] = dp[i - 1][j - 1];
			else
				dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
	}
}

std::stack<std::string> getTransformations(std::string p, std::string t, int dp[N][M])
{
	std::stack<std::string> order;
	int i = p.size();
	int j = t.size();

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
			int minVal = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
			if (dp[i - 1][j] == minVal)
			{
				order.push((std::string)"DELETE " + p[i - 1]);
				i--;
			}
			else if (dp[i][j - 1] == minVal)
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

void printDp(int dp[N][N], std::string p, std::string t, int n, int m)
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