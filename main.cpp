#include <iostream>
#include <chrono>
#include <stack>

#include "Utils.h"
#include "Timer.h"
#include "Solutions.h"

int dp[N][M];

int main()
{
	std::string p = "TACTG";
	std::string t = "CATGACTG";

	p = "INTENTION";
	t = "EXECUTION";

	std::stack<std::string> order;

	//---------------------------------------CPU---------------------------------------------
	std::cout << "CPU" << std::endl;
	{
		Timer timer;
		solveCPU(p, t, dp);
		order = getTransformations(p, t, dp);
	}

	std::cout << std::endl;
	printDp(dp, p, t, p.size(), t.size());
	std::cout << std::endl;
	while (!order.empty())
	{
		std::cout << order.top() << std::endl;
		order.pop();
	}
	std::cout << std::endl << std::endl;


	//---------------------------------------GPU----------------------------------------------
	std::cout << "GPU" << std::endl;
	{
		Timer timer;
		solveGPU(p, t, dp);
		order = getTransformations(p, t, dp);
	}
	std::cout << std::endl;
	printDp(dp, p, t, p.size(), t.size());
	std::cout << std::endl;
	while (!order.empty())
	{
		std::cout << order.top() << std::endl;
		order.pop();
	}
	
	return 0;
}