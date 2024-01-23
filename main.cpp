#include <iostream>
#include <fstream>
#include <chrono>
#include <stack>

#include "Utils.cuh"
#include "Timer.h"
#include "Solutions.h"

using namespace std;

int main(int argc, char** argv)
{
	std::string path = "";
	bool cpu = false;
	bool gpu = false;

	for (int i = 1; i < argc; i++)
	{
		if (string(argv[i]) == "--file")
			path = argv[++i];
		else if (string(argv[i]) == "--cpu")
			cpu = true;
		else if (string(argv[i]) == "--gpu")
			gpu = true;
	}

	if (!(cpu || gpu))
		cpu = gpu = true;

	std::string p = "TACTG";
	std::string t = "CATGACTG";

	if (path != "")
	{
		fstream file;
		file.open(path, ios::in);
		if (file.is_open())
		{
			std::getline(file, p);
			std::getline(file, t);
			file.close();
		}
		else
			std::cout << "Unable to open file" << std::endl;
	}

	std::stack<std::string> order;

	int n = p.size();
	int m = t.size();
	int* dp = new int[(n + 1) * (m + 1)];

	//---------------------------------------CPU---------------------------------------------
	if(cpu)
	{
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
	}


	//---------------------------------------GPU----------------------------------------------
	if (gpu)
	{
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
	}
	
	delete[] dp;

	return 0;
}