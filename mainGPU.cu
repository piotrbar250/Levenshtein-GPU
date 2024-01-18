#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <string.h>
#include <stack>

#include "Utils.h"
#include "Timer.h"

using namespace std;

int dp[N][M];
int X[AL][M];
string t, p;
int n, m;

void print()
{
	cout << "    ";
	for (int i = 0; i < m; i++)
		cout << t[i] << " ";
	cout << endl;
	for (int i = 0; i <= n; i++)
	{
		for (int j = 0; j <= m; j++)
		{
			if (i == 0 && j == 0)
				cout << "  ";
			else if (j == 0)
				cout << p[i - 1] << " ";

			cout << dp[i][j] << " ";
		}
		cout << endl;
	}
}

void printX()
{
	for (int i = 0; i < AL; i++)
	{
		cout << (char)(i + 65) << " ";
		for (int j = 0; j <= m; j++)
			cout << X[i][j] << " ";
		cout << endl;
	}
}

__host__ __device__ int min(int a, int b, int c)
{
	if (a <= b && a <= c)
		return a;
	if (b <= a && b <= c)
		return b;
	return c;
}

__global__ void testKernel(int dp[N][M], int n, int m)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i <= n && j <= m) {
		dp[i][j] = i * (m + 1) + j;
	}
}

__global__ void computeXKernel(int X[AL][M], char* text, int m)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < AL) {
		X[i][0] = 0;
		for (int j = 1; j <= m; j++)
		{
			if (i + 65 == (int)text[j - 1])
				X[i][j] = j;
			else
				X[i][j] = X[i][j - 1];
		}
	}
}

__global__ void computeDpKernel(int dp[N][M], int X[AL][M], char* pattern, char* text, int n, int m, int i)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i <= n && j <= m)
	{
		if (i == 0)
			dp[i][j] = j;
		else if (j == 0)
			dp[i][j] = i;
		else if (text[j - 1] == pattern[i - 1])
			dp[i][j] = dp[i - 1][j - 1];
		else
		{
			int l = ((int)pattern[i - 1] - 65);
			if (X[l][j] == 0)
			{
				dp[i][j] = 1 + min(dp[i - 1][j], dp[i - 1][j - 1], i + j - 1);
			}
			else
			{
				dp[i][j] = 1 + min(dp[i - 1][j], dp[i - 1][j - 1], dp[i - 1][X[l][j] - 1] + (j - 1 - X[l][j]));
			}
		}
	}
}

int main()
{
	t = "CATGACTG";
	p = "TACTG";

	{
		n = p.size();
		m = t.size();

		stack<string> order;
		{
			Timer timer;

			int(*device_X)[M];
			cudaMalloc((void**)&device_X, AL * M * sizeof(int));

			char* textC = const_cast<char*>(t.c_str());
			char* device_text;
			cudaMalloc((void**)&device_text, m + 1);
			cudaMemcpy(device_text, textC, m + 1, cudaMemcpyHostToDevice);

			char* patternC = const_cast<char*>(p.c_str());
			char* device_pattern;
			cudaMalloc((void**)&device_pattern, n + 1);
			cudaMemcpy(device_pattern, patternC, n + 1, cudaMemcpyHostToDevice);

			computeXKernel << <1, AL >> > (device_X, device_text, m);
			cudaDeviceSynchronize();

			int(*d_dp)[M];
			cudaMalloc((void**)&d_dp, N * M * sizeof(int));

			dim3 threadsPerBlock(256);
			dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x);

			for (int i = 0; i <= n; i++)
			{
				computeDpKernel << <numBlocks, threadsPerBlock >> > (d_dp, device_X, device_pattern, device_text, n, m, i);
				cudaDeviceSynchronize();
			}

			cudaMemcpy(dp, d_dp, sizeof(int) * N * M, cudaMemcpyDeviceToHost);

			int i = n;
			int j = m;

			while (i != 0 || j != 0)
			{
				if (i == 0)
				{
					order.push((string)"ADD " + t[j - 1]);
					j--;
				}
				else if (j == 0)
				{
					order.push((string)"DELETE " + t[i - 1]);
					i--;
				}
				else if (t[j - 1] == p[i - 1])
				{
					order.push((string)"PROCEED");
					i--;
					j--;
				}
				else
				{
					int minVal = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
					if (dp[i - 1][j] == minVal)
					{
						order.push((string)"DELETE " + p[i - 1]);
						i--;
					}
					else if (dp[i][j - 1] == minVal)
					{
						order.push((string)"ADD " + t[j - 1]);
						j--;
					}
					else
					{
						order.push((string)"SUBSTITUTE " + p[i - 1] + " WITH " + t[j - 1]);
						i--;
						j--;
					}
				}
			}
		}
		print();
		while (!order.empty())
		{
			cout << order.top() << endl;
			order.pop();
		}
	}
	return 0;
}