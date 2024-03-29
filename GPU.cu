#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <string.h>
#include <stack>

#include "Utils.cuh"
#include "Timer.h"
#include "Solutions.h"

using namespace std;

__host__ __device__ int flatIndex(int row, int column, int columns)
{
	return row * columns + column;
}

__host__ __device__ int min(int a, int b, int c)
{
	if (a <= b && a <= c)
		return a;
	if (b <= a && b <= c)
		return b;
	return c;
}

__global__ void computeXKernel(int* X, char* text, int m)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < AL)
	{
		X[flatIndex(i, 0, (m + 1))] = 0;

		for (int j = 1; j <= m; j++)
		{
			if (i + 65 == (int)text[j - 1])
				X[flatIndex(i, j, m + 1)] = j;
			else
				X[flatIndex(i, j, m + 1)] = X[flatIndex(i, j - 1, m + 1)];
		}
	}
}

__global__ void computeDpKernel(int* dp, int* X, char* pattern, char* text, int n, int m, int i)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i <= n && j <= m)
	{
		if (i == 0)
			dp[flatIndex(i, j, m + 1)] = j;
		else if (j == 0)
			dp[flatIndex(i, j, m + 1)] = i;
		else if (text[j - 1] == pattern[i - 1])
			dp[flatIndex(i, j, m + 1)] = dp[flatIndex(i - 1, j - 1, m + 1)];
		else
		{
			int l = ((int)pattern[i - 1] - 65);
			if (X[flatIndex(l, j, m + 1)] == 0)
			{
				dp[flatIndex(i, j, m + 1)] = 1 + min(dp[flatIndex(i - 1, j, m + 1)], dp[flatIndex(i - 1, j - 1, m + 1)], i + j - 1);
			}
			else
			{
				dp[flatIndex(i, j, m + 1)] = 1 + min(dp[flatIndex(i - 1, j, m + 1)], dp[flatIndex(i - 1, j - 1, m + 1)],
					dp[flatIndex(i - 1, X[flatIndex(l, j, m + 1)] - 1, m + 1)] + j - 1 - X[flatIndex(l, j, m + 1)]);
			}
		}

	}
}


void solveGPU(string p, string t, int* dp)
{
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return;
	}

	int n = p.size();
	int m = t.size();

	int* device_X;
	cudaStatus = cudaMalloc((void**)&device_X, AL * (m + 1) * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	char* textC = const_cast<char*>(t.c_str());
	char* device_text;
	cudaStatus = cudaMalloc((void**)&device_text, m + 1);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(device_text, textC, m + 1, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	char* patternC = const_cast<char*>(p.c_str());
	char* device_pattern;
	cudaStatus = cudaMalloc((void**)&device_pattern, n + 1);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(device_pattern, patternC, n + 1, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	computeXKernel << <1, AL >> > (device_X, device_text, m);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "computeXKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaDeviceSynchronize();
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching computeXKernel!\n", cudaStatus);
		goto Error;
	}

	int* X = new int[AL * (m + 1)];
	cudaStatus = cudaMemcpy(X, device_X, sizeof(int) * AL * (m + 1), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	int* d_dp;
	cudaStatus = cudaMalloc((void**)&d_dp, (n + 1) * (m + 1) * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	dim3 threadsPerBlock(256);
	dim3 numBlocks((m + 1 + threadsPerBlock.x - 1) / threadsPerBlock.x);

	for (int i = 0; i <= n; i++)
	{
		computeDpKernel << <numBlocks, threadsPerBlock >> > (d_dp, device_X, device_pattern, device_text, n, m, i);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "computeDpKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		cudaDeviceSynchronize();
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching computeDpKernel!\n", cudaStatus);
			goto Error;
		}
	}

	cudaStatus = cudaMemcpy(dp, d_dp, sizeof(int) * (n + 1) * (m + 1), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	delete[] X;

Error:
	cudaFree(device_X);
	cudaFree(device_pattern);
	cudaFree(d_dp);
}