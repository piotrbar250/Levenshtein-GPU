//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
//#include <stdio.h>
//#include <iostream>
//#include <string.h>
//
//using namespace std;
//
//#define N 1000
//#define M 1000
//#define AL 26
//int dp[N][M];
//int X[AL][M];
//
//int n, m;
//
//void print()
//{
//    for (int i = 0; i <= n; i++)
//    {
//        for (int j = 0; j <= m; j++)
//            cout << dp[i][j] << " ";
//        cout << endl;
//    }
//}
//
//void printX()
//{
//    for (int i = 0; i < AL; i++)
//    {
//        cout << (char)(i + 65) << " ";
//        for (int j = 0; j <= m; j++)
//            cout << X[i][j] << " ";
//        cout << endl;
//    }
//}
//
//__global__ void testKernel(int dp[N][M], int n, int m)
//{
//    int i = blockIdx.x * blockDim.x + threadIdx.x;
//    int j = blockIdx.y * blockDim.y + threadIdx.y;
//
//    if (i <= n && j <= m) {
//        dp[i][j] = i * (m + 1) + j;
//    }
//}
//
//__global__ void computeX(int X[AL][M], char* text, int m)
//{
//    int i = blockIdx.x * blockDim.x + threadIdx.x;
//
//    // i powinno byc od 0 do 25
//    if (i < AL) {
//        X[i][0] = 0;
//        for (int j = 1; j <= m; j++)
//        {
//            if (i + 65 == (int)text[j - 1])
//            {
//                X[i][j] = j;
//            }
//            else
//                X[i][j] = X[i][j - 1];
//        }
//    }
//}
//
//int main()
//{
//    string t = "tekstowo";
//    string p = "pattern";
//
//    t = "1234567";
//    p = "123";
//
//    t = "CATGACTG";
//    p = "ACGT";
//
//    n = p.size();
//    m = t.size();
//
//    //int(*d_dp)[M];
//
//    //cudaMalloc((void**)&d_dp, N * M * sizeof(int));
//    //dim3 threadsPerBlock(16, 16); // Adjust as needed
//    //dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
//    //    (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
//
//    //testKernel << <numBlocks, threadsPerBlock >> > (d_dp, n, m);
//
//    //cudaMemcpy(dp, d_dp, sizeof(int) * N * M, cudaMemcpyDeviceToHost);
//    //cudaDeviceSynchronize();
//
//    //cout << t << endl << p << endl;
//    //
//    //print();
//
//
//    int(*device_X)[M];
//    cudaMalloc((void**)&device_X, AL * M * sizeof(int));
//
//
//    char* textC = const_cast<char*>(t.c_str());
//    char* device_text;
//    cudaMalloc((void**)&device_text, m + 1);
//    cudaMemcpy(device_text, textC, m + 1, cudaMemcpyHostToDevice);
//
//
//
//    computeX << <1, AL >> > (device_X, device_text, m);
//    cudaDeviceSynchronize();
//
//    cudaMemcpy(X, device_X, sizeof(int) * AL * M, cudaMemcpyDeviceToHost);
//
//    printX();
//}