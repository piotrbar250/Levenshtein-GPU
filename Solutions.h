#pragma once
#include <string>
#include "Utils.cuh"

int min(int a, int b, int c);

int flatIndex(int row, int column, int columns);

void solveCPU(std::string p, std::string t, int * dp);

void printDp(int* dp, std::string p, std::string t, int n, int m);

void printDpOLD(int dp[N][N], std::string p, std::string t, int n, int m);

void printX(int* X, std::string p, std::string t, int m);

std::stack<std::string> getTransformations(std::string p, std::string t, int* dp);

void testFunction(std::string p, std::string t, int* dp);


void solveGPUnew(std::string p, std::string t, int* dp);
