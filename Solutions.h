#pragma once
#include <string>
#include "Utils.h"

void solveGPU(std::string p, std::string t, int dp[N][M]);

int min(int a, int b, int c);

void solveCPU(std::string p, std::string t, int dp[N][M]);

void printDp(int dp[N][N], std::string p, std::string t, int n, int m);

std::stack<std::string> getTransformations(std::string p, std::string t, int dp[N][M]);