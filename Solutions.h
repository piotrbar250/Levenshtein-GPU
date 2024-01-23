#pragma once
#include <string>
#include "Utils.cuh"

int min(int a, int b, int c);

int flatIndex(int row, int column, int columns);

void solveCPU(std::string p, std::string t, int * dp);

void solveGPU(std::string p, std::string t, int* dp);

void printDp(int* dp, std::string p, std::string t, int n, int m);

std::stack<std::string> getTransformations(std::string p, std::string t, int* dp);