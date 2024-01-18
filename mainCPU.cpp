//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
//#include <stdio.h>
//#include <iostream>
//#include <string.h>
//#include <stack>
//
//#include "Utils.h"
//#include "Timer.h"
//
//using namespace std;
//
//int dp[N][M];
//int n, m;
//string t, p;
//
//void print()
//{
//	cout << "    ";
//	for (int i = 0; i < m; i++)
//		cout << t[i] << " ";
//	cout << endl;
//	for (int i = 0; i <= n; i++)
//	{
//		for (int j = 0; j <= m; j++)
//		{
//			if (i == 0 && j == 0)
//				cout << "  ";
//			else if (j == 0)
//				cout << p[i - 1] << " ";
//
//			cout << dp[i][j] << " ";
//		}
//		cout << endl;
//	}
//}
//
//int min(int a, int b, int c)
//{
//	if (a <= b && a <= c)
//		return a;
//	if (b <= a && b <= c)
//		return b;
//	return c;
//}
//
//int main()
//{
//	t = "CATGACTG";
//	p = "TACTG";
//
//	n = p.size();
//	m = t.size();
//
//	stack<string> order;
//	{
//		Timer timer;
//
//		for (int i = 0; i <= n; i++)
//			dp[i][0] = i;
//
//		for (int j = 1; j <= m; j++)
//			dp[0][j] = j;
//
//		for (int i = 1; i <= n; i++)
//		{
//			for (int j = 1; j <= m; j++)
//				if (t[j - 1] == p[i - 1])
//					dp[i][j] = dp[i - 1][j - 1];
//				else
//					dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
//		}
//
//		int i = n;
//		int j = m;
//
//		while (i != 0 || j != 0)
//		{
//			if (i == 0)
//			{
//				order.push((string)"ADD " + t[j - 1]);
//				j--;
//			}
//			else if (j == 0)
//			{
//				order.push((string)"DELETE " + t[i - 1]);
//				i--;
//			}
//			else if (t[j - 1] == p[i - 1])
//			{
//				order.push((string)"PROCEED");
//				i--;
//				j--;
//			}
//			else
//			{
//				int minVal = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
//				if (dp[i - 1][j] == minVal)
//				{
//					order.push((string)"DELETE " + p[i - 1]);
//					i--;
//				}
//				else if (dp[i][j - 1] == minVal)
//				{
//					order.push((string)"ADD " + t[j - 1]);
//					j--;
//				}
//				else
//				{
//					order.push((string)"SUBSTITUTE " + p[i - 1] + " WITH " + t[j - 1]);
//					i--;
//					j--;
//				}
//			}
//		}
//	}
//
//	print();
//
//	while (!order.empty())
//	{
//		cout << order.top() << endl;
//		order.pop();
//	}
//
//}