 TSP问题（旅行商问题）是指旅行家要旅行n个城市，要求各个城市经历且仅经历一次然后回到出发城市，并要求所走的路程最短。

![img](https://img-blog.csdnimg.cn/20190923192935826.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NTU5NjQx,size_16,color_FFFFFF,t_70)

# 状态转移方程

![image](https://p-blog.csdn.net/images/p_blog_csdn_net/gfaiswl/612517/o_image_thumb_2.gif)

# 状压

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200219232447898.png)

# 代码

dp\[i][S] 表示当前所在节点，从i出发经过S中所有顶点各一次后又回到题目规定点的最短总路径长度

~~~c++
#include<iostream>
#include<stdlib.h>
#include<algorithm>
#define INF 100
using namespace std;

int n=5;
int d[5][5] = { {0,3,INF,4,INF},
			   {INF,0,5,INF,INF},
			   {4,INF,0,5,INF},
			   {INF,INF,INF,0,3},
			   {7,6,INF,INF,0} };
int dp[5][1 << 4];

void DP() {
	for (int i = 0; i < n; i++)
		dp[i][0] = d[i][0];//题目规定起点为t，则这里为d[i][t] 即从i点直接去t点的距离
	//以列从左往右计算dp数组
	for (int S = 1; S < 1 << 4; S++) {
		for (int v = 0; v < n; v++) {
			dp[v][S] = INF;
			//当S中包含v时跳过
			if (((S >> (v - 1)) & 1) == 1) continue;

			//遍历所有顶点，
			for (int k = 1; k < 5; k++) {
				if (((S >> (k - 1)) & 1) == 0) continue;
                //遍历所有顶点，若S中包含点k，则例举v到k，再经过S/k到0的距离
				dp[v][S] = min(dp[v][S], dp[k][S ^ (1 << (k - 1))] + d[v][k]);
			}
		}
	}
}

int main() {
	DP();
	//dp[0][(1 << 4) - 1]表示从0经过{1,2,3,4}再到0
	cout << dp[0][(1 << 4) - 1];
	system("pause");
}

~~~

