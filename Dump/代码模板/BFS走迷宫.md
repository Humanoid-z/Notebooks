~~~java
public static void bfs(int x, int y){
    q.add(x, y);  //将起点入队
    vis[x][y] = 1; // 标记已经走过
    res[x][y] = 0； // 标记起点的步数
        while(!q.isEmpty()){ // 当队列不空
            x = q.peek().x; //取出队头元素
            y = q.peek().y; 
            q.poll(); //删除队头元素
            if(x == n-1 && y == m-1) return; // 到达终点
            for(int i = 0; i < 4; i++){ // 当前点可以到达的下四个方向
                int nx = x + dx[i]; // 下个点的坐标
                int ny = y + dy[i]; 
                if(nx < 0 || nx >= n || ny < 0 || ny >= m || vis[nx][ny] == 1 ) continue; // 下个点不合法
                else{
                    res[nx][ny] = res[x][y] + 1; // 到达下一个点的步数为此父结点的步数加1
                    q.add(new pair(nx,ny));	//下个点合法，入队存储
                    vis[nx][ny] = 1; // 标记该点已经走过
                }
            }
        }
}
~~~

