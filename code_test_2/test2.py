from collections import deque
N, M = map(int, input().split())
adj = [[]for _ in range(N+1)]
for _ in range(M):
    sta, des = map(int, input().split())
    adj[sta].append(des)
    adj[des].append(sta)

def bfs(V):
    queue = deque()
    queue.append(V)
    visited[V] = 1
    while queue:
        V = queue.popleft()
        for nv in adj[V]:
            if visited[nv] == 0:
                visited[nv] = visited[V] + 1
                queue.append(nv)
    
    cnt = 0
    for i in range(1, N+1):
        cnt += visited[i] -1

    return cnt

ans = []
for i in range(1, N+1):
    visited = [0] * (N+1)
    ans.append(bfs(i))
print(ans.index(min(ans)) + 1)

