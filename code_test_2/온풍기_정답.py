import sys


def get_wind_diff(wind_diff, pos, delta, board, wall) :
    i, j, d = pos
    i = i + delta[d][0]
    j = j + delta[d][1]
    if (i, j) in wind_diff:
        wind_diff[(i, j)] += 5
    else:
        wind_diff[(i, j)] = 5

    heat = 4
    stack = [(i, j)]
    while stack and heat > 0:
        temp = set()
        while stack:
            i, j = stack.pop()
            ni = i + delta[d][0]
            nj = j + delta[d][1]

        # delta: 오0 왼1 상2 하3
        # 정면범위 벗어남, 종료
            if board[ni][nj] == -1:
                return

        # 정면 벽 없음, 정면 추가
            if not (i, j, d) in wall:
                temp.add((ni, nj))
            
        # 대각선 체크
            # 좌우10 일 때, 상하23 체크
            if d < 2:
                s = 2
                e = 4
            # 상하23 일 때, 좌우10 체크
            else:
                s = 0
                e = 2 
            for k in range(s, e):
                # k방향 벽 있음
                if (i, j, k) in wall:
                    continue
                
                # 벽 없음, 이동
                ni = i + delta[k][0]
                nj = j + delta[k][1]
                # 범위 안 체크, 정면 벽 체크, 추가
                if board[ni][nj] != -1 and not (ni, nj, d) in wall:
                    tni = ni + delta[d][0]
                    tnj = nj + delta[d][1]
                    temp.add((tni, tnj))

        stack = list(temp)
        # 온도변화 갱신
        for i, j in stack:
            if (i, j) in wind_diff:
                wind_diff[(i, j)] += heat
            else:
                wind_diff[(i, j)] = heat
        heat -= 1


def spread(R, C, delta, board, wall) :
    visited = set()
    diffusion = []
    for i in range(1, R+1):
        for j in range(1, C+1):
            if board[i][j] == 0:
                continue

            visited.add((i, j))
            for k in range(4):
                # 벽 존재
                if (i, j, k) in wall:
                    continue
                
                ni = i + delta[k][0]
                nj = j + delta[k][1]

                # 범위 밖 또는 이미 체크함
                if board[ni][nj] == -1 or (ni, nj) in visited:
                    continue

                # 체크
                t = board[i][j]
                nt = board[ni][nj]
                diff = abs(t - nt) // 4
                if t > nt:
                    diffusion.append((i, j, -diff))
                    diffusion.append((ni, nj, diff))
                elif t < nt:
                    diffusion.append((ni, nj, -diff))
                    diffusion.append((i, j, diff))

    # 변화 결과 갱신
    for i, j, diff in diffusion:
        board[i][j] += diff



def main():
    input = sys.stdin.readline
    R, C, K = map(int, input().split())
    board = [[-1] * (C+2)] + [[-1] + list(map(int, input().split())) + [-1] for _ in range(R)] + [[-1] * (C+2)]
    delta = [(0, 1), (0, -1), (-1, 0), (1, 0)]  # 오0 왼1 상2 하3
    heater = []
    sensor = []

    # 히터 정보, 센서 위치
    for i in range(1, R+1):
        for j in range(1, C+1):
            if board[i][j] == 0: continue
            if board[i][j] == 5:
                sensor.append((i, j))
            else:
                heater.append((i, j, board[i][j]-1))
            board[i][j] = 0

    # 벽 정보
    W = int(input())
    wall = set()
    for _ in range(W):
        r, c, d = map(int, input().split())
        if d == 0:  # 상
            wall.add((r, c, 2))
            wall.add((r-1, c, 3))
        else:   # 오른
            wall.add((r, c, 0))
            wall.add((r, c+1, 1))

    # 히터에 따른 좌표 온도 변화 정보
    wind_diff = dict()

    # 모든 히터 바람 온도 상승량 합산
    for pos in heater:
        get_wind_diff(wind_diff, pos, delta, board, wall)

    

    # 사이클
    time = 0
    while time < 100:
        # 1. 온풍기 바람 온도 상승
        for pos, diff in wind_diff.items():
            board[pos[0]][pos[1]] += diff
        
        for i in range(len(board)):
            print(board[i])
        print("-"* 50)
            
        # 2. 온도 조절 과정
        spread(R, C, delta, board, wall)

        for i in range(len(board)):
            print(board[i])
        print("-"* 50)

        # 3. 방 테두리 따라 온도 -1 일괄
        for k in range(2, R):
            if board[k][1] > 0:
                board[k][1] -= 1
            if board[k][C] > 0:
                board[k][C] -= 1
        for k in range(2, C):
            if board[1][k] > 0:
                board[1][k] -= 1
            if board[R][k] > 0:
                board[R][k] -= 1
        if board[1][1] > 0:
            board[1][1] -= 1
        if board[1][C] > 0:
            board[1][C] -= 1
        if board[R][1] > 0:
            board[R][1] -= 1
        if board[R][C] > 0:
            board[R][C] -= 1
        
        time += 1

        for i in range(len(board)):
            print(board[i])
        print("-"* 50)


        # 센서 값 체크
        for i, j in sensor:
            if board[i][j] < K:
                break
        else:
            print(time)
            return

    # 100을 넘는 경우에는 모두 101을 출력한다.
    print(101)


if __name__ == '__main__':
    main()