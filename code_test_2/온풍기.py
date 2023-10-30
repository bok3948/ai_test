def spread(i, j, dir):
    a, b = dir_list[dir]
    n_i, n_j = i + a, j + b
    if (-1 < n_i < R and -1 < n_j < C) and ((i, j, dir) not in walls):
        board[n_i][n_j] += 5
        new_ava_loc = [(n_i, n_j)]

        for heat in range(4, 0, -1):
            cur_ava_loc = [loc for loc in new_ava_loc]
            new_ava_loc= []
            for n_i, n_j in cur_ava_loc:
                if a == 0: #좌우 바람

                    # 바람 방향 
                    if (-1 < n_i < R and -1 < n_j+b < C) and (n_i, n_j, dir) not in walls:
                        new_ava_loc.append((n_i, n_j+b)) 
        
                    # 대각 바람
                    #상 
                    if (-1 < n_i-1 < R and -1 < n_j+b < C) and (((n_i, n_j, 2) not in walls) and ((n_i-1, n_j, dir) not in walls)):
                        new_ava_loc.append((n_i-1, n_j+b))
                    #하
                    if (-1 < n_i+1 < R and -1 < n_j+b < C) and (((n_i, n_j, 3) not in walls) and ((n_i+1, n_j, dir) not in walls)):
                        new_ava_loc.append((n_i+1, n_j+b))

                else:
                     # 상, 하 바람 
                    if (-1 < n_i+a < R and -1 < n_j < C) and (n_i, n_j, dir) not in walls:
                        new_ava_loc.append((n_i+a, n_j)) 
        
                    # 대각 바람
                    #우
                    if (-1 < n_i+a < R and -1 < n_j
                        +1 < C) and (((n_i, n_j, 0) not in walls) and ((n_i, n_j+1, dir) not in walls)):
                        new_ava_loc.append((n_i+a, n_j+1))
                    
                    #좌
                    if (-1 < n_i+a < R and -1 < n_j-1 < C) and (((n_i, n_j, 1) not in walls) and ((n_i, n_j-1, dir) not in walls)):
                        new_ava_loc.append((n_i+a, n_j-1))
    
            new_ava_loc = list(set(new_ava_loc))
            for (h, w) in new_ava_loc:
                board[h][w] += heat
    return 
                
def distribution():
    tem_board = [[i for i in j] for j in board]

    for i in range(R):
        for j in range(C):

            for dir, (a, b) in enumerate(dir_list):
                n_i, n_j = i + a, j + b

                if -1< n_i < R and -1 < n_j < C: 

                    if (i, j, dir) in walls: 
                        continue
                             
                    d = abs(tem_board[n_i][n_j] - tem_board[i][j]) // 4
                    if tem_board[n_i][n_j] > tem_board[i][j]:
                        board[i][j] += d

                    else:
                        board[i][j] -= d
                else:
                    continue

    return 

def substract():
    for i in range(1, R-1):
        if board[i][0] > 0:
            board[i][0] -= 1

        if board[i][C-1] > 0:
            board[i][C-1] -= 1

    for j in range(1, C-1):
        if board[0][j] > 0:
            board[0][j] -= 1
        
        if board[R-1][j] > 0:
            board[R-1][j]  -= 1

    if board[0][C-1] > 0:
        board[0][C-1] -= 1 
    
    if board[0][0] > 0:
        board[0][0] -= 1 

    if board[R-1][C-1] > 0:
        board[R-1][C-1] -= 1 

    if board[R-1][0] > 0:
        board[R-1][0] -= 1 

    return 

def test():
    cho = 0
    while cho < 100:
        for i, j, dir in wind_gen:
            spread(i, j, dir)
        distribution()
        #for i in range(len(board)):
        #    print(board[i])
        #print("-"* 50)
        substract()

        cho += 1

        thr = True
        for i, j in sensor:
            if board[i][j] < K:
                thr = False

        if thr == True:
            break

    if cho >= 100:
        cho = 101
    return cho


if __name__ == '__main__':
    R, C, K = map(int, input().split())
    board = [[0 for _ in range(C)] for _ in range(R)]

    wind_gen, sensor = [], []
    for i in range(R):
        
        row =  list(map(int, input().split()))
        
        for j, v in enumerate(row):
            if 0 < v <= 4:
                wind_gen.append((i, j, v-1))
            elif v == 5:
                sensor.append((i, j))
            
    wall_num = int(input())
    walls = []
    for i in range(wall_num):
        x, y, t = map(int, input().split())
        x -= 1
        y -= 1
        if t == 0: # 상
            walls.append((x, y, 2))
            walls.append((x-1, y , 3))
        else: #오른        
            walls.append((x, y, 0))
            walls.append((x, y+1, 1))

    dir_list = [[0, 1], [0, -1], [-1, 0], [1, 0]]
    cho = test()
    print(cho)
    

        


            

        
            



