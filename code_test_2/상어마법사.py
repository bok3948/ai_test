def move_fish(fish_board):
    new_fish_board = [[[0]*4 for _ in range(4)] for _ in range(8)]

    for i in range(4):
        for j in range(4):
            for c in range(8):

                if fish_board[c][i][j] > 0:
                    a, b = fish_dir_list[c]
                    n_i, n_j = i + a, j + b
                    count = 0
                    old_c = c
                    
                    while (
                        (n_i < 0 or n_j < 0 or n_i > 3 or n_j > 3) or 
                        (s_i == n_i and s_j == n_j) or 
                        (smell_board_global[n_i][n_j] > 0)
                    ):
                        
                        c = (c - 1) % 8
                        a, b = fish_dir_list[c]
                        n_i, n_j = i + a, j + b
                        
                        if count > 8:
                            n_i, n_j = i, j
                            c = old_c
                            break
                            
                        count += 1
                    new_fish_board[c][n_i][n_j] += fish_board[old_c][i][j]
                    
    return new_fish_board 

def one_move(n_i, n_j, d, sum_fish_board, count):
    a, b = shark_dir_list[d]

    if (-1 < n_i + a < 4) and (-1 < n_j + b < 4):
        move = True
        count += sum_fish_board[n_i + a][n_j + b]
        sum_fish_board[n_i + a][n_j + b] = 0
    else:
        move = False

    return n_i + a, n_j + b, sum_fish_board, count, move  

def shark_move(s_i, s_j):
    
    poss_path_list, count_list = [], []
    for i in range(4):
        for j in range(4):
            for k in range(4):

                sum_fish_board = [[0 for x in range(4)] for y in range(4)]
                for x in range(8):
                    for y in range(4):
                        for z in range(4):
                            sum_fish_board[y][z] += fish_board[x][y][z]

                count = 0
                n_i, n_j = s_i, s_j

                for u in [i, j, k]:
                    n_i, n_j, sum_fish_board, count, move = one_move(n_i, n_j, u, sum_fish_board, count)

                    if move == False:
                        break
                    
                if move == True:
                    count_list.append(count)
                    poss_path_list.append([i, j, k])

    max_v  = max(count_list)
    for i in range(len(count_list)):
        if count_list[i] == max_v:
            return poss_path_list[i]
        
def erase_and_make_smell(fish_board, best_path):
    global smell_board_global, smell_board_memory

    n_i, n_j = s_i, s_j
    smell_board = [[0 for _ in range(4)] for _ in range(4)]
    for d in best_path:
        a, b = shark_dir_list[d]
        n_i, n_j = n_i + a , n_j + b
        for c in range(8):
            if fish_board[c][n_i][n_j] > 0:
                fish_board[c][n_i][n_j] = 0
                smell_board[n_i][n_j] = 1
    for i in range(4):
        for j in range(4):
            smell_board_global[i][j] += smell_board[i][j]
    smell_board_memory.append(smell_board)

    if s >= 2:
        for i in range(4):
            for j in range(4):
                smell_board_global[i][j] -= smell_board_memory[s-2][i][j]
   
    return fish_board, n_i, n_j

def copy(fish_board, fish_board_1):
    for c in range(8):
        for i in range(4):
            for j in range(4):
                fish_board[c][i][j] += fish_board_1[c][i][j]
    return fish_board

def count(fish_board):
    count = 0
    for c in range(8):
        for i in range(4):
            for j in range(4):
                count += fish_board[c][i][j]
    return count
    
if __name__ == '__main__':
    # 8(fish dir) x 4 x 4
    fish_board = [[[0]*4 for _ in range(4)] for _ in range(8)]
    smell_board_memory = []
    smell_board_global = [[0]*4 for _ in range(4)]
    fish_dir_list = [(0, -1), (-1, -1), (-1, 0), (-1, 1),
                    (0, 1),  (1, 1), (1, 0), (1, -1)]
    shark_dir_list = [(-1, 0), (0, -1),
                      (1, 0), (0, 1) ] #상좌하우
    #init
    M, S = map(int, input().split())
    for fish in range(M):
        i, j, d = map(int, input().split())
        fish_board[d-1][i-1][j-1] += 1

    s_i, s_j = map(int, input().split())
    s_i, s_j = s_i - 1, s_j -1

    for s in range(S):
        
        fish_board_1 = [[[k for k in j] for j in i] for i in fish_board]
        fish_board = move_fish(fish_board)
        best_path = shark_move(s_i, s_j)
        fish_board, s_i, s_j = erase_and_make_smell(fish_board, best_path)
        fish_board = copy(fish_board, fish_board_1)

        """
        su = [[0 for _ in range(4)] for _ in range(4)]
        di = []
        for c in range(8):
            for i in range(4):
                for j in range(4):
                    su[i][j] +=  fish_board[c][i][j]
                    if fish_board[c][i][j] > 0:
                        di.append((c, (i, j)))
        print(su)
        print(di)
        
        print(best_path, s_i, s_j)
        print(smell_board_global)
        
        print("-"*60)
        """

    num_fish = count(fish_board)
    print(num_fish)
        






        



    

    




    