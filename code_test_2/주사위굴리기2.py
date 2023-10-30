#아래 6, 동 3
def move():
    A = east[1]
    B = board[0][1]
    h_i, w_j = 0, 1
    cur_dir = 0
    if A < B:
        cur_dir = (cur_dir+1) // 4
    elif B == A:
        pass
    else:
        cur_dir = (cur_dir-1) // 4
    res = 0

    res += N - (h_i+1)
    for i in range(N - (h_i+1)):
        if board[h_i+i][w_j] == B:
            
    res += abs(1 - (h_i+1))
    res += M - (w_j+1)
    res += abs(1 - (w_j+1))
    
    
    


if '__name__' == '__main__':
    N, M, K = map(int, input().split())
    board = []
    delta = [0, 1), (1, 0), (0, -1), (-1, 0)]
    turned = [
    (3, 1, 0, 5, 4, 2),
    (1, 5, 2, 3, 0, 4),
    (2, 1, 5, 0, 4, 3),
    (4, 0, 2, 3, 5, 1)
]
    for i in range(N):
        row = map(int, input().split())
        board.append(row)

    south = [6, 2, 1, 5] #남로 굴러갈때
    east = [6, 3, 1, 4] #동쪽으로 
    dir_list = [(0, 1), (1. 0), (0, -1), (-1, 0)] #동남서북 시계방향


