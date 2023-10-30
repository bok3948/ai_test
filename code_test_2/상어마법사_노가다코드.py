def fish_move(tank_fish_list, tank_dir_list):
    
    new_fish_tank_list, new_dir_tank_list = [], []
    for c in range(len(tank_fish_list)):
        new_one_fish_tank = [[0]*4 for _ in range(4)]
        new_one_dir_tank = [[0]*4 for _ in range(4)]
        for i in range(4):
            for j in range(4):

                #move fish
                if tank_fish_list[c][i][j] > 0:
                    
                    #change loc
                    d = tank_dir_list[c][i][j]
                    [a, b] = dir_list[d-1]
                    count = 0
                    n_i, n_j = i+a, j+b
                    

                    while (
                        ((i+a < 0) or (i+a > 3) or (j+b < 0) or (j+b > 3)) or 
                    (s_i == i+a and s_j == j+b) or 
                    (smell_tank_global[i+a][j+b] > 0)
                    ):
                        if  (s_i == i+a and s_j == j+b):
                            shark_pos_c = True

                        #make new dir
                        d -= 1
                        d = d % 8
                        
                        #update
                        [a, b]= dir_list[d-1] 

                        if count > 8:
                            #unmove
                            a, b = 0, 0
                            break
                        count += 1
            
                    #change loc
                    n_i, n_j = i+a, j+b
                    #make new tank
                    new_one_fish_tank[n_i][n_j] += 1
                    new_one_dir_tank[n_i][n_j] = d

        new_fish_tank_list.append(new_one_fish_tank)
        new_dir_tank_list.append(new_one_dir_tank)

    return new_fish_tank_list, new_dir_tank_list

def choose_shark_move(s_i, s_j, fish_tank):
    
    fish_tank = [[[k for k in j] for j in i] for i in fish_tank]

    # find possible dir 4^3s
    pass_list = []
    for i in range(4):
        for j in range(4):
            for k in range(4):

                #check possible
                fail = False
                tem_s_i, tem_s_j = s_i, s_j
                for u in [i, j, k]:
                    [a, b] = s_dir_list[u]
                    tem_s_i += a
                    tem_s_j += b
                    if (tem_s_i < 0 or tem_s_i >3 or tem_s_j < 0 or tem_s_j >3):
                        fail = True
                if fail == True:
                    continue
                else:
                    pass_list.append([i, j, k])
    
    #find best path
    count_list = []
    for i in range(len(pass_list)):
        #get 3 point
        [a, b] = s_dir_list[pass_list[i][0]]
        point1_i, point1_j = s_i + a, s_j + b

        [a, b] = s_dir_list[pass_list[i][1]]
        point2_i, point2_j = point1_i + a, point1_j + b

        [a, b] = s_dir_list[pass_list[i][2]]
        point3_i, point3_j = point2_i + a, point2_j + b

        #remove overlap
        pass_loc = list(set([(point1_i, point1_j), (point2_i, point2_j), (point3_i, point3_j)]))

        #count deleted num fish
        count = 0
        for c in range(len(fish_tank)):
            for j in range(len(pass_loc)):
                count += fish_tank[c][pass_loc[j][0]][pass_loc[j][1]] 

        count_list.append(count)
     
    #for trer, tyert in zip(pass_list, count_list):
    #   print(f"{trer} : {tyert}") 

    most_eat_path_list = []
    max_value = max(count_list)
    for e in range(len(count_list)):
        if count_list[e] == max_value:
            most_eat_path_list.append(pass_list[e])

    #print(most_eat_path_list)

    #find best path 

    #change ijk to int
    cho = []
    for path in most_eat_path_list:
        [i, j, k] = path
        cho.append((i+1)*100 + (j+1)*10 + k+1)

    #print(cho)

    min_v = min(cho)
    for i in range(len(cho)):
        if cho[i] == min_v:
            best_path = most_eat_path_list[i]

    #print(best_path)

    return best_path

def erase_and_smell(s_i, s_j, s_3_move, fish_tank, dir_tank):

    global smell_tank_global

    #move shark
    [a, b] = s_dir_list[s_3_move[0]]
    point1_i, point1_j = s_i + a, s_j + b

    [a, b] = s_dir_list[s_3_move[1]]
    point2_i, point2_j = point1_i + a, point1_j + b

    [a, b] = s_dir_list[s_3_move[2]]
    new_s_i, new_s_j = point2_i + a, point2_j + b

    points = [[point1_i, point1_j], [point2_i, point2_j], [new_s_i, new_s_j]]

    smell_tank = [[False]*4 for _ in range(4)]
    for point in points:
        for c in range(len(fish_tank)):
            if fish_tank[c][point[0]][point[1]] > 0:
                smell_tank[point[0]][point[1]] = True
                smell_tank_global[point[0]][point[1]] += 1
            fish_tank[c][point[0]][point[1]] = 0
            dir_tank[c][point[0]][point[1]] = 0

    smell_tank_memory.append(smell_tank)

    #erase smell -2
    if s >= 2:
        smell_tank_global = [[smell_tank_global[i][j] - smell_tank_memory[s-2][i][j] for j in range(4)] for i in range(4)]
        #print(smell_tank_global)

    #print(smell_tank_global)
    #print("-"* 70)
    #for smel in smell_tank_memory:
    #    print(smel)

    return new_s_i, new_s_j, fish_tank, dir_tank

def copy(fish_tank, dir_tank, fish_tank_1st, dir_tank_1st):
    for c in range(len(fish_tank)):
        fish_tank_1st.append(fish_tank[c])
        dir_tank_1st.append(dir_tank[c])
    return fish_tank_1st, dir_tank_1st

if __name__ == '__main__':
    M, S = map(int, input().split())
    tank_list = []


    smell_tank_global = [[0]*4 for i in range(4)] 
    smell_tank_memory = []
    shark_tank = [[0]*4 for i in range(4)]

    dir_list = [[0, -1],[-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1]]

    s_dir_list = [[-1, 0], [0, -1], [1, 0], [0, 1]]

    tank_fish_list, tank_dir_list = [], []
    for t in range(M):
        i, j, d = map(int, input().split())
        
        tank_one_fish = [[0]*4 for _ in range(4)]
        tank_one_dir = [[0]*4 for _ in range(4)]
        tank_one_fish[i-1][j-1] += 1
        tank_one_dir[i-1][j-1] = d

        tank_fish_list.append(tank_one_fish)
        tank_dir_list.append(tank_one_dir)
    
    s_i, s_j = map(int, input().split())
    s_i, s_j = s_i - 1, s_j - 1

    for s in range(S):

        #save
        tank_fish_list_1th = [[[k for k in j ] for j in i] for i in tank_fish_list] 
        tank_dir_list_1th = [[[k for k in j ] for j in i] for i in tank_dir_list]

        tank_fish_list, tank_dir_list = fish_move(tank_fish_list, tank_dir_list)

        #if s == 1:
         #   for y in range(len(tank_fish_list)):
         #       print(tank_dir_list[y])
         #   print("-"*50)
        #for y in range(len(tank_fish_list)):
        #    print(tank_fish_list[y])
        #print("-"*50)
            
        #print(tank_fish_list_1th)
        s_3_move = choose_shark_move(s_i, s_j, tank_fish_list)

        s_i, s_j, tank_fish_list, tank_dir_list =  erase_and_smell(s_i, s_j, s_3_move, tank_fish_list, tank_dir_list)
        #for y in range(len(tank_fish_list)):
        #    print(f"{tank_fish_list[y]}: {tank_dir_list[y]}")
        #print("-"*50)

       
        
        tank_fish_list, tank_dir_list = copy(tank_fish_list, tank_dir_list, tank_fish_list_1th, tank_dir_list_1th)
        #for y in range(len(tank_fish_list)):
        #    print(f"{tank_fish_list[y]}")
        #    print(tank_dir_list[y])
        #    print()
        #print("-"*50)
        #print(smell_tank_global)
        #print(s_i, s_j)

    all_sum = 0
    for ru in range(len(tank_fish_list)):
        for g in range(4):
            all_sum += sum(tank_fish_list[ru][g])
    print(all_sum)
