T = int(input())

ans = []
for i in range(T):
    N = int(input())
    woods = list(map(int, input().split()))
    woods.sort()
    pa_woods = woods[0::2]
    print(pa_woods)
    od_woods = [i for i in woods[1::2]]
    print(od_woods)
    max_v = 0
    for i in range(len(pa_woods) -1):
        d = pa_woods[i+1] - pa_woods[i]
        if d > max_v:
            max_v = d
    for i in range(len(od_woods) -1):
        d = od_woods[i+1] - od_woods[i]
        if d > max_v:
            max_v = d
    ans.append(max_v)

for i in range(len(ans)):
    print(ans[i])

        





    







    








    


    




    




















    
