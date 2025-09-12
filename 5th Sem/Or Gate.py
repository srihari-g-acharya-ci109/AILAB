w1=1
w2=1
t=1

values=[0,1]

print("x1\t x2\ty")
for i in values:
    for j in values:
        temp = 1 if ((w1 * i) + (w2 * (j))) >= t else 0
        print(i,"\t",j,"\t",temp)


