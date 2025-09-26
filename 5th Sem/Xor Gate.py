#xor
def andgate(x1, x2):
    return 1 if (x1 + x2) >= 2 else 0


def orgate(x1, x2):
    return 1 if (x1 + x2) >= 1 else 0


def notgate(x):
    return 1 if (-1 * x) >= 0 else 0


def xorgate(x1, x2):
    orans = orgate(x1, x2)
    andans = andgate(x1, x2)
    notofand = notgate(andans)
    return andgate(orans, notofand)


values = [0, 1]
print("x1\t x2\t XOR")
for i in values:
    for j in values:
        print(i, "\t", j, "\t", xorgate(i, j))
