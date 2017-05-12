import CNN

for i in range(100):
    featlist = []
    for j in range(1024):
        featlist += [j]
    a,b,c=CNN.process(featlist)
    print(a)
