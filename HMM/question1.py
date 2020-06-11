import numpy as np
from collections import Counter

x = ['the man saw the cut'.split(), 'the saw cut the man'.split(), 'the saw'.split()]
y = ['D N V D N'.split(), 'D N V D N'.split(), 'N N'.split()]

tagcounts = Counter()
bitagcounts = Counter()
paircounts = Counter()
o = Counter()
t = Counter()
print(x)

for i in range(3):
    for j in range(len(y[i])):
        tagcounts[y[i][j]] += 1
        paircounts[tuple(x[i][j] + y[i][j])] += 1
        if j + 1 <= 3: 
            bitagcounts[tuple(y[i][j:j+2])] += 1

print(tagcounts, bitagcounts, paircounts)

for i in range(3):
    for j in range(1, len(y[i])):
        t[tuple(y[i][j] + '|' + y[i][j-1])] = bitagcounts[tuple(y[i][j-1:j+1])] / tagcounts[y[i][j-1]]
    for j in range(len(y[i])):
        o[tuple(x[i][j] + '|' + y[i][j])] = paircounts[tuple(x[i][j] + y[i][j])] / tagcounts[y[i][j]]

print(o, t)
        
DNVDN = 1
prob = [1/3, 1/6, 1/6, 1/3]
test = ['D N V D N'.split(), 'N N V D N'.split(), 'N N N N N'.split(), 'D N N N N'.split()]
seq = 'the saw cut the man'.split()
for i in range(4):
    for j in range(4):
        prob[i] *= t[tuple(test[i][j+1] + '|' + test[i][j])]
    for j in range(5):
        prob[i] *= o[tuple(seq[j] + '|' + test[i][j])]

print(prob)
prediction = (prob[0] + prob[1]) / (prob[0] + prob[1]+ prob[2]+ prob[3])
print(prediction)
