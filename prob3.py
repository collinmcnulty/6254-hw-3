import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn import svm
import math
import heapq

mnist = fetch_mldata('MNIST original')

x = mnist.data
y = mnist.target

x4 = x[y==4,:]
x9 = x[y==9,:]
total = np.concatenate((x4, x9))
ytotal = np.concatenate((np.ones(len(x4))*4, np.ones(len(x9))*9))
training = x4[0:3499,:]
training = np.concatenate((training, x9[0:3499,:]))
holdout = np.concatenate((x4[3500:3999, :], x9[3500:3999, :]))
testing = np.append(x4[4000:, :], x9[4000:, :])
ytrain=np.ones((3499,1))
ytrain=np.append(ytrain*4, ytrain*9)

y_holdout = np.ones((499,1))
y_holdout = np.append(y_holdout*4, y_holdout*9)


clf = svm.SVC(C=1., kernel='linear')
clf.fit(training, ytrain)







Clist=range(2,2,1)
C=[]
for element in Clist:
    C.append(math.exp(float(element)))

Pe=[]
machines = []
for c in C:
    clf = svm.SVC(C=c, kernel='poly', degree=1)
    clf.fit(training, ytrain)
    Pe.append(1 - clf.score(holdout, y_holdout))
    machines.append(clf)

best_score= min(Pe)
best_C = C[Pe.index(best_score)]
best_machine = machines[Pe.index(best_score)]
a=best_machine.decision_function(total)
misclassified = []
for index, element in enumerate(a):
    if ytotal[index] == 4.:
        if element > 0:
            misclassified.append([index, abs(element)])
    elif ytotal[index] == 9.:
        if element < 0:
            misclassified.append([index, abs(element)])
    else:
        1

if misclassified == []:
    worst16_values = heapq.nsmallest(16, a)
else:
    worst16_values = heapq.nlargest(16, misclassified[:,2])

worst16 = []
for index, element in enumerate(worst16_values):
    worst16.append(a.tolist().index(element))

f, axarr = plt.subplots(4, 4)
for i in range(0,15):
    j = worst16[i]
    axarr[(i/4), i%4].imshow(x[j].reshape((28,28)), cmap='gray')
    axarr[i/4, i%4].set_title('{label}'.format(label=int(ytotal[j])))

plt.show()

print('The best C is {C} with an error probability of {p}'.format(C=best_C, p=best_score))

# plt.title('The {j}th image is a {label}'.format(j=j, label=int(y[j])))
# plt.imshow(x[j].reshape((28,28)), cmap='gray')
# plt.show()
