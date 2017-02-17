import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn import svm

mnist = fetch_mldata('MNIST original')

x = mnist.data
y = mnist.target

x4 = x[y==4,:]
x9 = x[y==9,:]
training = x4[0:3499,:]
training = np.concatenate((training, x9[0:3499,:]))
holdout = np.concatenate((x4[3500:3999, :], x9[3500:3999, :]))
testing = np.append(x4[4000:, :], x9[4000:, :])
ytrain=np.ones((3499,1))
ytrain=np.append(ytrain*4, ytrain*9)

y_holdout = np.ones((499,1))
y_holdout = np.append(y_holdout*4, y_holdout*9)
#ytrain = np.append(np.ones((3499,1))*4, np.ones((3499,1)*9))

clf = svm.SVC(C=1., kernel='linear')
clf.fit(training, ytrain)



Pe=1-clf.score(holdout, y_holdout)






j=4

plt.title('The {j}th image is a {label}'.format(j=j, label=int(y[j])))
plt.imshow(x[j].reshape((28,28)), cmap='gray')
plt.show()
