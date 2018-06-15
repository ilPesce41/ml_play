import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import svm

digits= datasets.load_digits()

clf = svm.SVC(gamma=.0001, C=100)

print(len(digits.data))

x,y = digits.data[:-500],digits.target[:-500]
clf.fit(x,y)

index = -148

print('Prediction:',clf.predict([digits.data[index]]))
plt.imshow(digits.images[index], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()