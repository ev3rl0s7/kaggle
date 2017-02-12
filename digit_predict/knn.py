import pandas as pd
import numpy as np
import sklearn
import matplotlib as plt

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs

labels = train.ix[:,0]

train.drop('label', axis=1, inplace=True)
print (train)
from sklearn.neighbors import KNeighborsClassifier


def doWork(train, test, labels):
    print ("Converting training to matrix")
    train_mat = np.mat(train)
    print ("Fitting knn")
    knn = KNeighborsClassifier(n_neighbors=5, algorithm="kd_tree")
    print (knn.fit(train_mat, labels))
    print ("Preddicting")
    predictions = knn.predict(test)
    print ("Writing to file")
    write_to_file(predictions)
    return predictions


def write_to_file(predictions):
    f = open("output-knn-skilearn.csv", "w")
    for p in predictions:
        f.write(str(p))
        f.write("\n")
    f.close()


if __name__ == '__main__':

    predictions = doWork(train, test, labels)
print (predictions)
