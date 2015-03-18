from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt
from sklearn import svm
import math

def main():
    #create the training & test sets, skipping the header row with [1:]
    dataset = genfromtxt(open('Data/train.csv','r'), delimiter=',', dtype='f8')[1:]
    target = [x[0] for x in dataset]
    train = [x[1:] for x in dataset]
    test = genfromtxt(open('Data/test.csv','r'), delimiter=',', dtype='f8')[1:]
    svc=svm.SVC(probability=True)
    svc.fit(train,target)



    predicted_probs = [[index + 1, x[1]] for index, x in enumerate(svc.predict_proba(test))]
    #print predicted_probs
    savetxt('Data/submission.csv', predicted_probs, delimiter=',', fmt='%d,%f',
            header='MoleculeId,PredictedProbability', comments = '')

if __name__=="__main__":
    main()