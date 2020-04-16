# -*- coding: utf-8 -*-
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import math

featureMatrix = []
labelList = []

def returnLabel(label):
    if label == "notdga":
        return 0
    else:
        return 1

def Entropy(name):
    counter = Counter(name)
    entropy = 0
    for c, cn in counter.items():
        p = float(cn)/len(name)
        entropy += -1 * p * math.log(p, 2)
    return entropy

def feature(filename):
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line =="":
                continue
            tokens = line.split(",")
            name = tokens[0]
            label = returnLabel(tokens[1])
            labelList.append(label)
            llen = len(name)
            nnum = 0
            for i in name:
                if i.isdigit():
                    nnum+=1
            entro = Entropy(name)
            featureMatrix.append([llen,nnum,entro])
  
def predictdag(filename,preMatrix):
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line =="":
                continue
            name = line
            llen = len(name)
            nnum = 0
            for i in name:
                if i.isdigit():
                    nnum+=1
            entro = Entropy(name)
            preMatrix.append([llen,nnum,entro])
    return preMatrix     

def output(outcome):
    with open('test.txt') as fromfile:
        lines = fromfile.readlines()
        for i in range(0,len(lines)):
            if outcome[i] == 0 :
                lines[i] = lines[i].strip()+ ",notdga" + '\n' 
            else:
                lines[i] = lines[i].strip()+ ",dga" + '\n' 
    with open('result.txt','w') as outfile:
        outfile.writelines(lines)
    
            
def main():
    feature("train.txt")
    clf = RandomForestClassifier(random_state=0)
    clf.fit(featureMatrix,labelList)
    preMatrix = []
    preMatrix = predictdag("test.txt",preMatrix)
    outcome = clf.predict(preMatrix)
    output(outcome)
    
    
if __name__ == '__main__':
    main()
