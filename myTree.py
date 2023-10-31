import math
import pandas
from sklearn.model_selection import train_test_split

class Node:
    def __init__(self, label: str, attributes = {}) -> None:
        self.label = label
        self.attributes = attributes
    def predict(self, data):
        if len(self.attributes) == 0:
            return self.label
        reduceData = dict(data)
        del reduceData[self.label]
        res = self.attributes[data[self.label]].predict(reduceData)
        return res
    
class MyDecisionTreeClassifier():
    def __init__(self, title: list, x_train, y_train, criterion='entropy') -> None:
        if len(title) != len(x_train[0]):
            raise ValueError('Parameters is not valid! size of title ({0}) different from number of data input ({1})'.format(len(title),len(x_train[0])))
        if criterion not in ['entropy', 'gini']:
            raise ValueError('Invalid value for paramater criterion!')
        self.title = title
        self.criterion = criterion
        self.yTrain = y_train
        self.xTrain = []
        if not str(type(x_train[0])) == "<class 'dict'>":
            for i in x_train:
                record = {}
                for index, val in enumerate(i):
                    record[str(self.title[index])] = str(val)
                self.xTrain.append(record)
        else:
            self.xTrain = x_train

    ## Hàm tính giá trị Infomation gain hoặc Gini index
    def point(self, data, className, title):
        att = set(map(lambda x: x[title], data))
        Point = 0 if self.criterion == 'entropy' else 1
        for i in att:
            yclass = []
            for index, val in enumerate(data):
                if val[title] == i:
                    yclass.append(className[index])
            numberOfClass = set(yclass)
            totalClass = len(yclass)
            subPoint = 0
            for j in numberOfClass:
                count = yclass.count(j)
                subPoint += -(count/totalClass)*math.log(count/totalClass) if self.criterion == 'entropy' else (count/totalClass)**2
            Point += totalClass/len(data)*subPoint if self.criterion == 'entropy' else -totalClass/len(data)*subPoint
        return Point

    ## Hàm xây dựng cây quyết định
    def buildTree(self, xData, yClass, labels: list):
        className = list(set(yClass))
        if len(className) == 1:
            return Node(className[0])
        if len(labels) == 0:
            count = []
            for i in className:
                count.append(yClass.count(i))
            return Node(str(className[count.index(max(count))]))
        
        Point = []
        for label in labels:
            Point.append(self.point(xData, yClass, label))
        
        minPointStr = labels.__getitem__(Point.index(min(Point)))
        listAtt = set(map(lambda x: x[minPointStr], xData))
        nodeAtt = {}
        labelsReduce = list(labels)
        labelsReduce.remove(minPointStr)
        for i in listAtt:
            subData = []
            subLabels = []
            for index, val in enumerate(xData):
                if val[minPointStr] == i:
                    subData.append(val)
                    subLabels.append(yClass[index])
            nodeAtt[i] = self.buildTree(subData, subLabels, labelsReduce)
        return Node(minPointStr, nodeAtt)
    
    def fit(self):
        self.tree = self.buildTree(self.xTrain, self.yTrain, self.title)

    def predict(self, data: list):
        if not str(type(data[0])) == "<class 'dict'>":
            newData = []
            for i in data:
                record = {}
                for index, val in enumerate(i):
                    record[(str(self.title[index]))] = val
                newData.append(record)
        else:
            newData = data
        return list(map(lambda x: self.tree.predict(x), newData))

try:
    data = pandas.read_csv('BThetmon/gianlanbaohiem.csv')
except:
    data = pandas.read_csv('gianlanbaohiem.csv')

data = data.drop(labels='policy_number', axis=1)
data = data.drop(labels='policy_bind_date', axis=1)
data = data.drop(labels='incident_location', axis=1)
data = data.drop(labels='insured_hobbies', axis=1)
data = data.drop(labels='months_as_customer', axis=1)


columns = (data.columns.values)
data_np = data.values
atts = columns[:len(columns)-1]

ENCODE_LABELS = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']

enum_cols = enumerate(columns)

for (i, col) in enum_cols:
    if(str(type(data_np[0,i])) != "<class 'str'>"):
        min_val = min(data[col].values)
        max_val = max(data[col].values)
        unique_count = len(set(data[col]))
        # print(f"{col}: min = {min_val}, max = {max_val}, lenght = {unique_count}")
        if unique_count > 10:
            split_range = []
            for split_index in range(10):
                split_range.append(min_val + ((max_val-min_val)/10*(split_index)))
            split_range.append(max_val+1)
            # print(split_range)
            for val in range(len(data[col].values)):
                for label_index in range(len(split_range)-1):
                    if (split_range[label_index] < float(data_np[val, i]) and float(data_np[val, i]) <= split_range[label_index+1]):
                        data_np[val, i] = ENCODE_LABELS[label_index]
                        break

train, test = train_test_split(data_np, train_size=0.7, test_size=0.3)
x, y = train[:,:-1], train[:,-1]
xt, yt = test[:,:-1], test[:,-1]

mytree = MyDecisionTreeClassifier(title=atts, x_train=x, y_train=y)
mytree.fit()

pred = []
for i in xt:
    try:
        pred.extend(mytree.predict([i]))
    except:
        pred.extend(['err'])
# print(pred)
print('err:', pred.count('err'))

count = 0
for i in range(len(pred)):
    if pred[i] != 'err':
        if pred[i] == yt[i]:
            count += 1
print('Score:', count/(len(pred)-pred.count('err')))