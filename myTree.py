import math
import pandas
from sklearn.model_selection import train_test_split

class Node:
    def __init__(self, name: str, attributes = {}) -> None:
        self.name = name
        self.attributes = attributes
    def predict(self, data):
        if len(self.attributes) == 0:
            return self.name
        reduceData = dict(data)
        del reduceData[self.name]
        data_att_val = data[self.name]
        next_node = self.attributes[data_att_val]
        result = next_node.predict(reduceData)
        return result
    
class MyDecisionTreeClassifier():
    def __init__(self, attributes: list, x_train, y_train, criterion='entropy') -> None:
        if len(attributes) != len(x_train[0]):
            raise ValueError(f'Parameters is not valid! size of attributes ({len(attributes)}) is different from number of data input ({len(x_train[0])})')
        if criterion not in ['entropy', 'gini']:
            raise ValueError('Invalid value for paramater criterion!')
        self.attributes = attributes
        self.criterion = criterion
        self.yTrain = y_train
        self.xTrain = []
        if not str(type(x_train[0])) == "<class 'dict'>":
            for i in x_train:
                record = {}
                for index, val in enumerate(i):
                    record[str(self.attributes[index])] = str(val)
                self.xTrain.append(record)
        else:
            self.xTrain = x_train

    ## Hàm tính giá trị Infomation gain hoặc Gini index
    def point(self, data, yClass, attribute_name):
        props = set(map(lambda x: x[attribute_name], data))
        Point = 0 if self.criterion == 'entropy' else 1
        for i in props:
            prop_targets = []
            for index, val in enumerate(data):
                if val[attribute_name] == i:
                    prop_targets.append(yClass[index])
            prop_classes = set(prop_targets)
            total_prop_target = len(prop_targets)

            prop_point = 0
            for prop_class in prop_classes:
                count_class = prop_targets.count(prop_class)
                prop_point += -(count_class/total_prop_target)*math.log(count_class/total_prop_target) if self.criterion == 'entropy' else (count_class/total_prop_target)**2
            Point += total_prop_target/len(data)*prop_point if self.criterion == 'entropy' else -total_prop_target/len(data)*prop_point
        return Point

    ## Hàm xây dựng cây quyết định
    def buildTree(self, xData, yClass, attributes: list):
        target_list = list(set(yClass))
        if len(target_list) == 1:
            return Node(target_list[0])
        if len(attributes) == 0:
            count = []
            for target in target_list:
                count.append(yClass.count(target))
            target_max_index = count.index(max(count))
            return Node(str(target_list[target_max_index]))
        
        Point = []
        for attribute_name in attributes:
            Point.append(self.point(xData, yClass, attribute_name))
        
        min_point_attribute_name = attributes[Point.index(min(Point))]
        list_att_props = set(map(lambda x: x[min_point_attribute_name], xData))

        att_prop_nodes = {}
        attributesReduce = list(attributes)
        attributesReduce.remove(min_point_attribute_name)

        for att_prop in list_att_props:
            prop_data = []
            att_prop_targets = []
            for index, val in enumerate(xData):
                if val[min_point_attribute_name] == att_prop:
                    prop_data.append(val)
                    att_prop_targets.append(yClass[index])
            att_prop_nodes[str(att_prop)] = self.buildTree(prop_data, att_prop_targets, attributesReduce)
        return Node(min_point_attribute_name, att_prop_nodes)
    
    def fit(self):
        self.tree = self.buildTree(self.xTrain, self.yTrain, self.attributes)

    def predict(self, data: list):
        if not str(type(data[0])) == "<class 'dict'>":
            newData = []
            for i in data:
                record = {}
                for index, val in enumerate(i):
                    record[(str(self.attributes[index]))] = val
                newData.append(record)
        else:
            newData = data
        return list(map(lambda x: self.tree.predict(x), newData))


#####
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

# Rời rạc dữ liệu
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

train, test = train_test_split(data_np, test_size=0.3, shuffle=True)
x, y = train[:,:-1], train[:,-1]
xt, yt = test[:,:-1], test[:,-1]

mytree = MyDecisionTreeClassifier(attributes=atts, x_train=x, y_train=y)
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