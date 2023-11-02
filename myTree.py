from math import log
import pandas
from sklearn.model_selection import train_test_split

class Node:
    def __init__(self, name: str, branches={}) -> None:
        self.name = name
        self.branches = branches

    def predict(self, data):

        # Nếu node không có nhánh con thì trả về tên node (giá trị cần tìm)
        if len(self.branches) == 0:
            return self.name
        

        data_att_val = data[self.name]                  # Lấy ra tên nhánh cần đi tiếp từ data
        next_node = self.branches[data_att_val]         # Lấy ra node của nhánh cần đi tiếp
        result = next_node.predict(data)                # Đệ quy để duyệt tiếp trên nhánh đã chọn
        return result


class MyDecisionTreeClassifier():
    '''Idea for this tree

    X_input
    -------
    example raw data:
    >>> id      outlook     temperature     humidity        wind        play
    >>> 1       sunny       hot             high            weak        no
    >>> 2       sunny       hot             high            strong      no
    >>> 3       overcast    hot             high            weak        yes
    >>> 4       rainy       mild            high            weak        yes
    >>> 5       rainy       cool            normal          weak        yes
    >>> 6       rainy       cool            normal          strong      no
    >>> 7       overcast    cool            normal          strong      yes
    >>> 8       sunny       mild            high            weak        no
    >>> 9       sunny       hot             normal          weak        yes
    >>> 10      rainy       mild            normal          weak        yes
    >>> 11      sunny       mild            normal          strong      yes
    >>> 12      overcast    mild            high            strong      yes
    >>> 13      overcast    hot             normal          weak        yes
    >>> 14      rainy       mild             high           strong      no

    is converted into:

    >>> [{'outlook': 'sunny', 'temp': 'hot', 'humidity': 'high', 'wind': 'weak'},           ##no
    >>> {'outlook': 'sunny', 'temp': 'hot', 'humidity': 'high', 'wind': 'strong'},          ##no
    >>> {'outlook': 'overcast', 'temp': 'hot', 'humidity': 'high', 'wind': 'weak'},         ##yes
    >>> {'outlook': 'rainy', 'temp': 'mild', 'humidity': 'high', 'wind': 'weak'},           ##yes
    >>> {'outlook': 'rainy', 'temp': 'cool', 'humidity': 'normal', 'wind': 'weak'},         ##yes
    >>> {'outlook': 'rainy', 'temp': 'cool', 'humidity': 'normal', 'wind': 'strong'},       ##no
    >>> {'outlook': 'overcast', 'temp': 'cool', 'humidity': 'normal', 'wind': 'strong'},    ##yes
    >>> {'outlook': 'sunny', 'temp': 'mild', 'humidity': 'high', 'wind': 'weak'},           ##no
    >>> {'outlook': 'sunny', 'temp': 'cool', 'humidity': 'normal', 'wind': 'weak'},         ##yes
    >>> {'outlook': 'rainy', 'temp': 'mild', 'humidity': 'normal', 'wind': 'weak'},         ##yes
    >>> {'outlook': 'sunny', 'temp': 'mild', 'humidity': 'normal', 'wind': 'strong'},       ##yes
    >>> {'outlook': 'overcast', 'temp': 'mild', 'humidity': 'high', 'wind': 'strong'},      ##yes
    >>> {'outlook': 'overcast', 'temp': 'hot', 'humidity': 'normal', 'wind': 'weak'},       ##yes
    >>> {'outlook': 'rainy', 'temp': 'mild', 'humidity': 'high', 'wind': 'strong'}]         ##no

    [outlook, temperature, humidity, wind]
    --------------------------------------
    is attributes, their values is props

    play
    ----
    is target or yClass

    Example tree
    ------------
    >>> Node('outlook', 
    >>>         {
    >>>         'sunny': Node('humidity', 
    >>>                             {'high': Node('no', {}), 
    >>>                             'normal': Node('yes', {})
    >>>                             }
    >>>                         ), 
    >>>         'overcast': Node('yes', {}), 
    >>>         'rainy': Node('wind', 
    >>>                             {'weak': Node('yes', {}), 
    >>>                             'strong': Node('no', {})
    >>>                             }
    >>>                         )
    >>>         }
    >>>     )
    '''

    # Hàm khởi tạo
    def __init__(self, attributes: list, x_train, y_train, criterion='entropy', max_depth=10) -> None:

        ## Kiểm tra tính hợp lệ của dữ liệu khởi tạo
        if len(attributes) != len(x_train[0]):
            raise ValueError(
                f'Parameters is not valid! size of attributes ({len(attributes)}) is different from number of data input ({len(x_train[0])})')
        if criterion not in ['entropy', 'gini']:
            raise ValueError('Invalid value for paramater criterion!')
        
        ## Tạo các thuộc tính của lớp
        self.attributes = attributes
        self.criterion = criterion
        self.yTrain = y_train
        self.max_depth = max_depth

        ## Xử lí bộ dữ liệu đầu vào
        self.xTrain = []
        if not str(type(x_train[0])) == "<class 'dict'>":
            for i in x_train:
                record = {}
                for index, val in enumerate(i):
                    record[str(self.attributes[index])] = str(val)
                self.xTrain.append(record)
        else:
            self.xTrain = x_train

    # Hàm tính giá trị Infomation gain hoặc Gini index
    def point(self, data, yClass, attribute_name):

        props = set(map(lambda x: x[attribute_name], data))     # Tạo danh sách tập các giá trị của attribute_name
        Point = 0 if self.criterion == 'entropy' else 1         # Khởi tạo điểm ban đầu

        ## Tính điểm của từng giá trị rồi cộng vào Point
        for i in props:

            ## Lấy ra tập nhãn được phân lớp (y) tương ứng với từng điểm dữ liệu của prop
            prop_targets = []
            for index, val in enumerate(data):
                if val[attribute_name] == i:
                    prop_targets.append(yClass[index])

            prop_classes = set(prop_targets)            # Tạo tập các lớp của prop
            total_prop_target = len(prop_targets)       # Lấy số lượng 

            prop_point = 0                              # Khởi tạo điểm entropy hoặc gini
            # Tính entropy hoặc gini theo xác xuất của mỗi lớp
            for prop_class in prop_classes:
                count_class = prop_targets.count(prop_class)
                prop_point += -(count_class/total_prop_target)*log(count_class/total_prop_target) if self.criterion == 'entropy' else (count_class/total_prop_target)**2
            Point += (total_prop_target/len(data))*prop_point if self.criterion == 'entropy' else - (total_prop_target/len(data))*prop_point
        
        return Point        # Trả về H(x,S)

    # Hàm xây dựng cây quyết định
    def buildTree(self, xData, yClass, attributes: list, depth):
        target_list = list(set(yClass))     # Lấy danh sách các nhóm được phân loại

        ## Nếu danh sách nhóm chỉ có 1 phần tử
        if len(target_list) == 1:
            return Node(target_list[0])     # Trả về nút lá 
        
        ## Nếu không có thuộc tính nào để xét
        if len(attributes) == 0 or depth >= self.max_depth:
            
            ## Đếm số lượng từng lớp (target) lưu vào count
            count = []
            for target in target_list:
                count.append(yClass.count(target))

            # Lấy ra vị trí tương ứng của lớp chiếm đa số
            target_max_index = count.index(max(count))

            return Node(str(target_list[target_max_index]))     # Trả về nút lá có tên là lớp chiếm da số

        ## Tính điểm entropy hoặc gini của từng thuộc tính rồi lưu vào danh sách Points
        Points = []
        for attribute_name in attributes:
            Points.append(self.point(xData, yClass, attribute_name))

        # Lấy ra tên thuộc tính có điểm nhỏ nhất
        min_point_attribute_name = attributes[Points.index(min(Points))]

        # Lấy ra danh sách các giá trị của thuộc tính nhỏ nhất trong data
        list_att_props = set(map(lambda x: x[min_point_attribute_name], xData))

        att_branches = {}       # Khởi tạo cấu trúc chứa các nhánh của node hiện tại

        ## Tạo danh sách các thuộc tính còn lại (loại bỏ thuộc tính có điểm nhỏ nhất được tính bên trên)
        attributesReduce = list(attributes)         
        attributesReduce.remove(min_point_attribute_name)

        ## Đệ quy cho từng nhánh để tìm ra node tương ứng của mỗi nhánh
        for att_prop in list_att_props:

            ## Tìm xData, yData (target) tương ứng của mỗi nhánh
            prop_data = []
            att_prop_targets = []
            for index, val in enumerate(xData):
                if val[min_point_attribute_name] == att_prop:
                    prop_data.append(val)
                    att_prop_targets.append(yClass[index])

            # Thêm nhánh vào danh sách
            att_branches[str(att_prop)] = self.buildTree(prop_data, att_prop_targets, attributesReduce, depth+1)
        
        # Kết thúc, trả về nút hiện tại
        return Node(min_point_attribute_name, att_branches)

    def fit(self):
        self.tree = self.buildTree(self.xTrain, self.yTrain, self.attributes, 1)

    def predict(self, data: list):

        ## Xử lí bộ dữ liệu đầu vào
        if not str(type(data[0])) == "<class 'dict'>":
            newData = []
            for i in data:
                record = {}
                for index, val in enumerate(i):
                    record[(str(self.attributes[index]))] = val
                newData.append(record)
        else:
            newData = data

        # Xử lí các dữ liệu gây lỗi
        predict_result = []
        for i in newData:
            try:
                predict_result.extend(self.tree.predict(i))
            except:
                predict_result.extend(['err'])

        # Trả về kết quả dự đoán
        return predict_result


########
try:
    data = pandas.read_csv('BThetmon/gianlanbaohiem.csv')
except:
    data = pandas.read_csv('gianlanbaohiem.csv')

## Loại bỏ một số cột gây nhiễu và lỗi
data = data.drop(labels='policy_number', axis=1)
data = data.drop(labels='policy_bind_date', axis=1)
data = data.drop(labels='incident_location', axis=1)
data = data.drop(labels='incident_date', axis=1)
data = data.drop(labels='insured_hobbies', axis=1)
data = data.drop(labels='months_as_customer', axis=1)

# Lấy tên các thuộc tính (bỏ cột cuối là nhãn)
columns = (data.columns.values)
atts = columns[:len(columns)-1]

# Lấy dữ liệu dạng numpy.array
data_np = data.values

# Xử lí rời rạc dữ liệu
ENCODE_LABELS = ['one', 'two', 'three', 'four',
                 'five', 'six', 'seven', 'eight', 'nine', 'ten']
enum_cols = enumerate(columns)
for (i, col) in enum_cols:
    if (str(type(data_np[0, i])) != "<class 'str'>"):
        min_val = min(data[col].values)
        max_val = max(data[col].values)
        unique_count = len(set(data[col]))
        # print(f"{col}: min = {min_val}, max = {max_val}, lenght = {unique_count}")
        if unique_count > 10:
            split_range = []
            for split_index in range(10):
                split_range.append(
                    min_val + ((max_val-min_val)/10*(split_index)))
            split_range.append(max_val+1)
            # print(split_range)
            for val in range(len(data[col].values)):
                for label_index in range(len(split_range)-1):
                    if (split_range[label_index] < float(data_np[val, i]) and float(data_np[val, i]) <= split_range[label_index+1]):
                        data_np[val, i] = ENCODE_LABELS[label_index]
                        break


train, test = train_test_split(data_np, test_size=0.3, shuffle=True)
x, y = train[:, :-1], train[:, -1]
xt, yt = test[:, :-1], test[:, -1]

mytree = MyDecisionTreeClassifier(attributes=atts, x_train=x, y_train=y, max_depth=5)
mytree.fit()

pred = mytree.predict(xt)
# print(pred)
print('err:', pred.count('err'))
# print(pred)

count = 0
for i in range(len(pred)):
    if pred[i] != 'err':
        if pred[i] == yt[i]:
            count += 1
print(f"Score: {count}/{(len(pred)-pred.count('err'))} = {count/(len(pred)-pred.count('err'))}")
print('max depth =', mytree.max_depth)