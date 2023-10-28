import pandas
import numpy
import dearpygui.dearpygui as dpg
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score

global mo_hinh_tot_nhat

data = pandas.read_csv('drug200.csv')
all_label = data.columns.values
all_label = all_label.tolist()

data = data.apply(LabelEncoder().fit_transform)

train, test = train_test_split(data, test_size=0.3, shuffle=True)
xTRAIN, yTRAIN = numpy.array(train.iloc[:,:5]), numpy.array(train.iloc[:,5])
XTEST, YTEST = test.iloc[:,:5], test.iloc[:,5]

kfold = KFold(10, shuffle=True)

max_score = 0

for KFtrain, validate in kfold.split(xTRAIN, yTRAIN):
    xTrain, yTrain = xTRAIN[KFtrain], yTRAIN[KFtrain]
    xValidate, yValidate = xTRAIN[KFtrain], yTRAIN[KFtrain]

    ID3_kfold = DecisionTreeClassifier(criterion='entropy', max_depth=3, splitter='random')
    CART_kfold = DecisionTreeClassifier(criterion='gini', max_depth=3, splitter='random')

    ID3_kfold.fit(xTrain, yTrain)
    CART_kfold.fit(xTrain, yTrain)

    y_pred_ID3 = ID3_kfold.predict(xValidate)
    y_cart_ID3 = CART_kfold.predict(xValidate)

    id3_score = accuracy_score(yValidate, y_pred_ID3)
    cart_score = accuracy_score(yValidate, y_cart_ID3)

    if(id3_score > max_score):
        max_score = id3_score
        mo_hinh_tot_nhat = ID3_kfold

    if(cart_score > max_score):
        max_score = cart_score
        mo_hinh_tot_nhat = CART_kfold

predict_test = mo_hinh_tot_nhat.predict(XTEST)
# print(predict_test)
print(accuracy_score(predict_test, YTEST))

all_label.pop()
print(all_label)

dpg.create_context()

with dpg.window(tag='cua_so_chinh'):
    dpg.add_button(label='click')
    for i in all_label:
        dpg.add_input_int(label=i, width=200, tag=i, default_value=1)

    data_input = []
    for i in all_label:
        data_input.append(dpg.get_value(i))
    print(data_input)

dpg.set_global_font_scale(2)
dpg.create_viewport(title='Credit Card Dataset for Clustering', width=1280, height=720, x_pos=360, y_pos=120)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("cua_so_chinh", True)
dpg.start_dearpygui()
dpg.destroy_context()
