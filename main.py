import pandas
import numpy
from random import randint
import dearpygui.dearpygui as dpg
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score #, recall_score, precision_score
# from sklearn.svm import SVC

global mo_hinh_tot_nhat

# Đọc file dữ liệu
try:
    data = pandas.read_csv('BThetmon/gianlanbaohiem.csv')
except:
    data = pandas.read_csv('gianlanbaohiem.csv')

# Lấy ra tên tất cả các thuộc tính
all_label = data.columns.values
all_label = all_label.tolist()

# Chọn các thuộc tính cần dùng
all_label = all_label[:25]

# Encode dữ liệu
data = data.apply(LabelEncoder().fit_transform)

# Tách dữ liệu thành tập train và test
train, test = train_test_split(data, test_size=0.3, shuffle=True)
xTRAIN, yTRAIN = numpy.array(train.iloc[:,:25]), numpy.array(train.iloc[:,len(data.iloc[0])-1])
XTEST, YTEST = test.iloc[:,:25], test.iloc[:,len(data.iloc[0])-1]

# Khai báo sử dụng k-fold
kfold = KFold(10, shuffle=True)

# Khởi tạo điểm đánh giá
max_score = 0

for KFtrain, validate in kfold.split(xTRAIN, yTRAIN):
    # Lấy ra dữ liệu được tách từ k-fold
    xTrain, yTrain = xTRAIN[KFtrain], yTRAIN[KFtrain]
    xValidate, yValidate = xTRAIN[KFtrain], yTRAIN[KFtrain]

    # Khởi tạo mô hình
    ID3_kfold = DecisionTreeClassifier(criterion='entropy', max_depth=20, splitter='best')
    CART_kfold = DecisionTreeClassifier(criterion='gini', max_depth=20, splitter='best')

    # Huấn luyện mô hình
    ID3_kfold.fit(xTrain, yTrain)
    CART_kfold.fit(xTrain, yTrain)

    # Dự đoán trên tập validate
    y_pred_ID3 = ID3_kfold.predict(xValidate)
    y_cart_ID3 = CART_kfold.predict(xValidate)

    # Tính điểm
    id3_accuracy_score = accuracy_score(yValidate, y_pred_ID3)
    cart_score = accuracy_score(yValidate, y_cart_ID3)

    # So sánh điểm tốt nhất rồi chọn ra mô hình tốt nhất
    if(id3_accuracy_score > max_score):
        max_score = id3_accuracy_score
        mo_hinh_tot_nhat = ID3_kfold
    if(cart_score > max_score):
        max_score = cart_score
        mo_hinh_tot_nhat = CART_kfold
        # Nếu có thêm mô hình khác thì
        # so sánh thêm

# Dự đoán và tính điểm của mô hình tốt nhất trên tập test
predict_test = mo_hinh_tot_nhat.predict(XTEST)
score = int((accuracy_score(predict_test, YTEST))*10000)/10000


### Bắt đầu xây dựng form
dpg.create_context()

# Hàm khi nhấn nút dự đoán
def du_doan():

    ## Tạo danh sách lưu dữ liệu dự đoán
    data_du_doan = []

    # Duyệt tất cả các input
    for i in(all_label):
        data_du_doan.append(dpg.get_value(i)) # Lấy giá trị của input và thêm vào danh sách
    ket_qua_du_doan = mo_hinh_tot_nhat.predict([data_du_doan]) # Thực hiện dự đoán

    # Chuyển đổi kết quả thành chữ
    ket_qua_du_doan = 'yes' if ket_qua_du_doan[0] == 1 else 'no'

    # Hiển thị kết quả
    dpg.set_value('kq', 'Ket qua du doan: '+ket_qua_du_doan)

# Hàm lấy ngẫu nhiên dữ liệu mẫu trong tập test
def lay_du_lieu():
    do_dai_tap_test = len(XTEST)                        # Lấy số lượng bộ dữ liệu của tập test
    phan_tu_ngau_nhien =  randint(0, do_dai_tap_test-1) # Lấy ngẫu nhiên vị trí của bộ dữ liệu trong tập test

    data_input = numpy.array(XTEST.iloc[phan_tu_ngau_nhien])    # Lấy dữ liệu từ vị trí đã chọn

    # Cập nhật giá trị cho các ô in =put
    for i, val in enumerate(all_label):
        dpg.set_value(val, int(data_input[i]) )

# Hiển thị cửa sổ chính
with dpg.window(tag='cua_so_chinh'):
    dpg.add_button(label='Du doan', pos=(810, 90), callback=du_doan) # Thêm button dự đoán
    dpg.draw_text(text='Accuracy score = ' + str(score), pos=(800, 20), size=30)    # Thêm đoạn chữ hiển trị thang điểm đánh giá mô hình
    dpg.add_text('Ket qua du doan:  ', pos=(810, 180), tag='kq')        # Thêm đoạn hiển thị kết quả dự đoán
    dpg.add_button(label='Lay du lieu mau', callback=lay_du_lieu)       # Thêm nút chọn ngẫu nhiên dữ liệu mẫu

    # Thêm các ô input dữ liệu dự đoán (giá trị mặc định ban đầu là bộ dữ liệu đầu tiên của tập test)
    data_input = numpy.array(XTEST.iloc[0])
    for index, val in enumerate(all_label):
        dpg.add_input_int(label=val, width=200, tag=val, default_value=int(data_input[index]))

# Các thủ tục của dearpygui
dpg.set_global_font_scale(2)
dpg.create_viewport(title='Fraud detecting', width=1280, height=720, x_pos=360, y_pos=120)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("cua_so_chinh", True)
dpg.start_dearpygui()
dpg.destroy_context()
