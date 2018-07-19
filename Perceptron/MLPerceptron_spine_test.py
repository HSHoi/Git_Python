import numpy as np
import pandas as pd
from subprocess import check_output
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

df = pd.read_csv('Dataset_spine.csv')
df = df.drop(['Unnamed: 13'], axis=1)

print(df.head())  # 데이터 기준으로 위에서 5개만 보여줌
#         Col1       Col2       Col3    ...          Col11    Col12  Class_att
# 0  63.027818  22.552586  39.609117    ...     -28.658501  43.5123   Abnormal
# 1  39.056951  10.060991  25.015378    ...     -25.530607  16.1102   Abnormal
# 2  68.832021  22.218482  50.092194    ...     -29.031888  19.2221   Abnormal
# 3  69.297008  24.652878  44.311238    ...     -30.470246  18.8329   Abnormal
# 4  49.712859   9.652075  28.317406    ...     -16.378376  24.9171   Abnormal
# [5 rows x 13 columns]

print(df.describe())
#              Col1        Col2     ...           Col11       Col12
# count  310.000000  310.000000     ...      310.000000  310.000000
# mean    60.496653   17.542822     ...      -14.053139   25.645981
# std     17.236520   10.008330     ...       12.225582   10.450558
# min     26.147921   -6.554948     ...      -35.287375    7.007900
# 25%     46.430294   10.667069     ...      -24.289522   17.189075
# 50%     58.691038   16.357689     ...      -14.622856   24.931950
# 75%     72.877696   22.120395     ...       -3.497094   33.979600
# max    129.834041   49.431864     ...        6.972071   44.341200
# [8 rows x 12 columns]

df = df.drop(['Col7', 'Col8', 'Col9', 'Col10', 'Col11', 'Col12'],
             axis=1)  # 데이터 제거
print(df.head())
#         Col1       Col2       Col3    ...            Col5       Col6  Class_att
# 0  63.027818  22.552586  39.609117    ...       98.672917  -0.254400   Abnormal
# 1  39.056951  10.060991  25.015378    ...      114.405425   4.564259   Abnormal
# 2  68.832021  22.218482  50.092194    ...      105.985135  -3.530317   Abnormal
# 3  69.297008  24.652878  44.311238    ...      101.868495  11.211523   Abnormal
# 4  49.712859   9.652075  28.317406    ...      108.168725   7.918501   Abnormal
# [5 rows x 7 columns]

# abnormal, normal
y = df['Class_att']  # Class_att 데이터 옮김
print(y.shape)
# (310,)

print(y)
# 1      Abnormal
# 2      Abnormal
# 3      Abnormal
# 4      Abnormal
# 5      Abnormal
# 6      Abnormal
# 7      Abnormal
# 8      Abnormal
# 9      Abnormal
#          ...
# 296      Normal
# 297      Normal
# 298      Normal
# 299      Normal
# 300      Normal
# 301      Normal
# 302      Normal
# 303      Normal
# 304      Normal
# 305      Normal
# 306      Normal
# 307      Normal
# 308      Normal
# 309      Normal

x = df.drop(['Class_att'], axis=1)
print(x)
# Name: Class_att, Length: 310, dtype: object
#           Col1       Col2    ...            Col5       Col6
# 0    63.027818  22.552586    ...       98.672917  -0.254400
# 1    39.056951  10.060991    ...      114.405425   4.564259
# 2    68.832021  22.218482    ...      105.985135  -3.530317
# 3    69.297008  24.652878    ...      101.868495  11.211523
# 4    49.712859   9.652075    ...      108.168725   7.918501
# 5    40.250200  13.921907    ...      130.327871   2.230652
# 6    53.432928  15.864336    ...      120.567523   5.988551
# ..         ...        ...    ...             ...        ...
# 304  45.075450  12.306951    ...      147.894637  -8.941709
# 305  47.903565  13.616688    ...      117.449062  -4.245395
# 306  53.936748  20.721496    ...      114.365845  -0.421010
# 307  61.446597  22.694968    ...      125.670725  -2.707880
# 308  45.252792   8.693157    ...      118.545842   0.214750
# 309  33.841641   5.073991    ...      123.945244  -0.199249

# print('y :', y)

# DataFramee, Series 타입으로 리턴됨.
# x_train(232x6), y_train(232,), x_test(78,6), y_test(78,)
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.25, random_state=27)  # test_size 데이터 크기
print("x_test :", x_test.shape)
# x_test : (78, 6)

print('y_train :', y_train[0])
# y_train : Abnormal

clf = MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=500, alpha=0.0001,
                    solver='sgd', verbose=10, random_state=21, tol=0.000000001)
# verbose 가 하는 역할 : 콘솔 아웃풋 지정
# alpha : L2 penalty (regularization term) parameter
# clf = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=500, alpha=0.0001,
#                    solver='sgd', verbose=10,  random_state=21,tol=0.000000001)
'''
print('x_train  shape', x_train.shape, 'type :', type(x_train))
print('y_train  shape', y_train.shape, 'type :', type(y_train))
print('x_test  shape', x_test.shape, 'type :', type(x_test))
print('y_test  shape', y_test.shape, 'type :', type(y_test))
'''
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)
print("cm :", cm)
# cm : [[48  5]
#       [ 7 18]]
sns.heatmap(cm, center=True)
plt.show()
