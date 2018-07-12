import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, minmax_scale

# Scikit-Learn 패키지에서 제공하는 전처리 기능은 4가지이며, preprocessing과
# feature_extraction 패키지로 제공

# Scaling을 통해 데이터의 Scale을 맞추면 Weight의 scale도 일관성 있도록 맞출 수 있음
stdscaler = StandardScaler()  # StandardScalar(표준화 방법)은 평균=0과 표준편차=1이 되도록 Scaling 하는 방법
minmax_scaler = MinMaxScaler()  # MinMax Scaling은 최댓값 = 1, 최솟값 = 0으로 하여, 그 사에 값들이 있도록 하는 방법
input_data = (np.arange(5, dtype=np.float) - 2).reshape(-1, 1)  # 1차원 배열을 2차원 배열로 변경
print(np.arange(5, dtype=np.float) - 2)
# [-2. -1.  0.  1.  2.]

minmax_scale_data = minmax_scaler.fit_transform(input_data)  # input_data 를 minmax scaling을 적용
print(minmax_scale_data)
# [[0.  ]
#  [0.25]
#  [0.5 ]
#  [0.75]
#  [1.  ]]

print(" 평균: ", minmax_scale_data.mean(axis=0))
# 평균:  [0.5]

print(" 표준편차: ", minmax_scale_data.std(axis=0))
# 표준편차:  [0.35355339]

df1 = pd.DataFrame(np.hstack([input_data, minmax_scale(input_data)]),
                   columns=[" input_data", "minmax_scale((input_data)"])
print(df1)
#    input_data  minmax_scale((input_data)
# 0         -2.0                       0.00
# 1         -1.0                       0.25
# 2          0.0                       0.50
# 3          1.0                       0.75
# 4          2.0                       1.00

from sklearn.preprocessing import OneHotEncoder  # one hot Encoder 모듈 선언

ohe = OneHotEncoder()  # Scikit-Learn의 OneHotEncoder 호출
X = np.array([[2], [1], [0]])
print(X)
# [[2]
#  [1]
#  [0]]

print(ohe.fit_transform(X).toarray())
# [[0. 0. 1.]
#  [0. 1. 0.]
#  [1. 0. 0.]]

# 입력 값이 몇 개의 분류로 구분되는지 확인
print(ohe.n_values_)  # n_values_ : 각 특징 당 분류할 수 있는 개수
# [3]

# one hot Encoding 시 벡터의 원소 값들이 어떻게 나뉘는지 표현.
print(ohe.feature_indices_)  # feature_indices_ : 입력 데이터가 벡터인 경우 각 원소를 나누기 위한 값
# [0 3]

# one hot Encoding에 사용되는 색인 값
print(ohe.active_features_)  # active_features_ : 실제로 분류를 위해 사용되는 색인 값
# [0 1 2]

from sklearn.preprocessing import LabelBinarizer

# 주어진 입력 데이터에서 문자열 라벨 정보에 대해 One hot Encoding을 하기 위해 사용되는 모듈
lb = LabelBinarizer()
X = ['A', 'B', 'C', 'D', 'A', 'B']

# fit(X[,y]) : X 값에 대해 One hoe Encoder를 맞춤
# - fit 이후 n_values_, feature_indices_, active_features_를 사용할 수 있음.
lb.fit(X)

print(lb.classes_)
# ['A' 'B' 'C' 'D']

# transform(X) : X를 One hot Encoding을 사용하여 변환함
print(lb.transform(X))
# [[1 0 0 0]
#  [0 1 0 0]
#  [0 0 1 0]
#  [0 0 0 1]
#  [1 0 0 0]
#  [0 1 0 0]]

from sklearn.preprocessing import Binarizer  # Binarizer 모듈 선언

# 주어진 입력 값에 대해 0과 1의 값으로 인코딩 시 미리 선정된 임계 값(Threshold)을 기준으로
# 인코딩 값을 결정하는 기법
binarizer = Binarizer()  # Binarizer 생성 default threshold = 0
X = np.array([[1., -1.], [-1., 0.], [0., 2.]])  # 2차원 입력 데이터 선언
print(X)
# [[ 1. -1.]
#  [-1.  0.]
#  [ 0.  2.]]

print(binarizer.transform(X))  # Binarizer 인코딩 변환 수행
# [[1. 0.]
#  [0. 0.]
#  [0. 1.]]

# 2차원 입력 데이터 선언
X = np.array([[1., -1.], [-1., 0.], [0., 2.]])
print(X)
# [[ 1. -1.]
#  [-1.  0.]
#  [ 0.  2.]]

binarizer1 = Binarizer(threshold=1.5)  # Binarizer 생성 (임계값 1.5)
print(binarizer1.transform(X))  # Binarizer 인코딩 변환 수행
# [[0. 0.]
#  [0. 0.]
#  [0. 1.]]
