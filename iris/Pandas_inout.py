import pandas as pd

# 로컬의 iris.csv 파일을 읽기 위해 read_csv 함수 호출
iris1 = pd.read_csv('iris.csv', names=['sl', 'sw', 'pl', 'pw', 'regression'])
print(iris1)
# sl sw pl pw regression
# 0 5.1 3.5 1.4 0.2 Iris-setosa
# 1 4.9 3.0 1.4 0.2 Iris-setosa
# 2 4.7 3.2 1.3 0.2 Iris-setosa
# :
# 149 5.9 3.0 5.1 1.8 Iris-virginica
# [150 rows x 5 columns]

# CSV 파일의 상위 5개 데이터를 확인하는 기능 제공
print(iris1.head())
#     sl   sw   pl   pw   regression
# 0  5.1  3.5  1.4  0.2  Iris-setosa
# 1  4.9  3.0  1.4  0.2  Iris-setosa
# 2  4.7  3.2  1.3  0.2  Iris-setosa
# 3  4.6  3.1  1.5  0.2  Iris-setosa
# 4  5.0  3.6  1.4  0.2  Iris-setosa

# CSV 파일의 아래의 데이터를 확인할 수 있는 기능 제공
print(iris1.tail(2))
#       sl   sw   pl   pw      regression
# 148  6.2  3.4  5.4  2.3  Iris-virginica
# 149  5.9  3.0  5.1  1.8  Iris-virginica

# txt 파일 불러오기
iris2 = pd.read_table('iris2.txt', sep='\s+', # sep : 구분자 '\s+' 스페이스가 여러개있어도 하나로보고 스페이스 기준으로 구분
                      names=['sl', 'sw', 'pl', ' pw', ' regression'])
print(iris2)
#     sl   sw   pl   pw   regression
# 0  5.1  3.5  1.4  0.2  Iris-setosa
# 1  4.9  3.0  1.4  0.2  Iris-setosa
# 2  4.7  3.2  1.3  0.2  Iris-setosa
# 3  4.6  3.1  1.5  0.2  Iris-setosa

# skiprows 를 통한 읽지 말고 건너뛸 행을 리스트로 설정함.
iris3 = pd.read_table('iris2.txt', sep='\s+', # sep 은 구분자 \s 스페이스
                      names=['sl', 'sw', 'pl', 'pw', 'regression'], skiprows=[0, 2])
print(iris3)
#     sl   sw   pl   pw   regression
# 0  4.6  3.1  1.5  0.2  Iris-setosa

