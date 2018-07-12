# Pandas : 데이터 분석 작업의 편의성을 제공하는 다양한 기능을 가지고 있음.
# Series : 1차원 배열과 비슷한 형태이지만 각 데이터에 대한 인덱스(Index)정보가 붙어 있는 형태의 자료
from pandas import Series

house_price = Series([10, 20, 30, 40, 50], index=['강원', '인천', '전라', '제주', '서울'])
print(house_price)
# 강원    10
# 인천    20
# 전라    30
# 제주    40
# 서울    50
# dtype: int64

print(house_price['제주'])
# 40

print(house_price.index) # Series 클래스 index 속성
# Index(['강원', '인천', '전라', '제주', '서울'], dtype='object')

print(house_price.values) # Series 클래스 values 속성
# [10 20 30 40 50]

print(type(house_price))
# <class 'pandas.core.series.Series'>

print(house_price.size)
# 5

print(house_price - 5) # Series 클래스 사칙연산
# 강원     5
# 인천    15
# 전라    25
# 제주    35
# 서울    45
# dtype: int64

print(house_price[[0, 3]]) # Series 클래스 인덱싱
# 강원    10
# 제주    40
# dtype: int64

print(house_price[0:3]) # Series 클래스 슬라이싱
# 강원    10
# 인천    20
# 전라    30
# dtype: int64

