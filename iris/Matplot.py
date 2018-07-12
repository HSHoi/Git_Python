import matplotlib.pyplot as plt

# matplotlib의 pyplot 모듈을 사용해야 한다. 다음 간단하게 제곱의 수인 1, 4, 9, 16,
# 25, 36 을 입력값으로 하는 그래프를 그려보도록 한다

input_value = [1, 4, 9, 16, 25, 36]  # 그래프를 그리기 위한 데이터
plt.plot(input_value)  # plt.plot() 함수를 통한 입력값 전달
plt.show()  # 그래프를 화면에 호출

# x 좌표
x_Value = [1, 2, 3, 4, 5, 6]

# y 좌표
y_Value = [1, 4, 9, 16, 25, 36]

plt.plot(x_Value, y_Value)  # plot() 함수에 x, y 좌표 값 전달
plt.show()

plt.plot(x_Value, y_Value, linewidth=5)  # 그래프 라인 변경
plt.title("Get Square", fontsize=20)  # 그래프 타이틀 설정
plt.xlabel("x_values", fontsize=15)  # 그래프 x 축 라벨 설정
plt.ylabel("y_values", fontsize=15)  # 그래프 y 축 라벨 설정
plt.tick_params(axis='both', labelsize=13)  # 그래프 x, y 축 눈금 레이블 표시
plt.show()

plt.scatter(x_Value, y_Value, s=20)  # x와 y를 입력 값으로 점 그리기
plt.show()

# imshow를 이용하여 이미지를 그려보기
from matplotlib.image import imread

img = imread('test.jpg') # imread 함수를 이용하여 img 읽기
plt.imshow(img) # imshow 함수를 이용하여 img 불러오기
plt.show()


