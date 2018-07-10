import numpy as np
# 손실함수
#   - 머신 러닝 모델에 의한 예측 값과 목표 값 사이의 차를 말하며 이를 표현한 함수

# 타켓(목표값) -> 내가 원하는 결과 데이터의 종류
# 숫자 0 1 2 3 4 5 6 7 8 9 을 one hot encoding 으로 변환 한것
# one hot encoding 계산 속도를 높이기 위한 것
t = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  # [0, 0, 0, 0, 0, 0, 0, 1, 0, 0] == 7
# [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] == 0
# [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] == 1

# 예측값
y1 = [0.1, 0, 0, 0.1, 0.05, 0.05, 0, 0.7, 0, 0]  # 7일 확률이 70%
y2 = [0.0, 0.1, 0.7, 0.05, 0.05, 0.1, 0, 0, 0, 0]  # 2일 확률이 70%


def mean_seq_error(t, y): # 평균 제곱 오차(Mean Squared Error)
    return 0.5 * np.sum((t - y) ** 2)


def cross_entropy(t, y): # 교차 엔트로피 오차(Cross Entropy Error)   - 평균 제곱 오차보다 더 예민함(log 함수를 쓰기 때문)
    tmp = 1e-7  # 로그 값에 의한 무한대 발생 방지 값
    return -np.sum(t * np.log(y + tmp)) # log(0)은 무한대가 되기 때문에 방지를 위해 아주작은 tmp 값을 넣어줌


print(mean_seq_error(np.array(t), np.array(y1)))
print(mean_seq_error(np.array(t), np.array(y2)))

print(cross_entropy(np.array(t), np.array(y1)))
print(cross_entropy(np.array(t), np.array(y2)))
