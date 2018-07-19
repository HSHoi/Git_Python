class Add:
    def foraward(self, x, y):
        return x + y

    def backward(self, dout):
        return dout, dout


class Mul:
    def foraward(self, x, y):
        self.x = x
        self.y = y
        return x * y

    def backward(self, dout):
        return dout * self.y, dout * self.x


def apple_graph():
    apple_price = 100
    apple_count = 2
    tax = 1.1

    layer_apple = Mul()
    layer_tax = Mul()

    # 순전파
    apple_total = layer_apple.foraward(apple_price, apple_count)
    print(layer_apple.x, layer_apple.y)

    total = layer_tax.foraward(apple_total, tax)
    print(layer_tax.x, layer_tax.y)
    print('forward : ', total)
    # forward :  220.00000000000003

    # 역전파 오류에 의한 영향
    d_total = 1.0

    # 세금 계산 층에서  갯수에 대한 영향, 세금에 대한 영향
    d_apple_total, d_tax = layer_tax.backward(d_total)

    #  사과 계산 계층에서의 사고 갯수에 대한 영향과 세금에 대한 영향
    d_apple_price, d_apple_count = layer_apple.backward(d_apple_total)

    print('backward price : {0} count : {1} tax : {2}'.format(d_apple_price, d_apple_count, d_tax))
    # backward price : 2.2 count : 110.00000000000001 tax : 200.0


def fruit_grapth():
    apple_price = 100
    apple_count = 2
    mango_price = 150
    mango_count = 3
    tax = 1.1

    layer_apple = Mul()
    layer_mango = Mul()
    layer_fruit = Add()
    layer_tax = Mul()

    # 순전파
    # 사과 계산 층에서의 갯수에 따른 계산
    apple_total = layer_apple.foraward(apple_price, apple_count)
    mango_total = layer_mango.foraward(mango_price, mango_count)

    # 사과와 망고의 계산 합 층
    fruit_total = layer_fruit.foraward(apple_total, mango_total)

    # 세금 계산 충
    total = layer_tax.foraward(fruit_total, tax)
    print('foward :', total)

    # 역전파
    # 1개 변화했을때의 영향
    d_total = 1.0
    d_fruit_total, d_tax = layer_tax.backward(d_total)

    d_apple_total, d_mango_total = layer_fruit.backward(d_fruit_total)
    d_apple_price, d_apple_count = layer_apple.backward(d_apple_total)
    d_mango_price, d_mango_count = layer_mango.backward(d_mango_total)

    print('      backward :', "d_fruit_total [", d_fruit_total, "] d_tax [", d_tax, "]")
    print('fruit backward :', "d_apple_total [", d_apple_total, "] d_mango_total [", d_mango_total, "]")
    print('apple backward :', "d_apple_price [", d_apple_price, "] d_apple_count [", d_apple_count, "] d_tax[", d_tax,
          "]")
    print('Mango backward :', "d_mango_price [", d_mango_price, "] d_mango_count [", d_mango_count, "]")


# 오류 역전파에 의해 주어진 목표 가격에 대한 과일 갯수를 계산해 보자
def back_propagation():
    apple_count = 2  # 사과의 개수, 고정값(학습데이터)
    mango_count = 3  # 망고의 개수, 고정값(학습데이터)
    tax = 1.1

    # 사과와 망고의 가격을 병경하여 총 가격이 720을 나오게 해야함
    target = 720  # 목표치...

    # weights
    apple_price = 100  # 사과의 가격, 웨이트값
    mango_price = 150  # 망고의 가격, 웨이트값

    layer_apple = Mul()
    layer_mango = Mul()
    layer_fruit = Add()
    layer_tax = Mul()

    for i in range(100):
        # 순전파
        apple_total = layer_apple.foraward(apple_price, apple_count)
        mango_total = layer_mango.foraward(mango_price, mango_count)
        fruit_total = layer_fruit.foraward(apple_total, mango_total)
        total = layer_tax.foraward(fruit_total, tax)
        print('foward :', total)

        # 역전파
        #        d_total = 1.0
        d_total = total - target  # 목표 값과의 차이
        if d_total == 0:  # 목표한 값과의 차이가 없다. 즉 목표값에 도달
            break

        print('d_total :', d_total)

        d_fruit_total, d_tax = layer_tax.backward(d_total)
        print('d_fruit_total : {0}, d_tax : {1}'.format(d_fruit_total, d_tax))

        d_apple_total, d_mango_total = layer_fruit.backward(d_fruit_total)
        d_apple_price, d_apple_count = layer_apple.backward(d_apple_total)
        d_mango_price, d_mango_count = layer_mango.backward(d_mango_total)

        # 0.1 은 학습률
        apple_price -= 0.1 * d_apple_price
        mango_price -= 0.1 * d_mango_price
        print('학습 횟수 : {} apple price : {:.2f}, mango_price : {:.2f}'.format(i, apple_price, mango_price))


# apple_graph()
# fruit_grapth()
back_propagation()
