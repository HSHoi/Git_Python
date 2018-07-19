

class Add :
    def foraward(self, x, y):
        return x + y

    def backward(self, dout):
        return dout, dout



class Mul :
    def foraward(self, x, y):
        self.x = x
        self.y = y
        return x * y

    def backward(self, dout):
        return dout*self.y, dout*self.x



def apple_graph() :
    apple_price = 100
    apple_count =2
    tax = 1.1

    layer_apple = Mul()
    layer_tax = Mul()

    # 순전파
    apple_total = layer_apple.foraward(apple_price, apple_count)
    total = layer_tax.foraward(apple_total, tax)
    print('foward :', total)

    # 역전파    오류에 의한 영향
    d_total = 1.0
    # 세금 계산 층에서  갯수에 대한 영향, 세금에 대한 영향
    d_apple_total, d_tax = layer_tax.backward(d_total)
    #  사과 계산 계층에서의 사고 갯수에 대한 영향과 세금에 대한 영향
    d_apple_price, d_apple_count = layer_apple.backward(d_apple_total)

    print('backward :', d_apple_price, d_apple_count, d_tax)


def fruit_grapth() :
    apple_price = 100
    apple_count =2
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

    print('      backward :', "d_fruit_total [",d_fruit_total,"] d_tax [", d_tax,"]")
    print('fruit backward :', "d_apple_total [",d_apple_total,"] d_mango_total [", d_mango_total,"]")
    print('apple backward :', "d_apple_price [",d_apple_price,"] d_apple_count [", d_apple_count,"] d_tax[", d_tax, "]")
    print('Mango backward :', "d_mango_price [", d_mango_price, "] d_mango_count [", d_mango_count,"]")

# 오류 역전파에 의해 주어진 목표 가격에 대한 과일 갯수를 계산해 보자
def back_propagation() :
    apple_count = 2
    mango_count = 3
    tax = 1.1


    target = 720 # 목표치...

    # weights       (가격)
    apple_price = 100
    mango_price = 150

    layer_apple = Mul()
    layer_mango = Mul()
    layer_fruit = Add()
    layer_tax = Mul()



    for i in range(100) :

        # 순전파
        apple_total = layer_apple.foraward(apple_price, apple_count)
        mango_total = layer_mango.foraward(mango_price, mango_count)
        fruit_total = layer_fruit.foraward(apple_total, mango_total)
        total = layer_tax.foraward(fruit_total, tax)
        print('foward :', total)

        # 역전파
#        d_total = 1.0
        d_total = total - target       #  목표 값과의 차이

        d_fruit_total, d_tax = layer_tax.backward(d_total)

        d_apple_total, d_mango_total = layer_fruit.backward(d_fruit_total)
        d_apple_price, d_apple_count = layer_apple.backward(d_apple_total)
        d_mango_price, d_mango_count = layer_mango.backward(d_mango_total)


        #
        apple_price -=  0.1 * d_apple_price
        mango_price -=  0.1 * d_mango_price

        print('{} : {:.2f}, {:.2f}'.format(i, apple_price, mango_price))

#apple_graph()
#fruit_grapth()
back_propagation()



