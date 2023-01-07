import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#문제: 키로 신발사이즈를 추론해보자 => 선형회귀모델 

height = 170
shoes = 260

#shoes = height * a + b

# Variable로 변수 선언하기
a = tf.Variable(0.1)
b = tf.Variable(0.2)

#return tf.square(실제값 - 예측값)    
def loss_func():
    predict_val = height * a + b
    return tf.square(260-predict_val)

#a, b 변수를 선언한 뒤 좋은 결과가 나올 때까지 학습시킬 것임
#a,b 경사하강법으로 구하기
#tf.keras.optimizers 경사하강법을 도와주는 메서드
#gradient를 알아서 스마트하게 바꿔줌
#learning_rate는 w를 얼마만큼 update할건 지 수치 옵션
opt = tf.keras.optimizers.Adam(learning_rate=0.1)

#경사하강법으로 업데이트할 weight Variable 목록
for i in range(300):
    opt.minimize(loss_func, var_list=[a,b])
    print(a.numpy(),b.numpy())
