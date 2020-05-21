# 正弦波の時系列予測


## 学習データ: 学習したい時系列データ全体
## 系列データ: 学習データをモデルに入力するフォーマットに変換したもの。


## 振幅 1.0
## ノイズ-0.1~0.1




import numpy as np
# import pandas  as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import SimpleRNN
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


np.random.seed(0)       # 乱数の固定



# sin波の生成

## 振幅 1.0
## ノイズ-0.1~0.1

def make_sin(x, noise = 0, T = 100):  #(x:x軸の大きさ, amp:ノイズの規模を定めるもの, T: 周期)
  # # xはnp.arrayで渡すので、xのサイズと同じだけyが生成される。
  return np.sin(2*np.pi*x / T) + noise    # SIN(2πx / T)

def noise(amp):
  return amp * np.random.randint(-10, 10, len(x)) # -10~10の間の整数を xと同じ長さだけ生成。

def show_sin_wave(x,y):   # xとyはそれぞれ配列。サイズを揃える。
  plt.plot(x,y)
  plt.show()




x = np.arange(200)         # [0,1,2,3,・・・199]
noise = noise(0.01)        # [-2,-4,-8,8,9]     (ランダムな要素)
y = make_sin(x, noise)     # sin波の作成.



# show_sin_wave(x,y)





affect_length = 64      # 過去のデータを考慮する数。出力へ影響を与えることが出来る範囲を指定。
# ↑ 64の場合は、factors = [[0,1,・・,63],[1,2,・・,64]・・] → answers = [64,65] のように、 0~63の64要素で、値64を学習させる。

# y[i-25:i]をモデルに入力し、y[i]を学習させる。
def make_dataset(y, affect_length):
  factors = np.array([])
  answers = np.array([])
  for i in range(len(y)-affect_length):
    factors = np.append(factors,y[i:i+affect_length])       # factors = [[0,1,・・,24],[1,2,・・,25],[2,3,・・,26]]
    answers = np.append(answers,y[i+affect_length])         # answers = [25,26,27,・・・・]
  return(factors, answers)


(factors, answers) = make_dataset(y, affect_length)   # factors,answersはそれぞれ numpy.array ではなく list形式であることに注意。

# print(np.array(factors).shape):    (175,25)

factors = factors.reshape(-1,affect_length,1)           # -1は他の次元の指定より適切なサイズを決定してくれる。
# factorsの変換: [[0,1,2,・・・,24],[1,2,・・,25],・・] → [ [ [0],[1],[2],・・[24] ] , [ [1],[2],[3],・・,[25] ] ]
# print(factors.shape):  (175,25,1)



# モデルの定義

## 入力数: 1
## 隠れ層: x
## 隠れ層ユニット数: 200
## 出力数: 1

## 活性化関数: linear
## 誤差関数(損失関数):  平均二乗誤差
## 学習方法:   勾配降下法
## 学習率(lr): 0.001

n_in = 1
n_out = 1
num_neurons = 1
n_hidden = 200


# ニューラルネットワークを重ねてモデルを構成。
model = Sequential()


# 入力層から再帰層までの定義(SimpleRNNNクラス)

## batch_input_shape: (バッチ数,学習データのステップ数、説明変数の数)を多プルで指定。
## return_sequences: Falseの場合は、最後の時刻のみの出力を得る。
model.add(SimpleRNN(n_hidden, batch_input_shape=(None, affect_length, num_neurons), return_sequences=False))









# 再帰層から出力層までの定義(Dense,Activation)
model.add(Dense(num_neurons)) # Dense == 全結合モデル。 num_neurons: ユニット数
model.add(Activation('linear'))   # 活性化関数を指定。(linear関数)

optimizer = Adam(lr = 0.001)  # 学習率: 0.001。  #Adam: 最適化手法の一つ。デファクトスタンダートとして広く使われる。




# モデルの構築(コンパイル)


## mean_squared_error: 平均二乗誤差。
model.compile(loss="mean_squared_error", optimizer=optimizer)



# 訓練中にfitメソッドからコールバックして使用するための定義。
# → 学習の進み具合に応じて、エポックを実行するか打ち切るかを判断する。


## monitor: 監視する値
## mode:  訓練を終了するタイミングを定義。{auto,min,max}を選択.
###        ・min:  監視する値の減少が停止した際に訓練終了。
###        ・max:  監視する値の増加が停止した際に訓練終了。
###        ・auto:  minかmaxか、自動的に推測。
## patience:  指定したエポック数の間に改善がないと訓練を停止。

early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=20)








# 学習させる


## バッチサイズ: 300
## エポック数(学習する回数):     100


## .fit: 学習を実行。
## 学習データ: factors
## 正解データ(教師データ): answers
## batch_size: 学習データを小分けにする。 「過学習」を防ぐ。
## validation_split:  指定した割合を指定。(ex, 0.1 → 最後の10%が検証のために利用される)0~1の間の少数で指定。
## callbacks=[]:  訓練中に適応される関数たちを登録。
model.fit(factors, answers, batch_size=300, epochs=100, validation_split=0.1, callbacks=[early_stopping])



# 本番データを入力して予測させる。
pred = model.predict(factors)

# print(pred)     # 学習結果を見る。




# グラフ表示

# plt.figure(figsize=(20, 4)) # figsize:  図のサイズを指定。
# plt.subplot(1, 3, 1)  # 図中を (1 × 3)の大きさで分割。1番目の枠に表示。
# plt.plot(x, y, color='blue')  # 図を表示。
# plt.xlabel('x')               # 横軸名
# plt.ylabel('raw_data')        # 縦軸名

# plt.subplot(1, 3, 2)
# plt.xlim(-10, 210)            # x軸の表示範囲。
# plt.plot(x[affect_length:], pred, color='red')
# plt.xlabel('x')
# plt.ylabel('pred')

# plt.subplot(1, 3, 3)
# plt.plot(x, y, color='blue', label='raw_data')
# plt.plot(x[affect_length:], pred, color='red', label='pred')
# plt.xlabel('x')
# plt.legend(loc='lower left')  # 図のラベルの位置を指定。

# plt.tight_layout()            #グラフの重なりを解消。
# plt.show()







# 今まで学習させてきた最後の要素からスタート。

start = factors[-1].reshape(1, affect_length)[0]                   # [175,176,・・・・,200]  (25個の要素を持つnp.array)



# print(model.predict(start[-affect_length:].reshape(1,affect_length,1)))

for _ in range(800):
  predicted = model.predict(start[-affect_length:].reshape(1, affect_length, 1))      # .predictに対しては、shape(1,25,1)などの3次元配列を渡さないといけないかも。？
  # → predicted:  [[0.332343・・・]]  (1行1列)
  # 今度は予測した値を自分自身に追加して、繰り返すことで未来のデータを生成。
  start = np.append(start, predicted)                         # np.append(第一,第二) 第一の配列に第二を追加。破壊的ではない。


pred_y = np.append(y, start[affect_length:])

plt.xlim(-10, 410)
plt.title("affect_length={}".format(affect_length))
x_ = np.arange(200, 1000)
plt.plot(x, y, color='blue', label='raw_data')
plt.plot(x_, start[affect_length:], color='red', label='predicted')
plt.legend(loc='upper right', ncol=2)               # ncol: 凡例の列数を指定。
plt.ylim(-1.5, 1.5)

# plt.show()



# 系列データの長さを変えてみる。


# affect_length をいろいろ試してみる。


def rnn_test(affect_length, width, height, n_hidden=200):
  num_neurons = 1
  plt.figure(figsize=(20, 20))
  for i, al in enumerate(affect_length):
    (factors, answers) = make_dataset(y, al)
    factors = np.array(factors).reshape(-1, al, 1)

    model = Sequential()
    model.add(SimpleRNN(n_hidden, batch_input_shape=(None, al, num_neurons), return_sequences=False))
    model.add(Dense(num_neurons))
    model.add(Activation('linear'))
    optimizer = Adam(lr = 0.001)
    model.compile(loss="mean_squared_error", optimizer=optimizer)
    early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=20)
    model.fit(factors, answers, batch_size=300, epochs=100, validation_split=0.1, callbacks=[early_stopping])

    pred = model.predict(factors)

    start = factors[-1].reshape(1, al)[0]
    for _ in range(800):
      predicted = model.predict(start[-al:].reshape(1, al, 1))
      start = np.append(start, predicted)

    pred_y = np.append(y, start[al:])



    # 学習データの再現結果。

    # plt.subplot(width, height, 2*i+1)
    # plt.title("affect_length={}".format(al))                 # 図のタイトルを設定。 l=1, l=2,のように変化。
    # plt.xlim(-10, 210)
    # plt.plot(x[al:], pred, color='red')
    # plt.xlabel('x')
    # plt.ylabel('pred')



    # 未来予測の結果

    plt.subplot(width, height, i+1)
    plt.title("affect_length={}".format(al))
    plt.xlim(-10, 610)
    x_ = np.arange(200, 1000)
    plt.plot(x, y, color='blue', label='raw_data')
    plt.plot(x_, start[al:], color='red', label='predicted')
    plt.legend(loc='upper right', ncol=2)
    plt.ylim(-1.5, 1.5)

  plt.show()

Affect_Length=[1, 2, 4, 8, 16, 32, 64, 128]
rnn_test(Affect_Length, width=2, height=4)
