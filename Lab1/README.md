# Regression House Sale Price Prediction

# 作法說明

## 1.資料處理
為了方便整體資料畫圖與處理，將training data 與 valid data 合併起來，形成一 (15128, 23) 的資料
```python
data_train = pd.read_csv("ntut-ml-regression-2021/train-v3.csv")
data_valid = pd.read_csv("ntut-ml-regression-2021/valid-v3.csv")
data_test = pd.read_csv("ntut-ml-regression-2021/test-v3.csv")
print(data_train.shape)
print(data_valid.shape)
print(data_test.shape)

data_all = pd.concat([data_train, data_valid])
```

## 2. 資料視覺化
針對每個變數進行簡單的統計，並畫出其分布圖，變數與房價之間的相關係數圖，對資料有個初步的認識

統計每個變數的簡單統計量
```python
num_data=data_all.select_dtypes(['int64','float64'])
describe_num=data_all.describe().transpose()
print (describe_num)
```

針對每個變數畫出其分佈圖，觀察資料，可以看到 'price', 'sqft_living','sqft_above', 'sqft_living15'，其分佈都集中在左半邊。
```python
plt.figure(figsize=(25,25))
for i, name in enumerate(list(num_data.columns)):
    
    sns.distplot(num_data[name].dropna()).set_title(name)
    plt.subplot(5,5,i+1)

plt.show()
```
![變數分佈圖](https://github.com/Machine-Learning-NYCU/regression-house-sale-price-prediction-challenge-310706038/blob/main/images/distribution_origin.png)

畫出變數與房價之間的相關係數熱圖，可以看到id、condition、zipcode對於房價的相關係數很低。
雖然日期的部分也很低，但年份與月份是有其意義存在的，可能在哪個年份的景氣比較好，或者是哪個季節買氣較高，因此日期的部分選擇保留。
```python
num_train=data_train
num_corr=num_train.corr().drop('id')    
fig,ax=plt.subplots(figsize=(15,1))
sns.heatmap(num_corr.sort_values(by=['price'], ascending=False).head(1), cmap='Reds')
plt.title("Correlation Matrix", weight='bold', fontsize=18)
plt.xticks(weight='bold')
plt.yticks(weight='bold', color='dodgerblue', rotation=0)
```
![相關係數熱圖](https://github.com/Machine-Learning-NYCU/regression-house-sale-price-prediction-challenge-310706038/blob/main/images/correlation.png)

## 3. 資料處理
透過視覺化資料，對資料有了初步了解後，針對變數進行處理

### 用在訓練模型上
從圖中可以看到 'sqft_lot', 'sqft_basement', 'sqft_lot15' 這三個變數的分布大部分集中在0，因此將不是0的部分轉換成1，形成一二元變數
並且根據correlation熱圖將相關係數較低的變數，"id", "sale_day", "zipcode", "condition"，此四個變數丟棄

```python
def data_processing(data):
#     zero processing
    zero = ['sqft_lot', 'sqft_basement','sqft_lot15']
    for z in zero: 
        data[z] = (data[z] != 0).astype(int)

    #drop lat and long
    data.drop(["id", "sale_day", "zipcode", "condition"], inplace = True, axis = 1)
    
    return data
```
### 未用在訓練模型上
以下資料處理有用在初步的模型訓練實驗中，但後來觀察實驗MAE數據未有太大的進步，就未採用以下原來資料處理的部分。

1. 在資料視覺化中有發現以下變數，'sqft_living','sqft_above', 'sqft_living15'，的資料分佈有集中在左半邊的部分，因此對這些變數取log，使得其分配會趨於常態分佈。
2. zipcode 變數採用one-hot encoding進行編碼
3. 針對經緯度採用kmeans的方法進行分群
4. 將只要有關年份的變數，以2023年進行相減，使得此年份變數變得有意義

### 標準化
最後將處理完成的資料，再切分回去為training與valid data，並以standard scaler的方式，進行標準化，並且連同房價也進行標準化（最後會再轉換回原來的數值）。

```python
from sklearn.preprocessing import StandardScaler

X_train = data_all[:12967].drop(columns=['price'])
y_train = data_all[:12967]['price']
X_test = data_all[12968:].drop(columns=['price'])
y_test = data_all[12968:]['price']
print(X_train.shape)

num_vars = [c for c in data_all.columns if 'uint' not in str(data_all[c].dtype)]
num_vars.remove('price')

ss_X = StandardScaler()
ss_y = StandardScaler()

X_train[num_vars] = ss_X.fit_transform(X_train[num_vars])
X_test[num_vars] = ss_X.fit_transform(X_test[num_vars])

y_train = ss_y.fit_transform(y_train.values.reshape(-1,1))
y_test = ss_y.fit_transform(y_test.values.reshape(-1,1))
```

# 模型設定
此次預測房價的模型採用neural network中的linear regression，以下為模型架構，建構了3層的hidden layer，其中的神經元數量分別為12、6、3。
有嘗試試著使用更深層更寬的身經網路架構進行訓練，經過多次的實驗，發現此筆資料若用太複雜的架構，容易導致overfitting的情況產生。
```
Sequential(
  (0): Linear(in_features=18, out_features=12, bias=True)
  (1): ReLU()
  (2): Linear(in_features=12, out_features=6, bias=True)
  (3): ReLU()
  (4): Linear(in_features=6, out_features=3, bias=True)
  (5): ReLU()
  (6): Linear(in_features=3, out_features=1, bias=True)
)
```

loss function 使用 MAE 判斷模型的預測結果

optimizer使用Adam，初始learning rate設定為0.01，並在loss function 加入懲罰項，防止模型overfitting，weight_decay值設定為0.001

並且採用learning rate scheduler，促使learning rate會在訓練的過程中遞減，防止learning rate太大無法進到最佳解中。

```python
loss_fn = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=5, verbose=True
)
```

訓練次數為100次，batch size為64
```python
n_epochs = 100   # number of epochs to run
batch_size = 64  # size of each batch
```


# 訓練與驗證loss圖
因為有將房價變數進行標準化的轉換，因此y軸部分的MAE數值較小。

可以看到training與valid的loss在訓練到後期時，都有明顯的收斂，不過有一點overfitting的現象產生。

![訓練驗證loss圖](https://github.com/Machine-Learning-NYCU/regression-house-sale-price-prediction-challenge-310706038/blob/main/images/loss.png)


# 改進與心得討論
## 改進

1. 本實驗只使用一個neural network的模型，peformance的部分的MAE最好大約在72000(kaggle)，無法再提升，因此認為可以採用stacking的方式，採用多模型進行疊加，將各個模型的優點集合起來，可以更加改善此預測的準確度。

2. 應採用K-Fold的方式進行訓練，提高模型泛化的效果，防止模型會有overfitting的情況產生。

3. 本實驗原來有採用的資料處理方式應還是可以使用，像是取log或是年份變數的處理，其中zipcode可以使用label encoding的方式，避免變數產生過多。

## 心得討論
此次kaggle競賽讓我學習到許多資料處理的方式，如何觀察一份資料，並且針對特定情況可以進行相對應的特徵工程處理，可以提升模型訓練的效能。透過聆聽同學的分享，也可以讓我從中學習到許多，並且發現我本來以為對實驗沒有幫助的方法，但在其他人的實驗是有所幫助的，讓我更加瞭解到這些方法應用情境為何。最後，也讓我學習到了模型疊加的方式，可以透過這樣的技巧，使得連續數值得預測變得更加精準。
