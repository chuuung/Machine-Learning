# Simpsons Characters Recognition

# 作法說明

## 1.資料處理
因為testing data加入了翻轉變形，改變顏色跟雜訊干擾等等變化，而原本的training data沒有。若用正常的資料進行訓練，那最後tetsing的結果會不理想。因此需要對原始資料進行處理，進行資料擴充的方式，透過以下的transform，針對圖片加入翻轉變形，改變顏色跟雜訊干擾等等變化。
```python
transforms.Compose([
    transforms.Resize(size=(RESCALE_SIZE, RESCALE_SIZE)),
    transforms.ToTensor(),
    transforms.RandomApply([transforms.RandomHorizontalFlip()], p=0.1),
    transforms.RandomApply([transforms.RandomVerticalFlip()], p=0.1),
    transforms.RandomApply([transforms.RandomRotation(10)], p=0.1),

    transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomInvert(p=0.1),
    # transforms.RandomPosterize(bits=2, p=0.1),
    transforms.RandomApply([transforms.RandomSolarize(threshold=1.0)], p=0.05),
    transforms.RandomApply([transforms.RandomAdjustSharpness(sharpness_factor=2)], p=0.1),

    transforms.RandomApply([AddGaussianNoise(0., 0.05)], p=0.1),  # mean and std
    transforms.RandomApply([AddPoissonNoise(lam=0.1)], p=0.1),  # mean and std
    transforms.RandomApply([AddSpeckleNoise(noise_level=0.1)], p=0.1),
    transforms.RandomApply([AddSaltPepperNoise(salt_prob=0.05, pepper_prob=0.05)], p=0.1),

    transforms.RandomApply([transforms.RandomPerspective(distortion_scale=0.6, p=1.0)], p=0.1),
    transforms.RandomApply([transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))], p=0.1),
    transforms.RandomApply([transforms.ElasticTransform(alpha=250.0)], p=0.1),

    transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))], p=0.1),

    transforms.RandomApply([AddGaussianNoise(0., 0.001)], p=1.0),  # mean and std
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
])
```
印出轉換過後的圖片，可以看到圖片會變得較不正常，這使得模型在訓練時也可以提高模型的泛化性。
```python
fig, ax = plt.subplots(nrows=3, ncols=3,figsize=(8, 8), \
                        sharey=True, sharex=True)

for fig_x in ax.flatten():
    random_characters = int(np.random.uniform(0,1000))
    im_val, label = val_dataset[random_characters]
    img_label = " ".join(map(lambda x: x.capitalize(),\
                val_dataset.label_encoder.inverse_transform([label])[0].split('_')))
    imshow(im_val.data.cpu(), \
          title=img_label,plt_ax=fig_x)
```
![圖片轉換後](https://github.com/Machine-Learning-NYCU/the-simpsons-characters-recognition-challenge-iii-310706038/blob/master/Lab2/plot/transform.png)

將training與validation切分為8:2
```python
from sklearn.model_selection import train_test_split
train_files, val_files = train_test_split(train_val_files, test_size=0.20, stratify=train_val_labels)
```



## 2. Pretrained Model
### EfficientNet
本次影像分類所使用的模型為EfficientNet，將pretrained模型載入。
```python
model_name = 'efficientnet-b2'
model = EfficientNet.from_pretrained(model_name)
```

修改模型的結構，使得模型輸出符合本資料集的任務，將最後一層的全連結層的輸出改為50(simpson character共50個)
```python
N_CLASSES = len(np.unique(train_val_labels))
num_ftrs = model._fc.in_features
model._fc = nn.Linear(num_ftrs, N_CLASSES)
model._fc
```
```
Linear(in_features=1408, out_features=50, bias=True)
```

### 模型訓練的設定
```
- feature_extr_epochs: 3
- training_epochs: 20
- optimizer: AdamW
- lr scheduler: StepLR
```

### 模型訓練
1. 凍結模型全部的layer，除了最後一層的fully connected layer，只訓練最後一層，目的是為進行特徵擷取。
```python
for param in model.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model._fc.in_features
model._fc = nn.Linear(num_ftrs, N_CLASSES)
```

2. 進行fine-tuined，訓練模型全部的layer
```python
for param in model.parameters():
    param.requires_grad = True
```

3. 採用早停策略，避免overfitting的情況產生，當loss沒有小於best_loss，超過patience所設定的threshold，則訓練停止。
```python
if val_loss < best_loss:
    best_loss = val_loss
    best_epoch = epoch
    best_model_wts = copy.deepcopy(model.state_dict())
else:
    epochs_since_best += 1

# early stopping
if epochs_since_best > patience:
    print(f'Stopping training. The validation metric has not improved for {patience} epochs.')
    break
```

最後訓練完成後，所得到的valid accuracy大約落在98％，因此所訓練的模型可以很有效的分類辛普森角色。
以下是訓練過程的accuracy和loss圖，可以看到其模型在最後是有收斂的。
![accuracy](https://github.com/Machine-Learning-NYCU/the-simpsons-characters-recognition-challenge-iii-310706038/blob/master/Lab2/plot/efficientnet-b2_3FeatureExtrEpochs-20FinetuningEpochs-AccuracyCurve.png)

![loss](https://github.com/Machine-Learning-NYCU/the-simpsons-characters-recognition-challenge-iii-310706038/blob/master/Lab2/plot/efficientnet-b2_3FeatureExtrEpochs-20FinetuningEpochs-LearningCurve.png)

## 3. Confusion matrix and Feature map
### Confusion matrix
![Confusion matrix](https://github.com/Machine-Learning-NYCU/the-simpsons-characters-recognition-challenge-iii-310706038/blob/master/Lab2/plot/confusion_matrix.png)

### Feature Map
EfficientNet的第一層convolution layer只有32個kernel，因此畫出來的map為 4*8，並且得到的結果應是在抓取色塊特徵。

![Feature map](https://github.com/Machine-Learning-NYCU/the-simpsons-characters-recognition-challenge-iii-310706038/blob/master/Lab2/plot/feature_map.png)

# 改進與心得討論
## 改進

1. 本實驗只使用一個EfficientNet的模型，peformance的部分的準確率大約97~98%，因此認為可以採用stacking的方式，採用多模型進行疊加，將各個模型的優點集合起來，可以更加改善此預測的準確度，達到99或100%的精確度。


## 心得討論
此次kaggle競賽讓我學習到圖片處理的方式，加入雜訊、翻轉等等，藉此增加資料的多樣性，使得模型具有教好的效果。並且使用transfer learning的方式，訓練pretrained model，可以讓模型訓練更加有效率，且效果顯著良好。