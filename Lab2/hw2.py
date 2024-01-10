#!/usr/bin/env python
# coding: utf-8

# In[19]:


# ignore deprication warnings
import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

# standard python modules
import os, sys
import time


# standard ml modules
import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, colors
# work in interactive moode
# get_ipython().run_line_magic('matplotlib', 'inline')


# loading files (in parallel)
from pathlib import Path
from multiprocessing.pool import ThreadPool


# working with images
import PIL
from PIL import Image
# from skimage import io

# preprocessing
from sklearn.preprocessing import LabelEncoder


# torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
# torchvision
import torchvision
from torchvision import transforms
# import os
# import torchvision.transforms.v2 as transform
# from PIL import Image
# import torch

# interacrive timimg
from tqdm import tqdm, tqdm_notebook

# saving models 
import pickle
import copy


# In[20]:


train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')


# In[21]:


# разные режимы датасета 
DATA_MODES = ['train', 'val', 'test']
# все изображения будут масштабированы к размеру 224x224 px
RESCALE_SIZE = 224
# работаем на видеокарте
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[22]:


SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


# In[23]:


import os
import torchvision.transforms.v2 as T
from PIL import Image
import torch

# Custom transform to add Gaussian noise
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

# Custom transform to add Speckle noise
class AddSpeckleNoise(object):
    """
    Add speckle noise to the image.
    """
    def __init__(self, noise_level=0.1):
        """
        :param noise_level: Standard deviation of the noise distribution
        """
        self.noise_level = noise_level

    def __call__(self, tensor):
        """
        :param tensor: PyTorch tensor, the image on which noise is added
        :return: PyTorch tensor, image with speckle noise
        """
        # Generate speckle noise
        noise = torch.randn_like(tensor) * self.noise_level

        # Add speckle noise to the image
        noisy_tensor = tensor * (1 + noise)

        # Clip the values to be between 0 and 1
        noisy_tensor = torch.clamp(noisy_tensor, 0, 1)

        return noisy_tensor

class AddPoissonNoise(object):
    """
    Add Poisson noise to the image.
    """
    def __init__(self, lam=1.0):
        """
        :param lam: Lambda parameter for Poisson distribution
        """
        self.lam = lam

    def __call__(self, tensor):
        """
        :param tensor: PyTorch tensor, the image to which noise is added
        :return: PyTorch tensor, image with Poisson noise
        """
        # Generate Poisson noise
        noise = torch.poisson(self.lam * torch.ones(tensor.shape))

        # Add Poisson noise to the image
        noisy_tensor = tensor + noise / 255.0  # Assuming the image is scaled between 0 and 1

        # Clip the values to be between 0 and 1
        noisy_tensor = torch.clamp(noisy_tensor, 0, 1)

        return noisy_tensor

# Custom transform to add Salt and Pepper noise
class AddSaltPepperNoise(object):
    def __init__(self, salt_prob=0.05, pepper_prob=0.05):
        self.salt_prob = salt_prob
        self.pepper_prob = pepper_prob

    def __call__(self, tensor):
        noise = torch.rand(tensor.size())
        tensor[(noise < self.salt_prob)] = 1  # Salt noise: setting some pixels to 1
        tensor[(noise > 1 - self.pepper_prob)] = 0  # Pepper noise: setting some pixels to 0
        return tensor




# In[24]:



class SimpsonsDataset(Dataset):
    """
    Class to work with image dastaset, which
    - loads them form the folders in parallel
    - converts to PyTorch tensors
    - scales the tensors to have mean = 0, standard deviation = 1
    """
    def __init__(self, files, mode):
        super().__init__()
        self.files = sorted(files) # list of files to be loaded
        self.mode = mode           # working mode

        if self.mode not in DATA_MODES:
            print(f"{self.mode} is not correct; correct modes: {DATA_MODES}")
            raise NameError

        self.len_ = len(self.files)
     
        self.label_encoder = LabelEncoder()

        if self.mode != 'test':
            self.labels = [path.parent.name for path in self.files]
            self.label_encoder.fit(self.labels)

            with open('label_encoder.pkl', 'wb') as le_dump_file:
                  pickle.dump(self.label_encoder, le_dump_file)

        if self.mode == "test":
            self.id = [int(os.path.basename(path).replace(".jpg", "")) for path in self.files]


                
    
    def __len__(self):
        return self.len_
    
    
    def load_sample(self, file):
        image = Image.open(file)
        image.load()
        return image
    
    
    def _prepare_sample(self, image):
        image = image.resize((RESCALE_SIZE, RESCALE_SIZE))
        return np.array(image)
    
    
    def __getitem__(self, index):
        # converts to PyTorch tensors and normalises the input
        
        data_transforms = {
            'train': transforms.Compose([
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
            ,
            'val_test': transforms.Compose([
                transforms.Resize(size=(RESCALE_SIZE, RESCALE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
            ]),
        }

        transform = (data_transforms['train'] if self.mode == 'train' else data_transforms['val_test'])
        
        x = self.load_sample(self.files[index])  # load image
        x = transform(x)                         # apply transform defined above
        
        if self.mode == 'test':
            return x, self.id[index]
        else:
            label = self.labels[index]
            label_id = self.label_encoder.transform([label])
            y = label_id.item()
            return x, y


# In[25]:


def imshow(inp, title=None, plt_ax=plt, default=False):
    """Imshow для тензоров"""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt_ax.imshow(inp)
    if title is not None:
        plt_ax.set_title(title)
    plt_ax.grid(False)


# In[26]:


TRAIN_DIR = Path('../train/train/')
TEST_DIR = Path('../test-final/test-final/')

train_val_files = sorted(list(TRAIN_DIR.rglob('*/*.jpg')))
test_files = sorted(list(TEST_DIR.rglob('*.jpg')))


# In[27]:


print(len(train_val_files), 'train files')
train_val_files[:5]


# In[28]:


print(len(test_files), 'test files')
test_files[:5]


# In[29]:


# path.parent.name returns a folder in which the image is, which corresponds to the label in nthis case
train_val_labels = [path.parent.name for path in train_val_files]


# In[30]:


print(len(train_val_labels), 'train_val_labels')
train_val_labels[:5]


# In[31]:


from sklearn.model_selection import train_test_split
train_files, val_files = train_test_split(train_val_files, test_size=0.20, stratify=train_val_labels)


# In[32]:



val_dataset = SimpsonsDataset(val_files, mode='val')


# In[33]:


fig, ax = plt.subplots(nrows=3, ncols=3,figsize=(8, 8),                         sharey=True, sharex=True)

for fig_x in ax.flatten():
    random_characters = int(np.random.uniform(0,1000))
    im_val, label = val_dataset[random_characters]
    img_label = " ".join(map(lambda x: x.capitalize(),                val_dataset.label_encoder.inverse_transform([label])[0].split('_')))
    imshow(im_val.data.cpu(),           title=img_label,plt_ax=fig_x)


# In[34]:


def fit_epoch(model, train_loader, criterion, optimizer):
    # initialize tracked variables
    running_loss = 0.0
    running_corrects = 0
    processed_data = 0
  
    for inputs, labels in train_loader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        
        # reset the gradient
        optimizer.zero_grad()
        
        # predictions (probabilities), loss, backprop
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # weights update
        optimizer.step()
        
        # predictions (classes)
        preds = torch.argmax(outputs, 1)
        
        # record tracked items
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        processed_data += inputs.size(0)
        
    # record train loss and train accuracy          
    train_loss = running_loss / processed_data
    train_acc = running_corrects.cpu().numpy() / processed_data
    return train_loss, train_acc


# In[35]:


def eval_epoch(model, val_loader, criterion):
    # set model model into the evaluation mode (e.g. for Dropout)
    model.eval()
    
    # initialize tracked variables
    running_loss = 0.0
    running_corrects = 0
    processed_size = 0

    for inputs, labels in val_loader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, 1)
        
        # record tracked items
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        processed_size += inputs.size(0)
        
    # record val loss and val accuracy
    val_loss = running_loss / processed_size
    val_acc = running_corrects.double() / processed_size
    return val_loss, val_acc.cpu().detach().numpy()


# In[36]:



def train(train_dataset, val_dataset, model, criterion,
          epochs, batch_size, optimizer, scheduler,
          shuffle=True, sampler=None, patience=5):
    
    # to record the total training time
    since = time.time()
    
    # note: 4 workers loading the data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # init variables to store best model weights, best accuracy, best epoch number, epochs since best accuracy acheived
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10
    best_epoch = 0
    epochs_since_best = 0
    
    # history and log
    history = []
    log_template = "\nEpoch {ep:03d} train_loss: {t_loss:0.4f}     val_loss {v_loss:0.4f} train_acc {t_acc:0.4f} val_acc {v_acc:0.4f}"

    with tqdm(desc="epoch", total=epochs) as pbar_outer:

        for epoch in range(1, epochs+1):
            print(f"epoch {epoch}:\n")
            
            print("Fitting on train data...")
            # all arguments except train loader are from parameters passed to train() arguments
            train_loss, train_acc = fit_epoch(model, train_loader, criterion, optimizer)
            print("train loss:", train_loss)
            
            print("Evaluating on validation data...")
            val_loss, val_acc = eval_epoch(model, val_loader, criterion)
            print("val loss:", val_loss)
            
            # record history
            history.append((train_loss, train_acc, val_loss, val_acc))
            
            # update learning rate for the optimizer
            scheduler.step()
            
            # display learning status
            pbar_outer.update(1)
            tqdm.write(log_template.format(ep=epoch, t_loss=train_loss,                                           v_loss=val_loss, t_acc=train_acc, v_acc=val_acc))
            
            # deep copy the model if it acheives the best validation performance
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
            
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    print('Best epoch: {}'.format(best_epoch))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
            
    return history


# In[37]:


def predict(model, test_loader):
    with torch.no_grad():
        logits = []
        ids = []
        preds = []
        for inputs, id in test_loader:
            inputs = inputs.to(DEVICE)
            model.eval()
            outputs = model(inputs).cpu()
            preds.append(torch.argmax(outputs, 1))
            logits.append(outputs)
            ids.append(id)
    probs = nn.functional.softmax(torch.cat(logits), dim=-1).numpy()
    ids = torch.cat(ids).numpy()
    return probs, ids, preds


# In[38]:


N_CLASSES = len(np.unique(train_val_labels))


# In[39]:


if val_dataset is None:
    val_dataset = SimpsonsDataset(val_files, mode='val')
    
train_dataset = SimpsonsDataset(train_files, mode='train')


# In[40]:



from efficientnet_pytorch import EfficientNet


# In[41]:


model_name = 'efficientnet-b2'
# model_name = 'resnet18'


# In[42]:


# model = torchvision.models.resnet18(pretrained=True)
model = EfficientNet.from_pretrained(model_name)


# In[43]:


model


# In[44]:


for param in model.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model._fc.in_features
# num_ftrs = model.fc.in_features

model._fc = nn.Linear(num_ftrs, N_CLASSES)
# model.fc = nn.Linear(num_ftrs, N_CLASSES)

# to GPU
model = model.to(DEVICE)

# loss
criterion = nn.CrossEntropyLoss()

# learning rate optimizer
optimizer = torch.optim.AdamW(model.parameters())

# scheduler for the lr optimizer
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, 0.5)


# In[45]:


model._fc
# model.fc


# In[46]:


# feature_extr_epochs = 1 # test run
feature_extr_epochs = 3 # performance run


# In[47]:


history_feature_extr = train(train_dataset, val_dataset, model=model, criterion=criterion,
                             epochs=feature_extr_epochs, batch_size=64, optimizer=optimizer, scheduler=scheduler)


# In[48]:


loss, acc, val_loss, val_acc = zip(*history_feature_extr)


# In[ ]:


plt.figure(figsize=(15, 9))
plt.plot(loss, label="train_loss")
plt.plot(val_loss, label="val_loss")
plt.legend(loc='best')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.savefig(f"plot/{model_name}_feature_extraction{feature_extr_epochs}.png")
# plt.show()


# In[30]:


for param in model.parameters():
    param.requires_grad = True


# In[31]:


finetuning_epochs = 20 # test run
# finetuning_epochs = 50 # performance run


# In[32]:


history_fine_tune = train(train_dataset=train_dataset, val_dataset=val_dataset, model=model, criterion=criterion,
                          epochs=finetuning_epochs, batch_size=16, optimizer=optimizer, scheduler=scheduler)


# In[ ]:


loss, acc, val_loss, val_acc = zip(*history_fine_tune)


# In[50]:


plt.figure(figsize=(15, 9))
plt.plot(acc, label="train_acc")
plt.plot(val_acc, label="val_acc")
plt.legend(loc='best')
plt.xlabel("epochs")
plt.ylabel("accuracy")


plt.savefig(f"plot/{model_name}_{feature_extr_epochs}FeatureExtrEpochs-{finetuning_epochs}FinetuningEpochs-AccuracyCurve.png")
# plt.show()


# In[ ]:


plt.figure(figsize=(15, 9))
plt.plot(loss, label="train_loss")
plt.plot(val_loss, label="val_loss")
plt.legend(loc='best')
plt.xlabel("epochs")
plt.ylabel("loss")


plt.savefig(f"plot/{model_name}_{feature_extr_epochs}FeatureExtrEpochs-{finetuning_epochs}FinetuningEpochs-LearningCurve.png")
# plt.show()


# In[ ]:


f"{model_name}_{feature_extr_epochs}FeatureExtrEpochs-{finetuning_epochs}FinetuningEpochs-LearningCurve.png"


# In[ ]:


model_weights = copy.deepcopy(model.state_dict())
torch.save(model_weights, f"{model_name}_{feature_extr_epochs}FeatureExtrEpochs-{finetuning_epochs}FinetuningEpochs-weights.pth")


# In[ ]:


model.load_state_dict(torch.load(f"{model_name}_{feature_extr_epochs}FeatureExtrEpochs-{finetuning_epochs}FinetuningEpochs-weights.pth"))
model


# In[ ]:


test_dataset = SimpsonsDataset(test_files, mode="test")
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=64, num_workers=4)


# In[ ]:


preds, ids = predict(model, test_loader)


# In[ ]:


label_encoder = pickle.load(open("label_encoder.pkl", 'rb'))
preds = label_encoder.inverse_transform(np.argmax(preds, axis=1))


# In[ ]:


my_submit = pd.DataFrame({'id': ids, 'character': preds})
my_submit = my_submit.sort_values(by=['id'])
print(my_submit.shape)
my_submit.head(10)


# In[ ]:


my_submit.to_csv(f"output/my_submit_{model_name}_{finetuning_epochs}.csv", index=False)


# ## Confusion Matrix

# In[28]:


model.load_state_dict(torch.load(f"model/efficientnet-b2_3FeatureExtrEpochs-20FinetuningEpochs-weights.pth"))
val_loader = DataLoader(val_dataset, shuffle=False, batch_size=64, num_workers=4)
logits, ids, preds = predict(model, val_loader)
preds = torch.cat(preds).numpy()


# In[85]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
cf_matrix = confusion_matrix(ids, preds) 
per_cls_acc = cf_matrix.diagonal()/cf_matrix.sum(axis=0)   
class_names = list(set(train_val_labels))
df_cm = pd.DataFrame(cf_matrix, class_names, class_names) 
plt.figure(figsize = (9,6))
sns.heatmap(df_cm, annot=False, fmt="d")
plt.xlabel("prediction")
plt.ylabel("label (ground truth)")
plt.savefig("plot/confusion_matrix.png")


# ## Visualization and Understanding Convoutional Neural Networks

# In[50]:


weights = model._conv_stem.weight.data.cpu().numpy()
weight_min = np.min(weights)
weight_max = np.max(weights)
weights = (weights-weight_min)/(weight_max-weight_min)


# In[51]:


n_kernels = weights.shape[0]
n_rows = int(np.ceil(n_kernels/8))
fig, axarr = plt.subplots(n_rows, 8)


# In[55]:


for idx in range(n_kernels):
    row = idx//8
    col = idx%8
    ax = axarr[row,col]
    ax.imshow(np.transpose(weights[idx], (1,2,0)), interpolation='nearest')
    ax.axis('off')
plt.show()


# In[63]:


fig.savefig("plot/feature_map.png")

