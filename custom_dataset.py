import  torch,torchvision
import numpy as np
from PIL import Image
import pandas as pd
import pathlib
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

path=pathlib.Path('/media/rahul/927f87f9-ee25-4f34-a75c-7d5e3e893ad3/Deep Learning/Deep Learning/Deep Learning/data/VOC/PASCAL_VOC/tmp')
classification_csv=path/'classification.csv'

class custom_dataset(Dataset):
    def __init__(self,csv_path):
        self.to_tensor=torchvision.transforms.ToTensor()
        self.data=pd.read_csv(csv_path)
        self.image_id=np.array(self.data.iloc[:,0])
        self.labels=np.array(self.data.iloc[:,1])

    def __getitem__(self,index):
        image=Image.open('/media/rahul/927f87f9-ee25-4f34-a75c-7d5e3e893ad3/Deep Learning/Deep Learning/Deep Learning/data/VOC/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/'+self.image_id[index])
        image=image.resize((224,224),Image.ANTIALIAS)
        label=self.labels[index]
        x=self.to_tensor(image)
        y=label
        return (x,y)

    def __len__(self):
        return len(self.data.index)

dataset=custom_dataset(classification_csv)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader=torch.utils.data.DataLoader(dataset=dataset,batch_size=10,shuffle=False)

class net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=torch.nn.Conv2d(3,8,kernel_size=(3,3),padding=1)
        self.conv1_bn=torch.nn.BatchNorm2d(8)
        self.conv2=torch.nn.Conv2d(8,16,kernel_size=(3,3),padding=1)
        self.conv2_bn=torch.nn.BatchNorm2d(16)
        self.conv3=torch.nn.Conv2d(16,32,kernel_size=(3,3),padding=1)
        self.conv3_bn=torch.nn.BatchNorm2d(32)
        self.conv4=torch.nn.Conv2d(32,64,kernel_size=(3,3),padding=1)
        self.conv4_bn=torch.nn.BatchNorm2d(64)
        self.conv5=torch.nn.Conv2d(64,128,kernel_size=(3,3),padding=1)
        self.conv5_bn=torch.nn.BatchNorm2d(128)
        self.fc1=torch.nn.Linear(6272,500)
        self.fc2=torch.nn.Linear(500,400)
        self.fc3=torch.nn.Linear(400,20)

    def forward(self,x):
        out=F.relu(F.max_pool2d(self.conv1_bn(self.conv1(x)),2))
        out=F.relu(F.max_pool2d(self.conv2_bn(self.conv2(out)),2))
        out=F.relu(F.max_pool2d(self.conv3_bn(self.conv3(out)),2))
        out=F.relu(F.max_pool2d(self.conv4_bn(self.conv4(out)),2))
        out=F.relu(F.max_pool2d(self.conv5(out),2))
        out=out.view(out.size(0),-1)
        out=F.relu(self.fc1(out))
        out=F.dropout(out)
        out=F.relu(self.fc2(out))
        out=F.dropout(out)
        out=self.fc3(out)
        return F.log_softmax(out,dim=1)

model=net().to(device)
criterion=torch.nn.CrossEntropyLoss().to(device)
optimizer=torch.optim.Adam(model.parameters())

epoch=10
from torch.autograd import Variable
for i in range(epoch):
    Total_loss=0
    Train_accuracy=0
    Total=0
    model.train()
    for b_idx,(images,target) in enumerate(train_loader):
        images,target=Variable(images.to(device)),Variable(target.to(device))
        optimizer.zero_grad()
        output=model(images)
        loss=criterion(output,target)
        loss.backward()
        optimizer.step()
    print(output)
