import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
import image_helper
from sklearn.preprocessing import MinMaxScaler

from model import Net


def loadData(fileList):
    helper = image_helper.ImageHelper()
    imageArrayList = []
    for file in fileList:
        print('Loading ' + file)
        try:
            helper.open(file)
            imageBlock = helper.getBlockArray(8)
            imageArrayList.append(imageBlock)
            print('Processed file ' + file)
        except:
            print('Warning- error while processing file: ' + file)

    return np.vstack(imageArrayList)


def train():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainFileList = [
        'images/somegodthing.jpg',
        'images/peppers.jpg',
        'images/rainbow.png',
        'images/firefox.jpg',
        'images/peppers-colorized.png',
        'images/cars.jpg'
    ]

    completeTrainBlock = loadData(trainFileList)

    # reshape the imageBlock so that it can make into the model
    completeTrainBlockFlat = np.reshape(completeTrainBlock, (completeTrainBlock.shape[0], 64))

    # min-max transform the array to [-10, 10] and save the transformation
    transformer = MinMaxScaler(feature_range=(-1, 1))
    transformer.fit(completeTrainBlockFlat)
    torch.save(transformer, 'transformer')

    completeTrainBlockFlatNorm = transformer.transform(completeTrainBlockFlat)
    np.random.shuffle(completeTrainBlockFlatNorm)

    net = Net()
    # net = torch.load('model.pth').to(device)

    num_epochs = 1000
    batch_size = 8192

    losslist = np.zeros(num_epochs)

    print(device)

    optimizer = optim.RMSprop(net.parameters(), lr=1e-4, weight_decay=3e-6)
    criterion = nn.MSELoss()

    net.to(device)

    for epoch in range(num_epochs):
        for i in range(int(completeTrainBlockFlatNorm.shape[0] / batch_size) + 1):
            inp = torch.tensor(completeTrainBlockFlatNorm[(i * batch_size): (i * batch_size + batch_size)],
                               dtype=torch.float).to(device)
            if inp.nelement() != 0:
                target = torch.tensor(completeTrainBlockFlatNorm[(i * batch_size): (i * batch_size + batch_size)],
                                      dtype=torch.float).to(device)

                optimizer.zero_grad()

                outputs = net(inp)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()

                losslist[epoch] = loss.item()

        print('Loss at epoch', epoch, 'is', losslist[epoch])

    torch.save(net, 'model.pth')
    torch.save(net.encoder, 'encoder.pth')
    torch.save(net.decoder, 'decoder.pth')

    # Checking the compression effectiveness
    helper1 = image_helper.ImageHelper()
    helper1.open('images/somegodthing.jpg', (256, 256))
    imageBlock = helper1.getBlockArray(8)
    imageBlockTesting = np.reshape(imageBlock, (imageBlock.shape[0], 64))
    imageBlockTestingNorm = transformer.transform(imageBlockTesting)
    opTensor = net(torch.tensor(imageBlockTestingNorm, dtype=torch.float).to(device))
    opArr = opTensor.cpu().detach().numpy()
    newImageArrFlat = transformer.inverse_transform(opArr)

    newImageArr = np.clip(np.reshape(newImageArrFlat, imageBlock.shape), 0, 255)

    newhelper = helper1.fromDCTForm(newImageArr)
    newhelper.image.show()
