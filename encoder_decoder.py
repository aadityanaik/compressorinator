import torch
import torch.nn as nn
from model import Net, Encoder, Decoder
from bitstring import BitArray, BitStream
import numpy as np
import image_helper
from sklearn.preprocessing import MinMaxScaler
import PIL.Image as img
import sys


def quantize(arr, levels):
    min = arr.min()
    max = arr.max()

    quantRange = (max - min) / levels

    quantLevels = []

    for i in range(1, levels * 2, 2):
        quantLevels.append(min + quantRange * i / 2)

    quantLevels = np.array(quantLevels)

    def assignQuantLevel(x):
        levelInd = abs(x - quantLevels).argmin()
        return quantLevels[levelInd]

    assignQuantLevel = np.vectorize(assignQuantLevel)

    quantizedArr = assignQuantLevel(arr)

    return quantizedArr.astype(np.float16), quantLevels.astype(np.float16)


def encodeImage(infile, outfile):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    helper = image_helper.ImageHelper()

    print('Opening image...')
    try:
        helper.open(infile)
    except:
        print('ERROR- COULD NOT OPEN FILE ' + infile)
        return False
    print('Success- Image opened')

    imageBlock = helper.getBlockArray(8)
    imageBlockInput = np.reshape(imageBlock, (imageBlock.shape[0], 64))

    print('Loading transformer...')
    transformer = torch.load('transformer')
    print('Transformer loaded')

    imageBlockInputNorm = transformer.transform(imageBlockInput)

    print('Loading encoder')
    encoder = torch.load('encoder.pth').to(device)
    print('Encoder model loaded')

    print('Encoding...')
    encoderOp = encoder(torch.tensor(imageBlockInputNorm, dtype=torch.float).to(device))
    print('Success- encoded')

    print('Quantizing...')
    quantizedOp, quantLevels = quantize(encoderOp.cpu().detach().numpy(), 16)

    # since we use 16 levels, we assign a dictionary to map the levels to 4 bits
    quantLevelMapping = dict(zip(quantLevels, '0x0 0x1 0x2 0x3 0x4 0x5 0x6 0x7 0x8 0x9 0xa 0xb 0xc 0xd 0xe 0xf'.split()))

    bitstring = ''

    for x in np.nditer(quantizedOp):
        bitstring += (quantLevelMapping[x.item()] + ' ')

    bitstring = BitArray(bitstring)
    print('Quantized')

    print('Size of bitstring- ', int(len(bitstring) / 8))
    print('Compression rate without overhead- ', (100 * (1 - len(bitstring) / (np.prod(helper.imageArray.shape) * 8))))

    print('Saving encoded')
    torch.save((quantLevelMapping, quantizedOp.shape, helper.imageArray.shape, bitstring), outfile)
    print('Encoded saved')


def decodeCompressed(compressedFile):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Loading decoder...')
    decoder = torch.load('decoder.pth').to(device)
    print('Decoder loaded')

    print('Loading transformer...')
    transformer = torch.load('transformer')
    print('Transformer loaded')

    helper = image_helper.ImageHelper()

    print('Opening data...')
    try:
        quantDict, newEncodeIPArrShape, newImageShape, bitstr = torch.load(compressedFile)

        assert quantDict is not None and newEncodeIPArrShape is not None and newImageShape is not None and bitstr is not None


    except:
        print('ERROR- COULD NOT OPEN FILE ' + compressedFile)
        return False
    print('Success- Data opened')

    # reverse the dictionary
    hexQuantMap = {}
    for key, val in quantDict.items():
        hexQuantMap[val] = key

    newQuantArr = []

    print('Reconstructing image')
    for i in bitstr.hex:
        newQuantArr.append(hexQuantMap['0x' + i])

    newQuantArr = np.reshape(np.array(newQuantArr), newEncodeIPArrShape)

    print('Decompressing...')
    decompressingTensor = torch.tensor(newQuantArr, dtype=torch.float).to(device)
    opTensor = decoder(decompressingTensor)
    newImageArrFlat = transformer.inverse_transform(opTensor.cpu().detach().numpy())
    newImageArrFlat = np.clip(newImageArrFlat, 0, 255)
    print('Decompressed')

    newImageShape = tuple([int(x) for x in newImageShape])

    finalImage = np.reshape(newImageArrFlat, newImageShape)
    if finalImage.shape[2] == 1:
        finalImage = finalImage[:, :, 0]

    print('Finished Recontruction')

    helper.imageArray = finalImage.astype(np.uint8)
    helper.image = img.fromarray(helper.imageArray)

    return helper
