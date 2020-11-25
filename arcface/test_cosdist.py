import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torchvision import datasets
import torch.nn as nn
import torchvision.transforms as transforms
from dataLoader import ClassFilterDataloader
from cnn_for_mnist import CNN
import pickle

# define hyper parameters
cos_threshold = 0.8  # cos距离的阈值（若两向量的cos值小于cos_threshold，则认为它们是同一类的）

# load data
test_data = datasets.MNIST(
    root='./data', train=False, download=True, transform=transforms.ToTensor())

# create the loader
test_loader = ClassFilterDataloader(test_data, range(10), 1, False, 1)

# declare the network
model = CNN()

# load the vector representation
pkl_file = open('vector_representation.pkl', 'rb')
Dict = pickle.load(pkl_file)


def classifier(data):
    """
    compute the cosine distance between data and the vector representation of each class
    if cosine distance > threshold, put data in this class
    if none of the classes satisfies this standard, return unseen
    """
    unseen = 10
    for label in Dict:
        cos = torch.cosine_similarity(Dict[label], data, dim=1)
        # print(data)
        # print(Dict[label])
        # print(cos)
        if abs(cos) > cos_threshold:
            return label
    return unseen


# test the cnn
def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            predicted = classifier(outputs)
            print(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the test images: %d %%' %(100 * correct / total))
    return 100.0*correct / total


def main():
    model.load_state_dict(torch.load('CNN_net_params.pkl'))
    test()


if __name__=='__main__':
    main()