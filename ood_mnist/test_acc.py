import torch
from torchvision import datasets, transforms
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from dataLoader import myDataset, myDataloader
from models.main_model import mainModel
from dataLoader_mnist import ClassFilterDataloader
from tensorboardX import SummaryWriter
import pdb
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# define hyper parameters
batch_size = 128

# declare the network
model = mainModel(28, 1, 10)

# gpu
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
model.device = device

# Create writer object
writer = SummaryWriter('log')


def train_classifier():
    """
    use data from five classes and date generated from the autoEncoder (view as negative data)
    to train the classifier only
    """
    model.load_state_dict(torch.load('vae_params_BCE.pkl'))

    n_epochs = 1
    LR = 0.0001

    # set the optimizer before training
    model.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    # set the optimizer before training
    train_data = datasets.MNIST(
        root='~/arcface_research/data', train=True, download=True,
        transform=transforms.ToTensor())
    test_data = datasets.MNIST(
        root='~/arcface_research/data', train=False, download=True,
        transform=transforms.ToTensor())
    # create the loader
    train_loader = ClassFilterDataloader(train_data, range(5), 1, False, batch_size)
    test_loader = ClassFilterDataloader(test_data, range(10), 1, False, batch_size)


    # view class 5,6,7,8,9 as negative data
    for epoch in range(n_epochs):
        batch = 0
        for data, target in train_loader:
            batch += 1
            data = data.to(device)
            target = target.long().to(device)
            train_loss = model.train_a_batch(data, target, epoch, batch, joint_train=False)
            model.eval()  # add this line before testing to avoid params update in the testing process

            if batch > 100:
                pred_correct = 0
                pred_cnt = 0
                ood_detect_correct = 0
                ood_detect_cnt = 0
                for data, target in test_loader:
                    data = data.to(device)
                    target = target.long().to(device)
                    output = model(data, generator_work=False, classifier_work=True)
                    predict = torch.max(output, 1)[1]
                    in_predict = torch.max(output[:, 0:5], 1)[1]
                    cnt = predict.shape[0]
                    for ind in range(cnt):
                        if target[ind] < 5:
                            # consider in-distribution-data, only consider 5 seen classes
                            pred_cnt += 1
                            if target[ind] == in_predict[ind]:
                                pred_correct += 1
                        else:
                            # consider out-of-distribution-data
                            ood_detect_cnt += 1
                            if predict[ind] == 5:
                                ood_detect_correct += 1

                pred_acc = float(pred_correct) / pred_cnt
                ood_detect_acc = float(ood_detect_correct) / ood_detect_cnt

                writer.add_scalar('pred_acc', pred_acc, global_step=batch)
                writer.add_scalar('ood_detect_acc', ood_detect_acc, global_step=batch)

                print('Epoch:', epoch, 'Batch:', batch, 'pred_Acc:', pred_acc, "ood_detect_Acc:", ood_detect_acc)


if __name__ == '__main__':
    train_classifier()
