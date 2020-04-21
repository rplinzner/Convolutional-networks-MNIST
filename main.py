from Net import Net
from Data import DataService
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from time import time
import torchvision
import numpy as np
import torch
import ctypes
ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')


# Hyperparameters
num_epochs = 3
num_classes = 10
batch_size = 256
learning_rate = 0.001
momentum = 0.5

model_name = 'mymodel'
feature_extracting_enabled = True  # only for squeezenet
optimizer_name = 'adam'

log_interval = 10
chart_interval = 2


DATA_PATH = 'C:\\MNISTData'

EXPERIMENT_NAME = 'my_model_1'

RESULTS_STORE_PATH = 'C:\\results\\' + EXPERIMENT_NAME + '\\'

comp_unit = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# comp_unit = torch.device("cpu")


# transforms to apply to the data
if model_name == 'squeezenet':
    trans = transforms.Compose(
        [transforms.Resize(224), transforms.ToTensor(), transforms.Lambda(lambda x: x.expand(3, -1, -1)),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
             0.229, 0.224, 0.225])
         ])
else:
    trans = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])


results = DataService(RESULTS_STORE_PATH)

train_dataset = torchvision.datasets.MNIST(
    root=DATA_PATH, train=True, transform=trans, download=True)
test_dataset = torchvision.datasets.MNIST(
    root=DATA_PATH, train=False, transform=trans)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, shuffle=False)

# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
# https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/
# https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627


def train(model, optimizer, criterion, epoch):
    time0 = time()
    model.train()

    running_loss = 0.0
    correct = 0

    conf_matrix = np.zeros([num_classes, num_classes], int)

    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        images = images.to(comp_unit)
        labels = labels.to(comp_unit)
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        correct += (predicted == labels).sum().item()

        for labels, predict in zip(labels.view(-1), predicted.view(-1)):
            conf_matrix[labels.long(), predict.long()] += 1

        if i % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * len(images), len(train_loader.dataset),
                100. * i / len(train_loader), loss.item()))

    training_time = time() - time0
    training_loss = running_loss / len(train_loader)

    if epoch % chart_interval == 0:
        results.train_losses.append(training_loss)
    results.saveTrainingStats(
        epoch, running_loss, training_loss, correct, len(train_loader.dataset), training_time, conf_matrix)

    print('\nEpoch {} - Train set: Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) Time: {:.0f}min {:.0f}sek\n'.format(epoch,
                                                                                                               training_loss, correct, len(
                                                                                                                   train_loader.dataset),
                                                                                                               100. * correct / len(train_loader.dataset), training_time // 60, training_time % 60))
    print(conf_matrix)


def test(model, criterion, epoch):
    time0 = time()
    model.eval()

    test_loss = 0.0
    correct = 0
    conf_matrix = np.zeros([num_classes, num_classes], int)

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(comp_unit)
            labels = labels.to(comp_unit)

            output = model(images)
            loss = criterion(output, labels)

            _, predicted = torch.max(output.data, 1)

            if epoch == num_epochs - 1:
                temp = (predicted == labels).squeeze()
                for j in range(len(images)):
                    if temp[j].item() == False:
                        save_image(
                            images[j], f'{results.photos_dir}{i*len(images)+j}_label_{labels[j]}_predicted_{predicted[j]}.jpg')

            test_loss += loss.item()
            correct += (predicted == labels).sum().item()
            for labels, predict in zip(labels.view(-1), predicted.view(-1)):
                conf_matrix[labels.long(), predict.long()] += 1

    training_time = time() - time0
    avg_loss = test_loss / len(test_loader)

    for t, p in zip(labels.view(-1), predicted.view(-1)):
        conf_matrix[t.long(), p.long()] += 1
    results.saveTestingStats(
        test_loss, avg_loss, correct, len(test_loader.dataset), training_time, conf_matrix, epoch)

    print('\nTest set: Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)  Time: {:.0f}min {:.0f}sek\n'.format(
        avg_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset), training_time // 60, training_time % 60))
    print(conf_matrix)

# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#set-model-parameters-requires-grad-attribute


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def main():
    if model_name == 'squeezenet':
        model = torchvision.models.squeezenet1_0(
            pretrained=False, num_classes=num_classes)
        set_parameter_requires_grad(model, feature_extracting_enabled)
        model.classifier[1] = nn.Conv2d(
            512, num_classes, kernel_size=(1, 1), stride=(1, 1))
    else:
        model = Net()

    model.to(comp_unit)

    criterion = nn.CrossEntropyLoss()
    if optimizer_name == adam:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(),
                              lr=learning_rate, momentum=momentum)

    for e in range(num_epochs):
        train(model, optimizer, criterion, e)
        test(model, criterion, e)
        torch.save(model.state_dict(), results.model_dir +
                   f'model_epoch_{e}.ckpt')
    results.generateTrainChart(chart_interval)
    print('done')


if __name__ == "__main__":
    main()
