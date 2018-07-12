import cv2
import time
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.autograd import Variable
from torch.nn.utils import clip_grad

class LogoConvNet(nn.Module):
    def __init__(self,
                 num_categories,
                 in_size,
                 in_channels,
                 conv_out_channels,
                 conv_kernel_size,
                 num_primary_caps,
                 primary_out_channels,
                 primary_kernel_size,
                 primary_stride,
                 digit_out_channels,
                 decoder_size_1,
                 decoder_size_2,
                 reconstruction_loss_weight,
                 use_gpu,
                 routing_iterations,
                 absent_loss_weight
                 ):
        '''
        :param num_categories: The number of categories to classify into
        :param in_size: The width and height of the input data
        :param in_channels: The number of channels of the input data
        :param conv_out_channels: The number of kernels to use in the first convolutional layer
        :param conv_kernel_size: The size of the kernesl in the first convolutional layer
        :param num_primary_caps: The number of primary capsules used in the CapsNet.
                Used to calculate the number of kernels in the second convolutional layer
        :param primary_out_channels: The number of kernels per capsule in the CapsNet.
                Used to calculate the number of kernels in the second convolutional layer
        :param primary_kernel_size: The size of the kernels in the PrimaryCaps layer of the CapsNet and the second
                convolutional layer.
        :param primary_stride: The stride used in the PrimaryCaps layer of the CapsNet and the second convolutional layer
        :param digit_out_channels: The number of dimensions that the DigitCaps projects the vectors into
        :param decoder_size_1: Not used. Only present so that it is easy to copy-paste parameters from the CapsNet
        :param decoder_size_2: Not used. Only present so that it is easy to copy-paste parameters from the CapsNet
        :param reconstruction_loss_weight: Not used. Only present so that it is easy to copy-paste parameters from the CapsNet
        :param use_gpu: Whether or not to use CUDA
        :param routing_iterations: Not used. Only present so that it is easy to copy-paste parameters from the CapsNet
        :param absent_loss_weight: Not used. Only present so that it is easy to copy-paste parameters from the CapsNet
        '''

        super(LogoConvNet, self).__init__()
        self.in_size = in_size
        self.num_categories = num_categories
        self.use_gpu = use_gpu

        # Initialize the layers
        self.conv1 = nn.Conv2d(in_channels, conv_out_channels, conv_kernel_size)
        # Calculate the number of kernels as the number of primary capsules multiplied by the kernels per capsule
        self.conv2 = nn.Conv2d(conv_out_channels, primary_out_channels * num_primary_caps, primary_kernel_size, primary_stride)

        # The output size of the second convolutional layer
        outsize = (in_size - (conv_kernel_size - 1) - (primary_kernel_size - 1)) // primary_stride + 1

        # Regular linear layers to the output

        # Project the data into the correct number of total dimensions
        self.fc1 = nn.Linear(primary_out_channels * num_primary_caps * outsize * outsize, digit_out_channels * num_primary_caps * outsize * outsize)

        # Calculate the equivalents of the pose vectors
        self.fc2 = nn.Linear(digit_out_channels * num_primary_caps * outsize * outsize, digit_out_channels * num_categories)

        # Calculate the output logits
        self.fc3 = nn.Linear(digit_out_channels * num_categories, num_categories)
        print(outsize)

    def forward(self, x):
        batch_size = x.size(0)
        # Run the data through the convolutional layers with leaky relu activations
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))

        # Flatten the data and run it through the linear layers with leaky relu activations
        x = x.view(batch_size, -1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))

        # Use sigmoid activation to get probabilities
        x = F.sigmoid(self.fc3(x))
        return x

    def loss(self, output, y):
        # Mean squared error
        return torch.sum((output - y)**2)

# Train the model
# See training.py for more info. These are essentially the same but with simpler error calculations
def train(
        batch_size,
        data_size,
        in_channels,
        start_epoch,
        num_epochs,
        learning_rate,
        train_loader,
        test_loader,
        model,
        writer,
        use_gpu,
        model_save_format):
    num_batches = len(train_loader)

    optimizer = torch.optim.Adam(model.parameters(), betas=(0.999, 0.999999), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)
    global_step = start_epoch * num_batches
    model.train()
    for epoch in range(start_epoch, start_epoch + num_epochs):
        scheduler.step()
        print("Learning Rate:", scheduler.get_lr()[0])
        writer.add_scalar("train/learning_rate", scheduler.get_lr()[0], global_step)

        start = time.clock()
        for batch_index, (x, logo_image, y, logo_index) in enumerate(train_loader):
            x = x.type(torch.FloatTensor)
            y = y.type(torch.FloatTensor)
            logo_image = logo_image.type(torch.FloatTensor)

            global_step = batch_index + epoch * num_batches

            x = Variable(x)
            y = Variable(y)
            if torch.cuda.is_available() and use_gpu:
                x = x.cuda()
                y = y.cuda()

            optimizer.zero_grad()

            output = model(x)

            loss = model.loss(output, y)
            if (loss == loss).all():
                loss.backward()
            else:
                print("broken")

            print("Total Gradient Norm:", clip_grad.clip_grad_norm(model.parameters(), 100))
            optimizer.step()

            writer.add_scalar("train/loss", loss.data[0], global_step)

            print("Epoch: {}\tBatch: {}/{}\tLoss: {:10.6f}".format(epoch, batch_index, num_batches, loss.data[0]))
            if cv2.waitKey(1) == 96:
                raise KeyboardInterrupt("Interrupted")

        print("Epoch %d Took %f seconds" % (epoch, time.clock() - start))
        loss, accuracy = test(test_loader, model, writer, global_step, use_gpu)
        torch.save(model, model_save_format % (epoch, loss, accuracy))


def test(test_loader,
         model,
         writer,
         step,
         use_gpu=True):

    total_loss = 0

    correct = 0

    model.eval()
    for x, logo_image, y, logo_index in test_loader:
        x = x.type(torch.FloatTensor)
        y = y.type(torch.FloatTensor)

        x = Variable(x, volatile=True)
        y = Variable(y, volatile=True)
        if torch.cuda.is_available() and use_gpu:
            x = x.cuda()
            y = y.cuda()

        output = model(x)
        loss = model.loss(output, y)
        total_loss += loss.data[0]

        pred = output.data.max(1, keepdim=True)[1].type(torch.LongTensor)
        print(pred.transpose(0,1))
        print(logo_index.transpose(0,1))
        print(pred.eq(logo_index.type(torch.LongTensor)).transpose(0,1))

        correct += pred.eq(logo_index.unsqueeze(0).type(torch.LongTensor)).cpu().sum()

    total_loss /= len(test_loader)
    accuracy = correct / len(test_loader.dataset)
    writer.add_scalar("test/loss", total_loss, step)
    writer.add_scalar("test/accuracy", accuracy, step)

    print("Test Loss: %f" % (total_loss))
    print("Accuracy: {:10.2f}%".format(100.0 * accuracy))
    print()
    return total_loss, accuracy
