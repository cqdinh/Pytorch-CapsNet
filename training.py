import torch
import time
from torch.autograd import Variable
from torch.nn.utils import clip_grad
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import cv2
import numpy as np


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
    '''
    :param batch_size: The batch size to use for training
    :param data_size: The width and height of the input data
    :param in_channels: The number of channels of the input data
    :param start_epoch: The initial epoch number. Used to ensure continuity when loading past runs
    :param num_epochs: The number of epochs to run
    :param learning_rate: The learning rate to initialize ADAM with
    :param train_loader: The data loader to use for training
    :param test_loader: The data loader to use for testing
    :param model: The model to train
    :param writer: The object to use to output training logs
    :param use_gpu: Whether or not to use CUDA
    :param model_save_format: The format to use to name the model logs
    :return:
    '''
    num_batches = len(train_loader)

    # Initialize the ADAM optimizer and the Exponential Learning Rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.999, 0.999999), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)

    # What step number of training to start on
    global_step = start_epoch * num_batches

    # Put the model in training mode
    model.train()
    for epoch in range(start_epoch, start_epoch + num_epochs):
        # Update the learning rate and record it
        scheduler.step()
        print("Learning Rate:", scheduler.get_lr()[0])
        writer.add_scalar("train/learning_rate", scheduler.get_lr()[0], global_step)

        # Used to estimate the training time left
        start = time.clock()

        # Iterate through each batch generated by the training data loader
        for batch_index, (x, logo_image, y, logo_index) in enumerate(train_loader):

            # Convert the image to a pytorch tensor
            x = x.type(torch.FloatTensor)

            # Convert the one-hot vector to a pytorch tensor
            y = y.type(torch.FloatTensor)
            logo_image = logo_image.type(torch.FloatTensor)

            # Calculate the step number
            global_step = batch_index + epoch * num_batches

            # Wrap the images and one-hot vectors in pytorch Variables
            x = Variable(x)
            y = Variable(y)
            logo_image = Variable(logo_image)

            # Move the variables onto the GPU if that is enabled and possible
            if torch.cuda.is_available() and use_gpu:
                x = x.cuda()
                y = y.cuda()
                logo_image = logo_image.cuda()

            # Reset the gradients
            optimizer.zero_grad()

            # Run the model
            output = model(x)

            # Calculate the loss values
            loss, margin_loss, reconstruction_loss = model.loss(output, y, logo_image)

            # If there are no NaNs, run backpropagation to calculate the gradients
            if (loss == loss).all():
                loss.backward()

            # Update the model weights
            optimizer.step()

            # Record the losses
            writer.add_scalar("train/loss", loss.data[0], global_step)
            writer.add_scalar("train/margin_loss", margin_loss.data[0], global_step)
            writer.add_scalar("train/reconstruction_loss", reconstruction_loss.data[0], global_step)

            # Print to the console so that progress can be seen without TensorBoard
            print("Epoch: {}\tBatch: {}/{}\tLoss: {:10.6f}\tMargin Loss: {:10.6f}\tReconstruction Loss: {:10.6f}".format(epoch, batch_index, num_batches, loss.data[0], margin_loss.data[0], reconstruction_loss.data[0]))

            # If the tilde key is pressed, quit. This exits gracefully which stops the GPU from crashing on exit
            if cv2.waitKey(1) == 96:
                raise KeyboardInterrupt("Interrupted")

        # Output how long the epoch took
        print("Epoch %d Took %f seconds" % (epoch, time.clock() - start))

        # Calculate the loss and accuracy of the model
        loss, accuracy = test(batch_size, data_size, in_channels, test_loader, model, writer, global_step, use_gpu)

        # Save the model
        model_path = model_save_format % (epoch, loss, accuracy)
        torch.save(model, model_path)


def test(batch_size,
         data_size,
         in_channels,
         test_loader,
         model,
         writer,
         step,
         use_gpu=True):

    '''
    :param batch_size: The batch size to use for testing
    :param data_size: The width and height of the input data
    :param in_channels: The number of channels in the input data
    :param test_loader: The data loader to use for testing
    :param model: The model to test
    :param writer: The object to use to log values
    :param step: The step that the training is currently on
    :param use_gpu: Whether or not to use CUDA
    :return:
    '''

    # Total loss values across the entire test set
    total_loss = 0
    total_margin_loss = 0
    total_reconstruction_loss = 0

    # How many images were classified correctly
    correct = 0

    # Set the model to evaluation mode
    model.eval()

    # Iterate over the test data
    for x, logo_image, y, logo_index in test_loader:

        # Convert the data into torch tensors
        x = x.type(torch.FloatTensor)
        y = y.type(torch.FloatTensor)
        logo_image = logo_image.type(torch.FloatTensor)

        # Wrap the data in Variables set to not calculate gradients. This minimizes memory usage
        x = Variable(x, volatile=True)
        y = Variable(y, volatile=True)
        logo_image = Variable(logo_image, volatile=True)

        # If enabled and possible, move data onto the GPU
        if torch.cuda.is_available() and use_gpu:
            x = x.cuda()
            y = y.cuda()
            logo_image = logo_image.cuda()

        # Calculate the output
        output = model(x)

        # Calculate the loss values and add them to the totals
        loss, margin_loss, reconstruction_loss = model.loss(output, y, logo_image)
        total_loss += loss.data[0]
        total_margin_loss += margin_loss.data[0]
        total_reconstruction_loss += reconstruction_loss.data[0]

        # Calculate the predictions from the raw outputs
        norms = torch.sqrt(torch.sum(torch.pow(output, 2), dim=2))

        # Calculate the predictions
        pred = norms.data.max(1, keepdim=True)[1].type(torch.LongTensor)

        # Print the predictions for analysis
        print(pred.transpose(0,1))
        print(logo_index.transpose(0,1))
        print(pred.eq(logo_index.type(torch.LongTensor)).transpose(0,1))

        # Remember how many predictions were correct.
        correct += pred.eq(logo_index.type(torch.LongTensor)).cpu().sum()

    # Add the visualization of the model's reconstructions to the logs
    reconstruction = model.reconstruction(output, y).view(batch_size, in_channels, data_size, data_size)
    reconstruction = vutils.make_grid(reconstruction.data, normalize=False, scale_each=True)
    writer.add_image("Image-%d" % step, reconstruction, step)
    writer.add_image("Goal_Image-%d" % step, logo_image, step)

    # Convert the totals to the means and record them
    total_loss /= len(test_loader)
    total_reconstruction_loss /= len(test_loader)
    total_margin_loss /= len(test_loader)
    accuracy = correct / len(test_loader.dataset)

    writer.add_scalar("test/loss", total_loss, step)
    writer.add_scalar("test/margin_loss", total_margin_loss, step)
    writer.add_scalar("test/reconstruction_loss", total_reconstruction_loss, step)
    writer.add_scalar("test/accuracy", accuracy, step)

    # Write the results to the console
    print("Test Loss: %f\tMargin Loss: %f\tReconstruction Loss: %f" % (total_loss, total_margin_loss, total_reconstruction_loss))
    print("Accuracy: {:10.2f}%".format(100.0 * accuracy))
    print()
    return total_loss, accuracy
