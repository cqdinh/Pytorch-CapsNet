from dataIO import DataGen
from CapsNet import LogoCapsNet
import torch
import os
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import time
from training import train
import ConvNet

# Initialize CUDA
torch.cuda.init()
torch.cuda.set_device(1)

# Whether or not to load a model
load_model = False

# The run number to load. This code grabs the number of the last run assuming none were deleted
run_number = len(os.listdir("./models/")) if not load_model else len(os.listdir("./models/"))

# The path of the model directory
model_dir = "./models/run_"+str(run_number)

# If the path does not exist, create it
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

# The format string to use to determine the path to each model file
model_save_format = model_dir + "/epoch_%d-loss_%f-acc_%f.pt"

# The epoch number to load
load_epoch = 34

# The loss value of the epoch to load
load_loss = 7.374138

# The accuracy value of the epoch to load
load_accuracy = 0.090000

# The path to the model to load
model_path = model_save_format % (load_epoch, load_loss, load_accuracy)

# If the model is being loaded, start from the next epoch after the loaded one
# Otherwise, start from epoch 0
start_epoch = load_epoch + 1 if load_model else 0

# The path of the folder containing the logos
logos_folder = "./data/logos"

# The path of the folder containing the backgrounds
backgrounds_folder = "./data/backgrounds"

# The width and height of the data to use
data_size = 25

# The bounds for the logo transformations
transform_bounds = 25

# Whether or not to convert the images to grayscale
grayscale = False

# Whether or not to place the images on backgrounds
use_background = True

# The size of each epoch
epoch_size = 1000

# The number of epochs to train for
num_epochs = 100

# The initial learning rate.
learning_rate = 0.00001

# Whether or not to use CUDA
use_gpu = True

# The size of each training batch
batch_size = 50

# The number of channels in the input data
in_channels = 1 if grayscale else 3

# The size of the kernels in the initial convolutional layer
conv_kernel_size = 9

# The number of kernels in the initial convolutional layer
conv_out_channels = 256

# Calculate the size of the output of the initial convolutional layer
conv_out_size = data_size - conv_kernel_size + 1
conv_out_shape = (conv_out_size, conv_out_size, conv_out_channels)

# The number of kernels in each Primary Capsule
primary_out_channels = 8
# The size of the kernels in the PrimaryCaps layer
primary_kernel_size = 9
# The stride of the kernels in the PrimaryCaps layer
primary_stride = 2
# The number of Primary Capsules
num_primary_caps = 32

# The number of dimensions to project the pose vectors into
digit_out_channels = 16

# The number of iterations of dynamic routing to use
routing_iterations = 2

# The size of the first decoder layer
decoder_size_1 = 512
# The size of the second decoder layer
decoder_size_2 = 1024

# The weights of the loss value for the non-present classes
absent_loss_weight = 0.5

# The weight of the reconstruction loss
reconstruction_loss_weight = 0.01

# Multiply this by the size of the training set to get the size of the test set
test_size_multiple = 0.1

# The training data generator
train_dataset = DataGen(
    logos_folder,
    backgrounds_folder,
    data_size,
    transform_bounds,
    epoch_size,
    grayscale,
    use_background
)

# The test data generator
test_dataset = DataGen(
    logos_folder,
    backgrounds_folder,
    data_size,
    transform_bounds,
    int(epoch_size  * test_size_multiple),
    grayscale,
    use_background
)

# The number of categories
num_categories = train_dataset.num_categories

# Wrap the data generators into DataLoader objects
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
)


# Whether or not to train the capsule network
train_capsnet = True

# Only train one at a time
train_convnet = not train_capsnet

# Either load a model or make a new one
if load_model:
    model = torch.load(model_path)
else:
    if train_capsnet:
        model = LogoCapsNet(
            num_categories,
            data_size,
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
        )

    elif train_convnet:
        model = ConvNet.LogoConvNet(
            num_categories,
            data_size,
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
        )
    else:
        # If not training either, quit
        quit()

# Print the model architecture to the console
print("Architecture")
print(model)

# List out how many parameters each part of the model has
print("\nSizes of parameters: ")
for name, param in model.named_parameters():
    totalparams = 1
    for i in list(param.size()):
        totalparams *= i
    print("{}: {} {:,}".format(name, list(param.size()), totalparams))

# Print the total number of parameters to console
n_params = sum([p.nelement() for p in model.parameters()])
print('\nTotal number of parameters: {:,} \n'.format(n_params))

# If enabled and possible, load the model onto the gpu
if torch.cuda.is_available() and use_gpu:
    print('Loading onto GPU')
    model.cuda()
    print('Loaded onto GPU')

# Initialize the log writer
writer = SummaryWriter()
print('Starting Training')
start = time.clock()
# Train the model
if train_capsnet:
    train(
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
        model_save_format)
elif train_convnet:
    ConvNet.train(
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
        model_save_format)

writer.close()
print('Training took %d seconds'%(time.clock()-start))