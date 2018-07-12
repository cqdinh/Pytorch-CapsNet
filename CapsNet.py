import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

class LogoCapsNet(nn.Module):
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
        super(LogoCapsNet, self).__init__()
        self.in_size = in_size
        self.num_categories = num_categories
        self.use_gpu = use_gpu
        self.conv = nn.Conv2d(in_channels, conv_out_channels, conv_kernel_size)
        self.primarycaps = nn.ModuleList([
            nn.Conv2d(conv_out_channels, primary_out_channels, primary_kernel_size, primary_stride) for i in range(num_primary_caps)
        ])

        self.routing_iterations = routing_iterations
        outsize = ((in_size-(conv_kernel_size-1))-(primary_kernel_size-1)-1)//primary_stride+1
        self.digitcaps_nodes = outsize * outsize * num_primary_caps
        self.digitcaps_weights = nn.Parameter(torch.randn(num_categories, self.digitcaps_nodes, primary_out_channels, digit_out_channels))

        self.decoder = nn.Sequential(
            nn.Linear(digit_out_channels * num_categories, decoder_size_1),
            nn.ReLU(),
            nn.Linear(decoder_size_1, decoder_size_2),
            nn.ReLU(),
            nn.Linear(decoder_size_2, in_size*in_size*in_channels),
            nn.Sigmoid()
        )

        self.absent_loss_weight = absent_loss_weight
        self.reconstruction_loss_weight = reconstruction_loss_weight


    def forward(self, x):
        batch_size = x.size(0)

        # Convolutional Layer
        x = F.leaky_relu(self.conv(x))

        # PrimaryCaps
        x = torch.cat([capsule(x).view(batch_size, 8, -1, 1) for capsule in self.primarycaps], dim=3)
        x = x.view(batch_size, 8, -1)
        x = x.transpose(1, 2)
        x = squash(x)

        # DigitCaps
        x = x.unsqueeze(2).unsqueeze(1)
        x = torch.matmul(x, self.digitcaps_weights).squeeze(3)
        # size: [batch size, num_categories, num_nodes, digit_out_channels]

        logits = Variable(torch.zeros(batch_size, self.num_categories, self.digitcaps_nodes, 1))
        # size: [batch size, num_categories, num_nodes, 1]
        if torch.cuda.is_available() and self.use_gpu:
            logits = logits.cuda()

        for r in range(self.routing_iterations):
            probs = F.softmax(logits, dim=1)
            # size: [batch size, num_categories, num_nodes, 1]

            weighted_vectors = x * probs
            # size: [batch size, num_categories, num_nodes, digit_out_channels]

            outputs = squash(torch.sum(weighted_vectors, dim=2))
            # size: [batch size, num_categories, digit_out_channels]


            delta_logits = torch.sum(x * outputs.unsqueeze(2), dim=3)
            # size: [batch size, num_categories, num_nodes]

            logits = logits + delta_logits.unsqueeze(3)


        probs = F.softmax(logits, dim=1)
        # size: [batch size, num_categories, num_nodes, 1]

        weighted_vectors = x * probs
        # size: [batch size, num_categories, num_nodes, digit_out_channels]
        outputs = squash(torch.sum(weighted_vectors, dim=2))
        # size: [batch size, num_categories, digit_out_channels]
        return outputs

    def loss(self, outputs, y, reconstruction_target):

        # x size: [batch size, in channels, in size, in size]
        # y size: [batch size, num_categories]
        batch_size = y.size(0)

        output_probs = torch.sqrt(torch.sum(outputs * outputs, dim=-1))
        # size: [batch size, num_categories]

        present_loss = F.relu(0.9 - output_probs, inplace=True)
        absent_loss = F.relu(output_probs - 0.1, inplace=True)

        present_loss = present_loss * present_loss
        absent_loss = absent_loss * absent_loss

        margin_loss = y * present_loss + self.absent_loss_weight * (1-y) * absent_loss
        # size: [batch size, num_categories]
        margin_loss = torch.sum(margin_loss)
        # Scalar

        reconstruction = self.reconstruction(outputs, y)
        diff = reconstruction - reconstruction_target
        reconstruction_loss = torch.sum(diff * diff)
        # Scalar

        loss = margin_loss + self.reconstruction_loss_weight * reconstruction_loss

        return loss / batch_size, margin_loss / batch_size, reconstruction_loss / batch_size


    def reconstruction(self, outputs, y):
        batch_size = y.size(0)
        masked_outputs = outputs * y.unsqueeze(2)
        reconstruction = self.decoder(masked_outputs.view(batch_size, -1)).view(batch_size, -1, self.in_size, self.in_size)
        return reconstruction

def squash(x):
    sum_squared = torch.sum(x * x, dim=-1, keepdim=True)
    norm = torch.sqrt(sum_squared)
    factor = norm / (1 + sum_squared)
    return x * factor


