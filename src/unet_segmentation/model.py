import torch
import torch.nn as nn
import torchvision.transforms.functional as TF



class DoubleConv(nn.Module):
    # Constructor (initializer) for the DoubleConv class
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()  # Initialize the parent class (nn.Module)
        # Define a sequence of convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, 3, 1, 1, bias=False
            ),  # 3x3 convolution, stride, padding, bias=false since we use batchnorm since bias would be cancelled by batchnorm
            nn.BatchNorm2d(out_channels),  # Batch normalization for regularization
            nn.ReLU(inplace=True),  # ReLU activation function
            nn.Conv2d(
                out_channels, out_channels, 3, 1, 1, bias=False
            ),  # Another 3x3 convolution
            nn.BatchNorm2d(out_channels),  # Batch normalization
            nn.ReLU(inplace=True),  # ReLU activation
        )

    # Forward method for the DoubleConv class
    def forward(self, x):
        return self.conv(x)  # Apply the sequence of convolutional layers to input x


class UNET(nn.Module):
    # Constructor (initializer) for the U-Net class
    def __init__(
        self,
        in_channels,
        out_channels,
        features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()  # Initialize the parent class (nn.Module)

        # Create lists to store the upsampling and downsampling layers
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        # Initialize a max-pooling layer for downsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        # Iterate through the list of features
        for feature in features:
            # Create a DoubleConv layer and add it to the downsampling list
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature  # Update in_channels for the next iteration

        # Up part of UNET
        # Iterate through the reversed list of features
        for feature in reversed(features):
            # Add an upsampling layer to the upsampling list
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            # Add a DoubleConv layer to the upsampling list
            self.ups.append(DoubleConv(feature * 2, feature))

        # Create a DoubleConv layer for the bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Create a final convolutional layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []  # Initialize a list to store skip connections
        # Loop through downsampling layers (contracting path)
        for down in self.downs:
            x = down(x)  # Pass data through the downsampling layers
            skip_connections.append(x)  # Store the current feature maps
            x = self.pool(x)  # Perform max pooling for downsampling
        x = self.bottleneck(x)  # Pass data through the bottleneck layer
        skip_connections = skip_connections[::-1]  # Reverse the skip connections list
        # Loop through upsampling layers (expanding path)
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # Pass data through the upsampling layer
            skip_connections_layer = skip_connections[
                idx // 2
            ]  # Retrieve skip connection from the list
            if x.shape != skip_connections_layer.shape:
                x = TF.resize(x, size=skip_connections_layer.shape[2:])

            concat_skip = torch.cat(
                (skip_connections_layer, x), dim=1
            )  # Concatenate skip and current features
            x = self.ups[idx + 1](
                concat_skip
            )  # Pass concatenated features through another upsampling layer
        return self.final_conv(x)  # Pass data through the final convolutional layer


def test():
    x = torch.randn((3, 1, 162, 162))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    assert preds.shape == x.shape


if __name__ == "__main__":
    test()