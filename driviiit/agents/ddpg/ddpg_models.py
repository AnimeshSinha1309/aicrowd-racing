import torch, torch.nn.functional

from driviiit.interface.config import IMAGE_SHAPE


class ActorNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=3, out_channels=10, kernel_size=(3, 3), padding=1
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=10, out_channels=10, kernel_size=(3, 3), padding=1
        )
        self.pool1 = torch.nn.MaxPool2d((2, 2))
        self.conv3 = torch.nn.Conv2d(
            in_channels=10, out_channels=10, kernel_size=(3, 3), padding=1
        )
        self.conv4 = torch.nn.Conv2d(
            in_channels=10, out_channels=10, kernel_size=(3, 3), padding=1
        )
        self.pool2 = torch.nn.MaxPool2d((2, 2))
        self.conv5 = torch.nn.Conv2d(
            in_channels=10, out_channels=10, kernel_size=(3, 3), padding=1
        )
        self.conv6 = torch.nn.Conv2d(
            in_channels=10, out_channels=10, kernel_size=(3, 3), padding=1
        )
        self.pool3 = torch.nn.MaxPool2d((2, 2))
        self.fc1 = torch.nn.Linear(
            10 * (IMAGE_SHAPE[0] // 8) * (IMAGE_SHAPE[1] // 8), 120
        )
        self.fc2 = torch.nn.Linear(120, 84)
        self.acceleration_head = torch.nn.Linear(84, 1)
        self.steering_head = torch.nn.Linear(84, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool1(x)
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.pool2(x)
        x = torch.relu(self.conv5(x))
        x = torch.relu(self.conv6(x))
        x = self.pool3(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        acceleration = torch.tanh(self.acceleration_head(x))
        steering = torch.tanh(self.steering_head(x))
        return torch.concat([acceleration, steering], dim=1)
