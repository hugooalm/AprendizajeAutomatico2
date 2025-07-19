
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        F1 = nn.Linear(input_size, hidden_size)     # Capa oculta
        Act = nn.Sigmoid()                          # Capa de activacion
        F_out = nn.Linear(hidden_size, output_size) # Capa de salida
        self.layers = nn.Sequential(F1, Act, F_out)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layers(x)

class CNN(nn.Module):
    def __init__(self, ouput_dim):
        super(CNN, self).__init__()


        # Primer bloque convolucional
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # Reduce la dimensión espacial a la mitad
        )


        # Segundo bloque convolucional
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # Reduce la dimensión espacial a la mitad
        )

        # Tercer bloque convolucional
        self.block3 = nn.Sequential(
           nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # Reduce la dimensión espacial a la mitad
        )

        # Capa de aplanado
        self.flatten = nn.Flatten()

        # Linear para la clasificación
        self.linear = nn.Linear(576, ouput_dim)

    def forward(self, x):

        # Primer bloque convolucional
        x = self.block1(x)

        # Segundo bloque convolucional
        x = self.block2(x)

        # Tercer bloque convolucional
        x = self.block3(x)

        # Aplanar la salida de la última capa convolucional
        x = self.flatten(x)

        # Linear para clasificación
        x = self.linear(x)

        return x


class FCNN(nn.Module):
    def __init__(self, output_dim):
        super(FCNN, self).__init__()

        # Primer bloque convolucional
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Segundo bloque convolucional
        self.block2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Tercer bloque convolucional
        self.block3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Cuarto bloque convolucional
        self.block4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Bloque de salida
        self.output_block = nn.Conv2d(64, output_dim, kernel_size=1)


    def forward(self, x):
        # Primer bloque convolucional
        x = self.block1(x)

        # Segundo bloque convolucional
        x = self.block2(x)

        # Tercer bloque convolucional
        x = self.block3(x)

        # Cuarto bloque convolucional
        x = self.block4(x)

        # Capa de salida
        x = self.output_block(x)   # Salida (N, num_classes, 1, 1)
        x = x.view(x.size(0), -1)   # (N, num_classes)
        return x

class MyBestModel(nn.Module):
    def __init__(self, output_dim):
        super(MyBestModel, self).__init__()

        # Primer bloque convolucional
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=3, padding=1),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Segundo bloque convolucional
        self.block2 = nn.Sequential(
            nn.Conv2d(20, 45, kernel_size=3, padding=1),
            nn.BatchNorm2d(45),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Tercer bloque convolucional
        self.block3 = nn.Sequential(
            nn.Conv2d(45, 90, kernel_size=3, padding=1),
            nn.BatchNorm2d(90),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Cuarto bloque convolucional
        self.block4 = nn.Sequential(
            nn.Conv2d(90, 180, kernel_size=3, padding=1),
            nn.BatchNorm2d(180),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )


        # Bloque de salida
        self.output_block = nn.Conv2d(180, output_dim, kernel_size=1)


    def forward(self, x):
        # Primer bloque convolucional
        x = self.block1(x)

        # Segundo bloque convolucional
        x = self.block2(x)

        # Tercer bloque convolucional
        x = self.block3(x)

        # Cuarto bloque convolucional
        x = self.block4(x)

        # Capa de salida
        x = self.output_block(x)   # Salida (N, num_classes, 1, 1)
        x = x.view(x.size(0), -1)   # (N, num_classes)
        return x
