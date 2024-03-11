from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

class CNN(nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size = 5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size = 5) 
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training = self.training)
        x = self.fc2(x)

        return F.softmax(x)

train_data = datasets.MNIST(
    root = 'data',
    train = True,
    transform = ToTensor(),
    download = True
)

test_data = datasets.MNIST(
    root = 'data',
    train = False,
    transform = ToTensor(),
    download = True
)

loaders = {
    'train': DataLoader(train_data,
                        batch_size = 100,
                        shuffle = True,
                        num_workers=0),
    
    'test': DataLoader(test_data,
                        batch_size = 100,
                        shuffle = True,
                        num_workers=0),
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CNN().to(device)

optimizer = optim.Adam(model.parameters(), lr = 0.001)

loss_fn = nn.CrossEntropyLoss()

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(loaders['train']):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print(f'Train Epoc: {epoch} [{batch_idx * len(data)}/{len(loaders["train"].dataset)} ({100. * batch_idx / len(loaders["train"]):.0f}%)]\t{loss.item():.6f}')

def test():
    model.eval()
    
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in loaders['test']:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim = 1, keepdim = True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(loaders['test'].dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy {correct}/{len(loaders["test"].dataset)} ({100. * correct / len(loaders["test"].dataset):.0f}%\n')

def predict(c):
    model.eval()

    data, target = test_data[c]

    data = data.unsqueeze(0).to(device)

    output = model(data)

    prediction = output.argmax(dim = 1, keepdim = True).item()

    print(f'Prediction {prediction}')

    image = data.squeeze(0).squeeze(0).cpu().numpy()

    plt.imshow(image, cmap="gray")
    plt.show()

for epoch in range (1,6):
    train(epoch)
    test()
    
for i in range (1,11):
    predict(i)

def preprocess_image(image_path):
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to match MNIST image size
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # Normalize like MNIST images
        ]
    )
    return transform(image)


def predict(image_paths):
    model.eval()
    for image_path in image_paths:
        image_tensor = preprocess_image(image_path)
        image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension
        output = model(image_tensor)
        prediction = output.argmax(dim=1, keepdim=True).item()
        print(f"Prediction for {image_path}: {prediction}")
        image = image_tensor.squeeze(0).squeeze(0).cpu().numpy()
        plt.imshow(image, cmap="gray")
        plt.show()

def predict_single_image(image_path):    
    model.eval()
        
    image_tensor = preprocess_image(image_path)
        
    output = model(image_tensor)
        
    prediction = output.argmax(dim=1, keepdim=True).item()
        
    print(f"Prediction for '{image_path}': {prediction}")
    
    image = image_tensor.squeeze(0).squeeze(0).cpu().numpy()
    plt.imshow(image, cmap="gray")
    plt.show()
    
    return prediction

for epoch in range(1, 3):
    train(epoch)
    test()

image_paths = ["image7.png", "image3.png"]
predict(image_paths)