from attacks import Attacks
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

torch.manual_seed(1234)

def is_adversarial(model,X,Y):
    labels = model(X).argmax(1)
    p = model(Y).argmax(1)
    return (p != labels)

def get_adversarial(model, X,labels, max_iter = 10000):
    adv = torch.zeros_like(X)
    adv2 = torch.zeros_like(X)
    adv2 = adv2 + X
    p = model(adv2).argmax(1)
    n = 0
    while torch.any(p == labels) and n < max_iter:
        n+=1
        noise = 2 * torch.rand_like(X) -1
        adv2 = (adv + noise)
        p = model(adv2).argmax(1)

    return adv2


def find_boundary(model, X, labels, eps = 1e-4, max_iter = 100000):
    best_adv = []
    for i in range(len(X)):
        starting_point = get_adversarial(model,X[i].unsqueeze(0),labels[i])
        upper_bound = starting_point
        lower_bound = X[i].unsqueeze(0)
        n = 0
        while (torch.norm(lower_bound-upper_bound) > eps and n < max_iter):
            mid_point = torch.zeros_like(X[i].unsqueeze(0))
            mid_point = (upper_bound + lower_bound)/2

            if is_adversarial(model,X[i].unsqueeze(0), mid_point):
                upper_bound = mid_point
            else :
                lower_bound = mid_point

        if not is_adversarial(model,X[i].unsqueeze(0), mid_point):
            mid_point = upper_bound

        best_adv.append(mid_point[0])
    best_adv = torch.stack(best_adv)

    return best_adv

def distance(a, b):
    return (a - b).flatten(1).norm(dim=1)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

model = torch.load('MNIST_gpu')

images = []
labels = []
for i in range(100):
    image, label = test_dataset[i]
    images.append(image)
    labels.append(label)

X = torch.stack(images)
labels = torch.tensor(labels)


starting_points = find_boundary(model,X,labels)
n_steps = 785
max_queries = 5000
batch = 1

for i in range(len(X)//batch):
    print(i, "Initial distortion : ",distance(starting_points[i*batch : (i+1)*batch],X[i*batch : (i+1)*batch]))
    without_normal_vector_attack = Attacks(method='surfree',steps=n_steps, starting_point=starting_points[i*batch : (i+1)*batch],max_queries=max_queries,keep_directions=True, theta_eps=1e-2)
    res1,dist1,quer1,badquer1 = without_normal_vector_attack(model,X[i*batch : (i+1)*batch])


    print(f'The distortion with SurFree is :{distance(X[i*batch : (i+1)*batch], res1)}')


    print(i, "Initial distortion",distance(starting_points[i*batch : (i+1)*batch],X[i*batch : (i+1)*batch]))
    with_estimated_normal_vector_attack = Attacks(method='cgba',steps=n_steps, starting_point=starting_points[i*batch : (i+1)*batch], N=50, max_queries=max_queries, std_dev=1e-2, keep_directions=True)
    res2,dist2, quer2,badquer2 = with_estimated_normal_vector_attack(model,X[i*batch : (i+1)*batch])

    print(f'The distortion with CGBA is :{distance(X[i*batch : (i+1)*batch], res2)}')



    print(i, "Initial distortion",distance(starting_points[i*batch : (i+1)*batch],X[i*batch : (i+1)*batch]))
    with_estimated_normal_vector_attack = Attacks(method='geoda',steps=n_steps, starting_point=starting_points[i*batch : (i+1)*batch], N=50, max_queries=max_queries, std_dev=1e-2, keep_directions=True, theta_eps=1e-2)
    res3,dist3, quer3,badquer3 = with_estimated_normal_vector_attack(model,X[i*batch : (i+1)*batch])

    print(f'The distortion with GeoDA is :{distance(X[i*batch : (i+1)*batch], res3)}')


    
    np.save(f'results_MNIST/results_{i}.npy', {'dist1' : dist1.cpu().numpy(), 'queries1' : quer1.cpu().numpy(), 'badquer1' : badquer1.cpu().numpy(),
                                    'dist2' : dist2.cpu().numpy(), 'queries2' : quer2.cpu().numpy(), 'badquer2' : badquer2.cpu().numpy(),
                                    'dist3' : dist3.cpu().numpy(), 'queries3' : quer3.cpu().numpy(), 'badquer3' : badquer3.cpu().numpy()})
    

