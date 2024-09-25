import torch
import numpy as np
from torchvision import transforms
import os
import torchvision
import requests
from torchvision.io import read_image
from attacks import Attacks


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1234)

def get_model():
    model = torchvision.models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1').eval()
    mean = torch.Tensor([0.485, 0.456, 0.406])
    std = torch.Tensor([0.229, 0.224, 0.225])
    normalizer = torchvision.transforms.Normalize(mean=mean, std=std)
    return torch.nn.Sequential(normalizer, model).eval()

def get_imagenet_labels():
    response = requests.get("https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json")
    return eval(response.content)


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
        noise = 2 * torch.rand_like(X, ) -1
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

print("Load Model")
model = get_model()

print("Get understandable ImageNet labels")
imagenet_labels = get_imagenet_labels()

print("Load Data")
X = []
transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224)])
for img in os.listdir("./images"):
    X.append(transform(read_image(os.path.join("./images", img))).unsqueeze(0))
X = torch.cat(X, 0) / 255
y = model(X).argmax(1)


if torch.cuda.is_available():
    model = model.to(device)
    X = X.to(device)
    y = y.to(device)

print(X.shape)
starting_points = find_boundary(model,X,y)
dist = distance(X, starting_points)
np.save(f'starting_points.npy', {'dist' : dist.cpu().numpy(), 'queries' : torch.zeros_like(dist).cpu().numpy()})
n_steps = 1000
frequencies_to_remove = 32
max_queries = 5000
batch = 1
std_dev = 0.01

for i in range(len(X)//batch):
    print(i, "Initial distortion : ",distance(starting_points[i*batch : (i+1)*batch],X[i*batch : (i+1)*batch]))
    without_normal_vector_attack = Attacks(method ='surfree', steps=n_steps, starting_point=starting_points[i*batch : (i+1)*batch], with_dct=True, frequency_to_remove=frequencies_to_remove, max_queries=max_queries,std_dev=std_dev,theta_eps=1e-2)
    res1,dist1,quer1,badquer1 = without_normal_vector_attack(model,X[i*batch : (i+1)*batch])


    print(f'La distance pour SurFree est :{distance(X[i*batch : (i+1)*batch], res1)}')

    print(i, "Initial distortion",distance(starting_points[i*batch : (i+1)*batch],X[i*batch : (i+1)*batch]))
    with_estimated_normal_vector_attack = Attacks(method ='cgba',steps=n_steps, starting_point=starting_points[i*batch : (i+1)*batch], with_dct=True, frequency_to_remove=frequencies_to_remove, N=50, max_queries=max_queries, std_dev=1e-2, keep_directions=True, theta_eps=1e-2)
    res2,dist2, quer2,badquer2 = with_estimated_normal_vector_attack(model,X[i*batch : (i+1)*batch])

    print(f'The distortion with CGBA is :{distance(X[i*batch : (i+1)*batch], res3)}')

    print("Initial distortion",distance(starting_points[i*batch : (i+1)*batch],X[i*batch : (i+1)*batch]))
    with_estimated_normal_vector_attack = Attacks(method='geoda',steps=n_steps, starting_point=starting_points[i*batch : (i+1)*batch], with_dct=True, frequency_to_remove=frequencies_to_remove, N=50, max_queries=max_queries, std_dev=1e-2, keep_directions=True, theta_eps=1e-2)
    res3,dist3, quer3,badquer3 = with_estimated_normal_vector_attack(model,X[i*batch : (i+1)*batch])

    print(f'The distortion with GeoDA is :{distance(X[i*batch : (i+1)*batch], res3)}')
    
    np.save(f'results_ImageNet/results_{i}.npy', {'dist1' : dist1.cpu().numpy(), 'queries1' : quer1.cpu().numpy(), 'badquer1' : badquer1.cpu().numpy(),
                                    'dist2' : dist2.cpu().numpy(), 'queries2' : quer2.cpu().numpy(), 'badquer2' : badquer2.cpu().numpy(),
                                    'dist3' : dist3.cpu().numpy(), 'queries3' : quer3.cpu().numpy(), 'badquer3' : badquer3.cpu().numpy()})

