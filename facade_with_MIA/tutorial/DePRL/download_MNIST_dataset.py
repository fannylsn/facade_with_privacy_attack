import torchvision

if __name__ == "__main__":
    torchvision.datasets.MNIST(root="./eval/data/", train=True, download=True)
    torchvision.datasets.MNIST(root="./eval/data/", train=False, download=True)
    print("Downloaded the MNIST dataset.")
