# 预测使用
import torchvision
from PIL import Image
import torch
import torch.nn.functional as F


def preictImg(img_path, model):
    image = Image.open(img_path)
    image = image.convert('1')
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize((28, 28)),
                                                torchvision.transforms.ToTensor()])

    image = transform(image).to(device)
    image = torch.unsqueeze(image, dim=0)
    print(image.shape)

    out1 = net(image)
    out1 = F.softmax(out1, dim=1)
    proba, class_ind = torch.max(out1, 1)

    proba = float(proba)
    class_ind = int(class_ind)
    print("预测的类别为： %s .  概率为： %3f" % (classes[class_ind], proba))


if __name__ == '__main__':
    # 训练的时候类别是怎么放的，这里的classes也要对应写
    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_path = "./data/prediction/7.png"
    model_path = "mnist_cnn.pth"
    net = torch.load(model_path, map_location=device)
    preictImg(img_path, net)


