import syft as sy
import torch
import torchvision
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import cifar10_dataloader
from datetime import datetime
from torch.autograd import Variable
import torchvision as tv            #里面含有许多数据集
import torch
import torchvision.transforms as transforms    #实现图片变换处理的包
from torchvision.transforms import ToPILImage

hook = sy.TorchHook(torch)
logger = logging.getLogger(__name__)
date = datetime.now().strftime('%Y-%m-%d %H:%M')

class Argument():
    def __init__(self):
        self.user_num = 3000    # number of total clients P
        self.K = 20  # number of participant clients K
        self.lr = 0.0005      # learning rate of global model
        self.alpha = 0.1     # parameter for momentum
        self.batch_size = 8       # batch size of each client for local training
        self.test_batch_size = 128     # batch size for test datasets
        self.total_iterations = 10000   # total number of iterations
        self.classNum = 2     # number of data classes on each client, which can determine the level of non-IID data
        self.itr_test = 100     # number of iterations for the two tests on test datasets
        self.itr_train = 100    # number of iterations for the two tests on training datasets
        self.seed = 1    # parameter for the server to initialize the model
        self.cuda_use = False

args = Argument()
use_cuda = args.cuda_use and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")

#定义网络
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(x.size()[0],-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

##################################获取模型层数和各层的形状#############
def GetModelLayers(model):
    Layers_shape = []
    Layers_nodes = []
    for idx, params in enumerate(model.parameters()):
        Layers_num = idx
        Layers_shape.append(params.shape)
        Layers_nodes.append(params.numel())
    return Layers_num + 1, Layers_shape, Layers_nodes

##################################设置各层的梯度为0#####################
def ZerosGradients(Layers_shape):
    ZeroGradient = []
    for i in range(len(Layers_shape)):
        ZeroGradient.append(torch.zeros(Layers_shape[i]))
    return ZeroGradient


#################################计算范数################################
def L_norm(Tensor):
    norm_Tensor = torch.tensor([0.])
    for i in range(len(Tensor)):
        norm_Tensor += Tensor[i].float().norm()**2
    return norm_Tensor.sqrt()

############################定义测试函数################################
def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    with torch.no_grad():
        for imagedata, labels in test_loader:
            outputs = model(Variable(imagedata))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, labels)
            test_loss += loss.item()
    test_acc = 100. * correct / total
    print('10000张测试集: testacc is {:.4f}%, testloss is {}.'.format(test_acc, test_loss))
    return test_loss, test_acc

##########################定义训练过程，返回梯度########################
def train(learning_rate, model, train_data, train_targets, device, optimizer):
    model.train()
    model.zero_grad()
    train_targets = Variable(train_targets.long())
    optimizer.zero_grad()
    outputs = model(train_data)
    # 计算准确率
    #_, predicted = torch.max(outputs.data, 1)
    #total = labels.size(0)
    #correct = (predicted == labels).sum()
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, train_targets)
    loss.backward()

    Gradients_Tensor = []
    for params in model.parameters():
        Gradients_Tensor.append(params.grad.data)    # 把各层的梯度添加到张量Gradients_Tensor
    return Gradients_Tensor, loss

##################################计算Non-IID程度##################################
def JaDis(datasNum, userNum):
    sim = []
    for i in range(userNum):
        data1 = datasNum[i]
        for j in range(i+1, userNum):
            data2 = datasNum[j]
            sameNum = [min(x, y) for x, y in zip(data1, data2)]
            sim.append(sum(sameNum) / (sum(data1) + sum(data2) - sum(sameNum)))
    distance = 2*sum(sim)/(userNum*(userNum-1))
    return distance

###################################################################################
##################################模型和用户生成###################################
model = Net()
workers = []
models = {}
optims = {}
taus = {}
for i in range(1, args.user_num+1):
    exec('user{} = sy.VirtualWorker(hook, id="user{}")'.format(i,i))
    exec('models["user{}"] = model.copy()'.format(i))
    exec('optims["user{}"] = optim.SGD(params=models["user{}"].parameters(), lr=args.lr)'.format(i,i))
    exec('workers.append(user{})'.format(i))    # 列表形式存储用户
    exec('taus["user{}"] = {}'.format(i, 1))      # 列表形式存储用户
    # exec('workers["user{}"] = user{}'.format(i,i))    # 字典形式存储用户
optim_sever = optim.SGD(params=model.parameters(), lr=args.lr)    # 定义服务器优化器
###################################################################################
###############################数据载入############################################
# 使用torchvision加载并预处理CIFAR10数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))])
#把数据变为tensor并且归一化range [0, 255] -> [0.0,1.0]
trainset = tv.datasets.CIFAR10(root='data2/', train=True, download=False, transform=transform)
federated_data, dataNum = cifar10_dataloader.dataset_federate_noniid(trainset, workers, transform, args.classNum)
#Jaccard = JaDis(dataNum, args.user_num)
#print('Jaccard distance is {}'.format(Jaccard))

testset = tv.datasets.CIFAR10('data2/', train=False, download=False)
testset = cifar10_dataloader.testLoader(testset)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True, drop_last=True, num_workers=0)

# 定义记录字典
logs = {'train_loss': [], 'test_loss': [], 'test_acc': []}
test_loss, test_acc = test(model, test_loader, device)   # 初始模型的预测精度
logs['test_acc'].append(test_acc.item())
logs['test_loss'].append(test_loss)

###################################################################################
#################################联邦学习过程######################################
#获取模型层数和各层形状
Layers_num, Layers_shape, Layers_nodes = GetModelLayers(model)
e = torch.exp(torch.tensor(1.))

#定义训练/测试过程
worker_have_list = []

Gradients_pa = ZerosGradients(Layers_shape)
Gradients_Momentum = ZerosGradients(Layers_shape)
Gradients_unbiased = ZerosGradients(Layers_shape)

for itr in range(1, args.total_iterations + 1):
    #按设定的每回合用户数量和每个用户的批数量载入数据，单个批的大小为batch_size
    #为了对每个样本上的梯度进行裁剪，令batch_size=1，batch_num=args.batch_size*args.batchs_round，将样本逐个计算梯度
    federated_train_loader = sy.FederatedDataLoader(federated_data, worker_have_list=worker_have_list,
                                                    worker_one_round=True, batch_size=args.batch_size, shuffle=True,
                                                    worker_num=args.K, batch_num=1)
    workers_list = federated_train_loader.workers    # 当前回合抽取的用户列表
    worker_have_list = federated_train_loader.worker_have_list
    if len(worker_have_list) == args.user_num:
        worker_have_list = []
        for i in range(Layers_num):
            Gradients_Momentum[i] = Gradients_pa[i]
        Gradients_pa = ZerosGradients(Layers_shape)

    # 生成与模型梯度结构相同的元素=0的列表
    Loss_train = torch.tensor(0.)
    Collect_Gradients = ZerosGradients(Layers_shape)
    for idx_outer, (train_data, train_targets) in enumerate(federated_train_loader):
        model_round = models[train_data.location.id]
        optimizer = optims[train_data.location.id]
        train_data, train_targets = train_data.to(device), train_targets.to(device)
        train_data, train_targets = train_data.get(), train_targets.get()
        # optimizer = optims[data.location.id]
        # 返回梯度张量，列表形式；同时返回loss；gradient=False，则返回-lr*grad
        Gradients_Sample, loss = train(args.lr, model_round, train_data, train_targets, device, optimizer)
        Loss_train += loss
        for i in range(Layers_num):
            Collect_Gradients[i] += Gradients_Sample[i] / args.K

    for i in range(Layers_num):
        Gradients_unbiased[i] = args.alpha * Gradients_Momentum[i] + Collect_Gradients[i]
        Gradients_pa[i] = Gradients_pa[i] + Gradients_unbiased[i] / args.user_num

    for grad_idx, params_sever in enumerate(model.parameters()):
        params_sever.data.add_(-args.lr, Collect_Gradients[grad_idx])
    # 平均训练损失
    Loss_train /= (idx_outer + 1)

    #同步更新不需要下面代码；异步更新需要下段代码
    for worker_idx in range(len(workers_list)):
        worker_model = models[workers_list[worker_idx]]
        for idx, (params_server, params_client) in enumerate(zip(model.parameters(),worker_model.parameters())):
            params_client.data = params_server.data
        models[workers_list[worker_idx]] = worker_model###添加把更新后的模型返回给用户

    if itr == 1 or itr % args.itr_test == 0:
        print('itr: {}'.format(itr))
        test_loss, test_acc = test(model, test_loader, device)  # 初始模型的预测精度
        logs['test_acc'].append(test_acc.item())
        logs['test_loss'].append(test_loss)
        logs['train_loss'].append(Loss_train)

with open('./results/cifar10_GSGM_testacc.txt', 'a+') as fl:
    fl.write('\n' + date + ' Results (UN is {}, K is {}, BZ is {}, LR is {}, total itr is {}, itr_test is {}, classNum is {})\n'.
             format(args.user_num, args.K, args.batch_size, args.lr, args.total_iterations, args.itr_test, args.classNum))
    fl.write('GSGM: ' + str(logs['test_acc']))

with open('./results/cifar10_GSGM_testloss.txt', 'a+') as fl:
    fl.write('\n' + date + ' Results (UN is {}, K is {}, BZ is {}, LR is {}, total itr is {}, itr_test is {}, classNum is {})\n'.
             format(args.user_num, args.K, args.batch_size, args.lr, args.total_iterations, args.itr_test, args.classNum))
    fl.write('testloss: ' + str(logs['test_loss']))

with open('./results/cifar10_GSGM_trainloss.txt', 'a+') as fl:
    fl.write('\n' + date + ' Results (UN is {}, K is {}, BZ is {}, LR is {}, total itr is {}, itr_test is {}, classNum is {})\n'.
             format(args.user_num, args.K, args.batch_size, args.lr, args.total_iterations, args.itr_test, args.classNum))
    fl.write('trainloss: ' + str(logs['train_loss']))




