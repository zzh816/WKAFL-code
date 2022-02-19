import torch
import syft as sy  # <-- NEW: import the Pysyft library
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from datetime import datetime
hook = sy.TorchHook(torch)  # <-- NEW: hook PyTorch ie add extra functionalities to support Federated Learning
import Datasets
import os
import logging
logger = logging.getLogger(__name__)
date = datetime.now().strftime('%Y-%m-%d')

#定义参量
class Arguments():
    def __init__(self):
        self.batch_size = 4     # batch size of each client for local training
        self.test_batch_size = 10     # batch size for test datasets
        self.lr = 0.00001      # learning rate of global model
        self.no_cuda = True
        self.seed = 1    # parameter for the server to initialize the model
        self.log_train = 100        # number of iterations for the two neighbour tests on test datasets
        self.log_test = 100       # number of iterations for the two neighbour tests on training datasets
        self.users_total = 2000        # number of total clients P
        self.K = 15        # number of participant clients K
        self.batchs_round = 1  # number of batches used by a client
        self.itr_numbers = 5000        # total number of iterations


args = Arguments()

use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

#定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 4, 1)
        #self.conv2 = nn.Conv2d(10, 20, 4, 1)
        self.fc1 = nn.Linear(40*40*10, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        #x = F.relu(self.conv2(x))
        #x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 40*40*10)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

###################################################################################
###############################函数定义############################################

################################定义虚拟用户数量#######################
def Virtual_Users_num(Leaf_split, LEAF=True):
    Users_num_total = 0
    if LEAF:
        Users_num_total = len(Leaf_split)
        Ratio = Leaf_split
    else:
        Users_num_total = args.users_total
        #生成用户数据切分比例
        Ratio = [random.randint(1, 10) for _ in range(Users_num_total)]
    return Users_num_total, Ratio

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

##########################定义训练过程，返回梯度########################
def train(learning_rate, model, train_data, train_target, device, optimizer, gradient=True):
    model.train()
    model.zero_grad()
    output = model(train_data.float())
    loss = F.nll_loss(output, train_target.long())
    loss.backward()
    Gradients_Tensor = []
    if gradient == False:
        for params in model.parameters():
            Gradients_Tensor.append(-learning_rate*params.grad.data)#返回-lr*grad
    if gradient == True:
        for params in model.parameters():
            Gradients_Tensor.append(params.grad.data)#把各层的梯度添加到张量Gradients_Tensor
    return Gradients_Tensor, loss

############################定义测试函数################################
def test(model, device, test_loader):
    model.eval()
    test_loader_len = len(test_loader.dataset)
    test_loss = 0
    correct = 0
    test_acc = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = torch.squeeze(data)
            data, target = data.to(device), target.to(device)
            output = model(data.float())
            test_loss +=  F.nll_loss(output, target.long(), reduction='sum').item() # sum up batch loss
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= test_loader_len
    test_acc = correct / test_loader_len

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, test_loader_len,
        100. * test_acc))
    return test_loss, test_acc

###################################################################################
###############################数据载入############################################
#载入训练集和测试集
data_root_train = "../data/CELEBA/raw_data/train_data.json"
data_root_test = "../data/CELEBA/raw_data/test_data.json"
IMAGES_DIR = "../data/CELEBA/raw_data/img_align_celeba/"
#训练集
train_loader = Datasets.celeba(data_root_train, IMAGES_DIR, args.users_total)#训练集载入
Leaf_split = train_loader.num_samples#LEAF提供的用户数量和训练集上的数据切分
#测试集
test_loader = torch.utils.data.DataLoader(Datasets.celeba(data_root_test, IMAGES_DIR, args.users_total),
    batch_size=args.test_batch_size, shuffle=True, num_workers=0, **kwargs)

###################################################################################
##################################模型和用户生成###################################
model = Net().to(device)
workers = []
models = {}
optims = {}
taus = {}
Users_num_total, Ratio = Virtual_Users_num(Leaf_split, LEAF=False)
for i in range(1, Users_num_total+1):
    exec('user{} = sy.VirtualWorker(hook, id="user{}")'.format(i,i))
    exec('models["user{}"] = model.copy()'.format(i))
    exec('optims["user{}"] = optim.SGD(params=models["user{}"].parameters(), lr=args.lr)'.format(i,i))
    exec('workers.append(user{})'.format(i))   # 列表形式存储用户
    exec('taus["user{}"] = {}'.format(i, 1))  # 列表形式存储用户
    # exec('workers["user{}"] = user{}'.format(i,i))#字典形式存储用户

optim_sever = optim.SGD(params=model.parameters(), lr=args.lr)#定义服务器优化器
print('Total users number is {}, the minimal number of per user is {}'.format(Users_num_total, min(Leaf_split)))

###################################################################################
########################生成文件用于记录实验结果###################################
test_loss, test_acc = test(model, device, test_loader) # 初始模型的预测精度

###################################################################################
###############################联邦数据集生成######################################
Federate_Dataset = Datasets.dataset_federate_noniid(train_loader, workers, Ratio=Ratio)
# Criteria = nn.CrossEntropyLoss()

###################################################################################
#################################联邦学习过程######################################
#定义记录字典
logs = {'train_loss': [], 'test_loss': [], 'test_acc': []}
logs['test_loss'].append(test_loss)
logs['test_acc'].append(test_acc)

#获取模型层数和各层形状
Layers_num, Layers_shape, Layers_nodes = GetModelLayers(model)
#设置学习率
lr = args.lr
#定义训练/测试过程
for itr in range(1, args.itr_numbers + 1):
    #按设定的每回合用户数量和每个用户的批数量载入数据，单个批的大小为batch_size
    federated_train_loader = sy.FederatedDataLoader(Federate_Dataset, batch_size=args.batch_size, shuffle=True,
                                                    worker_num=args.K, batch_num=args.batchs_round, **kwargs)
    workers_list = federated_train_loader.workers#当前回合抽取的用户列表

    # 生成与模型梯度结构相同的元素=0的列表
    Collect_Gradients = ZerosGradients(Layers_shape)
    Loss_train = torch.tensor(0.)
    K_Gradients = []
    K_tau = []
    for idx_outer, (train_data, train_targets) in enumerate(federated_train_loader):
        model_round = models[train_data.location.id]
        optimizer = optims[train_data.location.id]
        K_tau.append(taus[train_data.location.id])
        train_data, train_targets = train_data.to(device), train_targets.to(device)
        train_data, train_targets = train_data.get(), train_targets.get()
        # optimizer = optims[data.location.id]
        # 返回梯度张量，列表形式；同时返回loss；gradient=False，则返回-lr*grad
        Gradients_Tensor, loss = train(lr, model_round, train_data, train_targets, device, optimizer, gradient=True)
        Loss_train += loss
        K_Gradients.append(Gradients_Tensor)

    Collect_Gradients = ZerosGradients(Layers_shape)
    K_tau = torch.tensor(K_tau) * 1.

    for i in range(args.K):
        Gradients_Sample = K_Gradients[i]
        for j in range(Layers_num):
            Collect_Gradients[j] += Gradients_Sample[j] * args.lr / K_tau[i]

    # 升级延时信息
    for tau in taus:
        taus[tau] = taus[tau] + 1
    for worker in workers_list:
        taus[worker] = 1


    #平均训练损失
    Loss_train /= (idx_outer + 1)
    #利用平均化梯度更新服务器模型并且把更新后的模型发送给对应学习者
    for idx_para, params_sever in enumerate(model.parameters()):
        params_sever.data.add_(-1., Collect_Gradients[idx_para])

    #同步更新不需要下面代码；异步更新需要下面代码
    for worker_idx in range(len(workers_list)):
        worker_model = models[workers_list[worker_idx]]
        for _, (params_server, params_client) in enumerate(zip(model.parameters(),worker_model.parameters())):
            params_client.data = params_server.data
        models[workers_list[worker_idx]] = worker_model###添加把更新后的模型返回给用户

    ############间隔给定迭代次数打印损失#################
    if itr == 1 or itr % args.log_train == 0:
        print('Train Iteration: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            itr, itr, args.itr_numbers,
            100. * itr / args.itr_numbers, Loss_train.item()))
        logs['train_loss'].append(Loss_train.item())

    #############间隔给定迭代次数打印预测精度############
    if itr % args.log_test == 0:
        test_loss, test_acc = test(model, device, test_loader)
        logs['test_loss'].append(test_loss)
        logs['test_acc'].append(test_acc)


with open('./results/CELEBA_Asyn_testloss.txt', 'a+') as fl:
    fl.write('\n' + date + ' Results (UN is {}, K is {}, BZ is {}, LR is {}, total itr is {}, itr_test is {})\n'.
             format(args.users_total, args.K, args.batch_size, args.lr, args.itr_numbers, args.log_test))
    fl.write(str(logs['test_loss']) + '\t')
with open('./results/CELEBA_Asyn_testacc.txt', 'a+') as fl:
    fl.write('\n' + date + ' Results (UN is {}, K is {}, BZ is {}, LR is {}, total itr is {}, itr_test is {})\n'.
             format(args.users_total, args.K, args.batch_size, args.lr, args.itr_numbers, args.log_test))
    fl.write(str(logs['test_acc']) + '\n')
    fl.write(str(test_acc))
with open('./results/CELEBA_Asyn_trainloss.txt', 'a+') as fl:
    fl.write('\n' + date + ' Results (UN is {}, K is {}, BZ is {}, LR is {}, total itr is {}, itr_test is {})\n'.
             format(args.users_total, args.K, args.batch_size, args.lr, args.itr_numbers, args.log_test))
    fl.write(str(logs['train_loss']) + '\t')


