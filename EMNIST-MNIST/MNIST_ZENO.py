import numpy as np
import syft as sy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import rawDatasetsLoader
from datetime import datetime
hook = sy.TorchHook(torch)
logger = logging.getLogger(__name__)
date = datetime.now().strftime('%Y-%m-%d %H:%M')

class Argument():
    def __init__(self):
        self.user_num = 100
        self.K = 1
        self.lr = 0.00005
        self.itr_test = 20
        self.batch_size = 4
        self.test_batch_size = 128
        self.total_iterations = 1500
        self.classes = 1

        self.yunItr = 5
        self.yunBatchSize = 50
        self.yunNum = 100
        self.yunclasses = 10
        self.gamma = 1
        self.rho = 0.2      # sim
        self.epsilon = 0.1
        self.seed = 1
        self.cuda_use = False

args = Argument()
use_cuda = args.cuda_use and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cpu" if args.cuda_use else "cpu")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.fc1 = nn.Linear(4*4*64, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

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

################################ 定义剪裁 #################################
def TensorClip(Tensor, ClipBound):
    norm_Tensor = L_norm(Tensor)
    for i in range(Layers_num):
        Tensor[i] = 1.*Tensor[i]*ClipBound/norm_Tensor
    return Tensor

############################ 定义测试函数 ################################
def test(model, test_loader, device):
    model.eval()
    test_loader_len = len(test_loader.dataset)
    test_loss = 0
    correct = 0
    test_acc = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = torch.squeeze(data)
            data = data.unsqueeze(1)
            data, target = data.to(device), target.to(device)
            output = model(data.float())
            test_loss +=  F.nll_loss(output, target.long(), reduction='sum').item() # sum up batch loss
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= test_loader_len
    test_acc = correct / test_loader_len

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, test_loader_len,
        100. * test_acc))
    return test_loss, test_acc

##########################定义训练过程，返回梯度########################
def train(learning_rate, model, train_data, train_target, device, optimizer, gradient=True):
    model.train()
    model.zero_grad()
    train_data = train_data.unsqueeze(1)
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

################################# 计算角相似度 ############################
def score(user_Gradient, yun_gradient, args):
    sim = torch.tensor([0.])
    for i in range(len(user_Gradient)):
        sim = sim + torch.sum(user_Gradient[i] * yun_gradient[i])
    gradient_score = sim - args.rho * (L_norm(user_Gradient) * L_norm(user_Gradient))
    return gradient_score


##################################云端随机抽取数据##################################
def yunSample(data, args):
    images = data['train_images']
    labels = data['train_labels']
    length = labels.shape[0]
    if args.yunclasses==10:
        index = torch.randperm(length)[0:args.yunNum]
        yunData = images[index, :, :]
        yunLabels = labels[index]
    else:
        yunData = images[0:args.yunNum, :, :]
        yunLabels = labels[0:args.yunNum]
        label_sel = torch.randperm(10)[0:args.yunclasses]
        label_num = int(args.yunNum / args.yunclasses)
        for i in range(args.yunclasses):
            index = (labels==label_sel[i])
            sel_images = images[index, :, :]
            sel_labels = labels[0:args.yunNum]
            sel_length = len(sel_labels)
            sel_index = torch.randperm(sel_length)[0:label_num]
            yunData[label_num*i:label_num*(i+1), :, :] = sel_images[sel_index,:,:]
            yunLabels[label_num*i:label_num*(i+1)] = sel_labels[sel_index]
    return yunData, yunLabels

def yunTrain(args, model, yunData, yunLabels, device, optim_sever):
    data_index = torch.randperm(args.yunNum)[0:args.yunBatchSize]
    data = yunData[data_index,:,:]
    labels = yunLabels[data_index.tolist()]
    yun_gradient, yun_loss = train(args.lr, model, data, labels, device, optim_sever, gradient=True)
    return yun_gradient
###################################################################################
##################################模型和用户生成###################################
model = Net()
yunOptim = optim.SGD(params=model.parameters(), lr=args.lr)
workers = []
models = {}
optims = {}

taus = {}
for i in range(1, args.user_num+1):
    exec('user{} = sy.VirtualWorker(hook, id="user{}")'.format(i,i))
    exec('models["user{}"] = model.copy()'.format(i))
    exec('optims["user{}"] = optim.SGD(params=models["user{}"].parameters(), lr=args.lr)'.format(i,i))
    exec('workers.append(user{})'.format(i))    # 列表形式存储用户
    exec('taus["user{}"] = {}'.format(i,1))      # 列表形式存储用户
    # exec('workers["user{}"] = user{}'.format(i,i))    #字典形式存储用户
optim_sever = optim.SGD(params=model.parameters(), lr=args.lr)    # 定义服务器优化器
###################################################################################
###############################数据载入############################################
dataType = 'mnist'   # 可选bymerge, byclass, digits, mnist, letters, balanced
datasets = rawDatasetsLoader.loadDatesets(trainDataSize=70000, testDataSize=20000, dataType=dataType)
#训练集，测试集, datasNum为列表，datasNum[i]表示第i个学习者的信息，为字典，['3']=45表示图片三有45张
federated_data, datasNum = rawDatasetsLoader.dataset_federate_noniid(datasets, workers, args)
yunData, yunLabels = yunSample(datasets, args)

test_data = rawDatasetsLoader.testImages(datasets)
del datasets
test_loader = torch.utils.data.DataLoader(test_data,
    batch_size=args.test_batch_size, shuffle=True, drop_last=True, num_workers=0)

#定义记录字典
logs = {'totalItr': [], 'test_loss': [], 'test_acc': [], '0.1Num': [], '0.01Num': [], 'truthItr':[]}
test_loss, test_acc = test(model, test_loader, device)  # 初始模型的预测精度
logs['test_acc'].append(test_acc)

###################################################################################
#################################联邦学习过程######################################
#获取模型层数和各层形状
Layers_num, Layers_shape, Layers_nodes = GetModelLayers(model)


yun_gradient = yunTrain(args, model, yunData, yunLabels, device, optim_sever)
# 定义训练/测试过程
successItr = 0
failItr = 0
itr = 0
first = True
while successItr < args.total_iterations+1:
    itr += 1
    # 按概率0.1生成当前回合用户数量
    Users_Current = args.K
    # 按设定的每回合用户数量和每个用户的批数量载入数据，单个批的大小为batch_size
    # 为了对每个样本上的梯度进行裁剪，令batch_size=1，batch_num=args.batch_size*args.batchs_round，将样本逐个计算梯度
    federated_train_loader = sy.FederatedDataLoader(federated_data, batch_size=args.batch_size, shuffle=True,
                                                    worker_num=Users_Current, batch_num=1)
    workers_list = federated_train_loader.workers    # 当前回合抽取的用户列表

    # 生成与模型梯度结构相同的元素=0的列表
    K_Gradients = []
    Loss_train = torch.tensor(0.)
    weight = []
    K_tau = []
    for idx_outer, (train_data, train_targets) in enumerate(federated_train_loader):
        K_tau.append(taus[train_data.location.id])   #添加延时信息
        model_round = models[train_data.location.id]
        optimizer = optims[train_data.location.id]
        train_data, train_targets = train_data.to(device), train_targets.to(device)
        train_data, train_targets = train_data.get(), train_targets.get()
        # 返回梯度张量，列表形式；同时返回loss；gradient=False，则返回-lr*grad
        user_gradient, loss = train(args.lr, model_round, train_data, train_targets, device, optimizer, gradient=True)
        user_gradient = TensorClip(user_gradient, L_norm(yun_gradient))
        if successItr % args.yunItr == 0:
            yun_gradient = yunTrain(args, model, yunData, yunLabels, device, optim_sever)
        if score(user_gradient, yun_gradient, args) >= -args.epsilon:
            successItr += 1
            first = True
            for grad_idx, params_sever in enumerate(model.parameters()):
                params_sever.data.add_(-args.lr, user_gradient[grad_idx])
        else:
            failItr += 1
    if torch.isinf(L_norm(user_gradient)) or torch.isnan(L_norm(user_gradient)):
        print('inf or nan')
    # 升级延时信息
    for tau in taus:
        taus[tau] = taus[tau] + 1
    for worker in workers_list:
        taus[worker] = 1

    # 同步更新不需要下面代码；异步更新需要下段代码
    for worker_idx in range(len(workers_list)):
        worker_model = models[workers_list[worker_idx]]
        for idx, (params_server, params_client) in enumerate(zip(model.parameters(), worker_model.parameters())):
            params_client.data = params_server.data
        models[workers_list[worker_idx]] = worker_model    # 添加把更新后的模型返回给用户


    if successItr == 1 or successItr % args.itr_test == 0 and first:
        first = False
        print('itr: {}, fail itr is {}, truth itr is {}'.format(itr, failItr, successItr))
        test_loss, test_acc = test(model, test_loader, device)  # 初始模型的预测精度
        logs['truthItr'].append(test_acc)
        logs['totalItr'].append(itr)
    if itr>100000:
        break


with open('./results/MNIST_ZENO_testacc.txt', 'a+') as fl:
    fl.write('\n%' + date + ' % Results (user_num is {}, K is {}, yunItr is {}, yunBZ is {}, yunNum is {}, yunclass is {} gamma is {} rho is {}, epsilon is {}'.
             format( args.user_num, args.K, args.yunItr, args.yunBatchSize, args.yunNum, args.yunclasses, args.gamma, args.rho, args.epsilon))
    fl.write(' BZ is {}, LR is {}, itr_test is {}, total itr is {}， classesis{})\n'.
        format( args.batch_size, args.lr, args.itr_test, args.total_iterations, args.classes))
    fl.write('ZENO: ' + str(logs['truthItr']) + ';')


