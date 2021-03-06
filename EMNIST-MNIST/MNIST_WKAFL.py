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
        self.user_num = 1000      # number of total clients P
        self.K = 10     # number of participant clients K
        self.CB1 = 100   # clip parameter in both stages
        self.CB = 10   # clip parameter B at stage two
        self.lr = 0.005      # learning rate of global model
        self.itr_test = 50    # number of iterations for the two neighbour tests on test datasets
        self.batch_size = 4      # batch size of each client for local training
        self.test_batch_size = 128    # batch size for test datasets
        self.total_iterations = 5000  # total number of iterations
        self.threshold = 0.3    # threshold to judge whether gradients are consistent
        self.alpha = 0.1    # parameter for momentum to alleviate the effect of non-IID data
        self.classes = 1     # number of data classes on each client, which can determine the level of non-IID data
        self.seed = 1     # parameter for the server to initialize the model
        self.cuda_use = False

args = Argument()
use_cuda = args.cuda_use and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")


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

##################################????????????????????????????????????#############
def GetModelLayers(model):
    Layers_shape = []
    Layers_nodes = []
    for idx, params in enumerate(model.parameters()):
        Layers_num = idx
        Layers_shape.append(params.shape)
        Layers_nodes.append(params.numel())
    return Layers_num + 1, Layers_shape, Layers_nodes

##################################????????????????????????0#####################
def ZerosGradients(Layers_shape):
    ZeroGradient = []
    for i in range(len(Layers_shape)):
        ZeroGradient.append(torch.zeros(Layers_shape[i]))
    return ZeroGradient

################################???????????????###############################
def lr_adjust(args, tau):
    tau = 0.1*tau+1
    lr = args.lr/tau
    return lr

#################################????????????################################
def L_norm(Tensor):
    norm_Tensor = torch.tensor([0.])
    for i in range(len(Tensor)):
        norm_Tensor += Tensor[i].float().norm()**2
    return norm_Tensor.sqrt()

################################# ?????????????????? ############################
def similarity(user_Gradients, yun_Gradients):
    sim = torch.tensor([0.])
    for i in range(len(user_Gradients)):
        sim = sim + torch.sum(user_Gradients[i] * yun_Gradients[i])
    if L_norm(user_Gradients) == 0:
        print('?????????0.')
        sim = torch.tensor([1.])
        return sim
    sim = sim/(L_norm(user_Gradients)*L_norm(yun_Gradients))
    return sim

#################################??????####################################
def aggregation(Collect_Gradients, K_Gradients, weight, Layers_shape, args, Clip=False):
    sim = torch.zeros([args.K])
    Gradients_Total = torch.zeros([args.K+1])
    for i in range(args.K):
        Gradients_Total[i] = L_norm(K_Gradients[i])
    Gradients_Total [args.K] = L_norm(Collect_Gradients)
    #print('Gradients_norm', Gradients_Total)
    for i in range(args.K):
        sim[i] = similarity(K_Gradients[i], Collect_Gradients)
    index = (sim > args.threshold)
    #print('sim:', sim)
    if sum(index) == 0:
        print("??????????????????")
        return Collect_Gradients
    Collect_Gradients = ZerosGradients(Layers_shape)

    totalSim = []
    Sel_Gradients = []
    for i in range(args.K):
        if sim[i] > args.threshold:
            totalSim.append((torch.exp(sim[i] * 10) * weight[i]).tolist())
            Sel_Gradients.append(K_Gradients[i])
    totalSim = torch.tensor(totalSim)
    totalSim = totalSim / torch.sum(totalSim)
    for i in range(len(totalSim)):
        Gradients_Sample = Sel_Gradients[i]
        if Clip:
            standNorm = L_norm(Collect_Gradients)
            Gradients_Sample = TensorClip(Gradients_Sample, args.CB2 * standNorm)
        for j in range(len(K_Gradients[i])):
            Collect_Gradients[j] += Gradients_Sample[j] * totalSim[i]
    return Collect_Gradients

################################ ???????????? #################################
def TensorClip(Tensor, ClipBound):
    norm_Tensor = L_norm(Tensor)
    if ClipBound<norm_Tensor:
        for i in range(Layers_num):
            Tensor[i] = 1.*Tensor[i]*ClipBound/norm_Tensor
    return Tensor

############################ ?????????????????? ################################
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

##########################?????????????????????????????????########################
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
            Gradients_Tensor.append(-learning_rate*params.grad.data)#??????-lr*grad
    if gradient == True:
        for params in model.parameters():
            Gradients_Tensor.append(params.grad.data)#?????????????????????????????????Gradients_Tensor
    return Gradients_Tensor, loss

###################################################################################
##################################?????????????????????###################################
model = Net()
workers = []
models = {}
optims = {}
taus = {}
for i in range(1, args.user_num+1):
    exec('user{} = sy.VirtualWorker(hook, id="user{}")'.format(i,i))
    exec('models["user{}"] = model.copy()'.format(i))
    exec('optims["user{}"] = optim.SGD(params=models["user{}"].parameters(), lr=args.lr)'.format(i,i))
    exec('workers.append(user{})'.format(i))    # ????????????????????????
    exec('taus["user{}"] = {}'.format(i,1))      # ????????????????????????
    # exec('workers["user{}"] = user{}'.format(i,i))    #????????????????????????
optim_sever = optim.SGD(params=model.parameters(), lr=args.lr)    # ????????????????????????
###################################################################################
###############################????????????############################################
dataType = 'mnist'   # ??????bymerge, byclass, digits, mnist, letters, balanced
datasets = rawDatasetsLoader.loadDatesets(trainDataSize = 70000, testDataSize = 20000, dataType=dataType)
#?????????????????????, datasNum????????????datasNum[i]?????????i????????????????????????????????????['3']=45??????????????????45???
federated_data, datasNum = rawDatasetsLoader.dataset_federate_noniid(datasets, workers, args)
#Jaccard = JaDis(datasNum, args.user_num)
#print('Jaccard distance is {}'.format(Jaccard))

test_data = rawDatasetsLoader.testImages(datasets)
del datasets
test_loader = torch.utils.data.DataLoader(test_data,
    batch_size=args.test_batch_size, shuffle=True, drop_last=True, num_workers=0)

#??????????????????
logs = {'train_loss': [], 'test_loss': [], 'test_acc': []}
test_loss, test_acc = test(model, test_loader, device) # ???????????????????????????
logs['test_acc'].append(test_acc)

###################################################################################
#################################??????????????????######################################
#?????????????????????????????????
Layers_num, Layers_shape, Layers_nodes = GetModelLayers(model)
e = torch.exp(torch.tensor(1.))
# ????????????/????????????
for itr in range(1, args.total_iterations + 1):
    # ????????????????????????????????????????????????????????????????????????????????????????????????batch_size
    # ???????????????????????????????????????????????????batch_size=1???batch_num=args.batch_size*args.batchs_round??????????????????????????????
    federated_train_loader = sy.FederatedDataLoader(federated_data, batch_size=args.batch_size, shuffle=True,
                                                    worker_num=args.K, batch_num=1)
    workers_list = federated_train_loader.workers    # ?????????????????????????????????

    # ??????????????????????????????????????????=0?????????
    K_Gradients = []
    Loss_train = torch.tensor(0.)
    weight = []
    K_tau = []
    for idx_outer, (train_data, train_targets) in enumerate(federated_train_loader):
        K_tau.append(taus[train_data.location.id])   #??????????????????
        model_round = models[train_data.location.id]
        optimizer = optims[train_data.location.id]
        train_data, train_targets = train_data.to(device), train_targets.to(device)
        train_data, train_targets = train_data.get(), train_targets.get()
        # ????????????????????????????????????????????????loss???gradient=False????????????-lr*grad
        Gradients_Sample, loss = train(args.lr, model_round, train_data, train_targets, device, optimizer, gradient=True)
        if itr > 1:
            for j in range(Layers_num):
                Gradients_Sample[j] = Gradients_Sample[j] + args.alpha * Collect_Gradients[j]
        K_Gradients.append(TensorClip(Gradients_Sample, args.CB1))
        Loss_train += loss

    Collect_Gradients = ZerosGradients(Layers_shape)
    K_tau = torch.tensor(K_tau)*1.
    _, index = torch.sort(K_tau)
    normStandard = L_norm(K_Gradients[index[0]])
    weight = (e/2)**(-K_tau)
    if torch.sum(weight) == 0:
        print("???????????????")
        for i in range(Layers_num):
            weight[index[0]] = 1.
            Collect_Gradients = K_Gradients[index[0]]
    else:
        weight = weight/torch.sum(weight)
        for i in range(args.K):
            Gradients_Sample = K_Gradients[i]
            Gradients_Sample = TensorClip(Gradients_Sample, normStandard * args.CB1)
            for j in range(Layers_num):
                Collect_Gradients[j] += Gradients_Sample[j]*weight[i]

    #print('weight:', weight, 'tau', K_tau)
    if itr < 1000:
        Collect_Gradients = aggregation(Collect_Gradients, K_Gradients, weight, Layers_shape, args)
    elif itr>100:
        Collect_Gradients = aggregation(Collect_Gradients, K_Gradients, weight, Layers_shape, args, Clip=True)

    # ??????????????????
    for tau in taus:
        taus[tau] = taus[tau] + 1
    for worker in workers_list:
        taus[worker] = 1

    lr = lr_adjust(args, torch.min(K_tau))
    for grad_idx, params_sever in enumerate(model.parameters()):
        params_sever.data.add_(-lr, Collect_Gradients[grad_idx])


    # ??????????????????????????????????????????????????????????????????
    for worker_idx in range(len(workers_list)):
        worker_model = models[workers_list[worker_idx]]
        for idx, (params_server, params_client) in enumerate(zip(model.parameters(),worker_model.parameters())):
            params_client.data = params_server.data
        models[workers_list[worker_idx]] = worker_model    # ??????????????????????????????????????????

    if itr == 1 or itr % args.itr_test == 0:
        print('itr: {}'.format(itr))
        test_loss, test_acc = test(model, test_loader, device)  # ???????????????????????????
        logs['test_acc'].append(test_acc)
        logs['test_loss'].append(test_loss)
    if itr == 1 or itr % args.itr_test == 0:
        # ??????????????????
        Loss_train /= (idx_outer + 1)
        logs['train_loss'].append(Loss_train)

with open('./results/MNIST_WKAFL_testacc.txt', 'a+') as fl:
    fl.write('\n' + date + ' % Results (user_num is {}, K is {}, CB is {}, B is {}, sim_threshold is {},'.
             format( args.user_num, args.K, args.CB1, args.CB, args.threshold))
    fl.write(' BZ is {}, LR is {}, itr_test is {}, total itr is {}??? classesis{})\n'.
        format( args.batch_size, args.lr, args.itr_test, args.total_iterations, args.classes))
    fl.write('WKAFL: ' + str(logs['test_acc']))

with open('./results/MNIST_WKAFL_trainloss.txt', 'a+') as fl:
    fl.write('\n' + date + ' %Results (user_num is {}, K is {}, CB is {}, B is {}, sim_threshold is {},'.
             format(args.user_num, args.K, args.CB1, args.CB, args.threshold))
    fl.write(' BZ is {}, LR is {}, itr_test is {}, total itr is {})\n'.
        format( args.batch_size, args.lr, args.itr_test, args.total_iterations))
    fl.write('train_loss = ' + str(logs['train_loss']))

with open('./results/MNIST_WKAFL_testloss.txt', 'a+') as fl:
    fl.write('\n' + date + ' %Results (user_num is {}, K is {}, CB is {}, B is {}, sim_threshold is {},'.
             format(args.user_num, args.K, args.CB1, args.CB, args.threshold))
    fl.write(' BZ is {}, LR is {}, itr_test is {}, total itr is {})\n'.
             format( args.batch_size, args.lr, args.itr_test, args.total_iterations))
    fl.write('test_loss: ' + str(logs['test_loss']))
