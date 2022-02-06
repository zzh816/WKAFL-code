import os
import pickle
import re
from PIL import Image
from torch.utils.data import Dataset
import json
import torch
import logging
import string
import syft as sy  # <-- NEW: import the Pysyft library
import numpy as np

logger = logging.getLogger(__name__)

class Femnist(Dataset):
    def __init__(self, data_root):
        '''
                data_root: 数据集位置
        '''

        self.num_samples = []
        self.data = []
        self.targets = []

        for r in data_root:
            print('载入数据集：',r)
            with open(r) as file:
                js = json.load(file)
                self.num_samples += js['num_samples']
                for u in js['users']:
                    self.data += js['user_data'][u]['x']
                    self.targets += js['user_data'][u]['y']

        self.data = torch.tensor(self.data).view(-1,1,28,28)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        data = self.data[idx]
        targets = self.targets[idx]
        return data, targets

class Shakespeare(Dataset):
    def __init__(self, data_root):
        self.ALL_LETTERS = '''1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz -,\'!;"[]?().:'''#符号表
        self.NUM_LETTERS = len(self.ALL_LETTERS)  # 76，符号表长度
        print(data_root)
        with open(data_root) as file:
            js = json.load(file)
            self.data = []
            self.targets = []
            self.num_samples = js['num_samples']
            for u in js['users']:
                for d in js['user_data'][u]['x']:
                    self.data.append([self.word_to_indices(d)])
                for t in js['user_data'][u]['y']:
                    self.targets.append(self.letter_to_vec(t))
                if len(self.targets)>100:
                    break


        self.data = torch.tensor(self.data)#.view(-1,1,80)


    def letter_to_vec(self, letter):
        '''returns one-hot representation of given letter
        '''
        index = self.ALL_LETTERS.find(letter)
        return index

    def word_to_indices(self, word):
        '''returns a list of character indices
        Args:
        word: string
        Return:
        indices: int list with length len(word)
        '''
        indices = []
        for c in word:
            indices.append(self.ALL_LETTERS.find(c))
        return indices

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        data = self.data[idx]
        target = self.targets[idx]
        return data,target

class sentiment140(Dataset):
    def __init__(self, data_root,vocab_root):
        self.word_emb_arr,self.indd,self.vocab = self.get_vocab(vocab_root)
        print(data_root)
        with open(data_root) as file:
            js = json.load(file)
            self.data = []
            self.targets = []
            self.length = []
            self.num_samples = js['num_samples']
            for u in js['users']:
                for d in js['user_data'][u]['x']:
                    emba,length = self.line_to_indices(d[4])
                    self.data.append([emba])
                    self.length.append(length)
                self.targets += js['user_data'][u]['y']
                #if len(self.targets)>100:
                    #break

        self.data = torch.tensor(self.data)
        self.targets = list(zip(self.targets,self.length))


    def split_line(self,line):
        return re.findall(r"[\w']+|[.,!?;]", line)


    def line_to_indices(self,line,  max_words=25):
        unk_id = len(self.indd)
        line_list = self.split_line(line)  # split phrase in words
        indl = [self.indd[w] if w in self.indd else unk_id for w in line_list[:max_words]]
        length = len(indl)
        indl += [unk_id] * (max_words - len(indl))
        emba = []
        for i in range(0,len(indl)):
            emba.append(self.word_emb_arr[indl[i]])
        return emba,length


    def get_vocab(self,path):
        with open(path, 'r') as inf:
            embs = json.load(inf)
        vocab = embs['vocab']
        word_emb_arr = embs['emba']
        indd = {}
        for i in range(len(vocab)):
            indd[vocab[i]] = i
        vocab = {w: i for i, w in enumerate(embs['vocab'])}
        return word_emb_arr, indd, vocab

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        data = self.data[idx]
        target = self.targets[idx]
        return data,target

class reddit(Dataset):
    def __init__(self, data_root,vocab_root):
        '''
                data_root: 数据集位置
                vocab_root：词汇表位置
        '''
        self.vocab = pickle.load(open(vocab_root,'rb'))
        self.num_vocab = self.vocab['size']  # 10000

        self.num_samples = []
        self.data = []
        self.targets = []

        for r in data_root:
            print('载入数据集：', r)
            with open(r) as file:
                js = json.load(file)
                self.num_samples += js['num_samples']
                for u in js['users']:
                    for d in js['user_data'][u]['x']:
                        for dd in d:
                            self.data.append([self.word_to_indices(dd)])
                    for t in js['user_data'][u]['y']:
                        for tt in t['target_tokens']:
                            self.targets.append(self.letter_to_index(tt[9]))
                    #if len(self.targets) > 100:
                        #break



        #print(self.data)

        self.data = torch.tensor(self.data)

    def letter_to_index(self, letter):
        '''returns one-hot representation of given letter
        '''
        if letter in self.vocab['vocab'].keys():
            index = self.vocab['vocab'][letter]
        else:
            index = 1
        return index

    def word_to_indices(self, word):
        '''returns a list of character indices
        Args:
        word: string
        Return:
        indices: int list with length len(word)
        '''
        indices = []
        for c in word:
            if c in self.vocab['vocab'].keys():
                indices.append(self.vocab['vocab'][c])
            else:
                indices.append(1)


        return indices

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        data = self.data[idx]
        target = self.targets[idx]
        return data,target

class synthetic(Dataset):
    def __init__(self, data_root):
        '''
        data_root: 数据集位置
        '''
        self.num_samples = []
        self.data = []
        self.targets = []

        print('载入数据集：', data_root)
        with open(data_root) as file:
            js = json.load(file)
            self.num_samples += js['num_samples']
            for u in js['users']:
                self.data += js['user_data'][u]['x']
                self.targets += js['user_data'][u]['y']

        self.data = torch.tensor(self.data).view(-1, 1, 60)


    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        data = self.data[idx]
        targets = self.targets[idx]
        return data, targets

class celeba(Dataset):
    def __init__(self, data_root,IMAGES_DIR,userNum):
        '''
        data_root: 数据集位置
        IMAGES_DIR：图片集位置
        '''
        self.IMAGE_SIZE = 84
        self.data = []
        self.targets = []
        print('载入数据集：', data_root)
        with open(data_root) as file:
            js = json.load(file)
            self.data = []
            self.targets = []
            self.num_samples = js['num_samples'][0:userNum]
            for i in range(userNum):
                u = js['users'][i]
                for img_name in js['user_data'][u]['x']:
                    self.data.append([self._load_image(img_name,IMAGES_DIR)])
                self.targets += js['user_data'][u]['y']

        print('长度', len(self.targets))
        self.data = torch.tensor(self.data)

    def _load_image(self, img_name,IMAGES_DIR):
        #根据图片编号载入图片数据
        img = Image.open(os.path.join(IMAGES_DIR, img_name))
        img = img.resize((self.IMAGE_SIZE, self.IMAGE_SIZE)).convert('RGB')
        img = np.array(img)
        img = (img.swapaxes(0,2)).swapaxes(1,2)
        return img

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        data = self.data[idx]
        targets = self.targets[idx]
        return data, targets

def dataset_federate_noniid(dataset, workers,Ratio = [1, 1, 1], net='NOT CNN' ):
    """
    Add a method to easily transform a torch.Dataset or a sy.BaseDataset
    into a sy.FederatedDataset. The dataset given is split in len(workers)
    part and sent to each workers
    """
    logger.info(f"Scanning and sending data to {', '.join([w.id for w in workers])}...")
    datasets = []
    N = 0
    dataset_list = list(dataset)
    for n in range(0, len(workers)):
        ratio = Ratio[n] / sum(Ratio)#计算比例
        num = round(ratio * len(dataset))#根据比例计算要抽取的数据的长度
        Subset = dataset_list[N:N+num]#抽取数据
        N = N+num
        data = []
        targets = []
        for d, t in Subset:
            data.append(d)
            targets.append(t)

        data = torch.cat(data)
        if net =='CNN':
            data = torch.unsqueeze(data, 1)

        targets = torch.tensor(targets)
        worker = workers[n]
        logger.debug("Sending data to worker %s", worker.id)
        data = data.send(worker)
        targets = targets.send(worker)
        datasets.append(sy.BaseDataset(data, targets))  # .send(worker)

    logger.debug("Done!")
    return sy.FederatedDataset(datasets)