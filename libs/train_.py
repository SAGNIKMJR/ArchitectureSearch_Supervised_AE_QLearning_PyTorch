import argparse
import os
import shutil
import time
import collections
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from Datasets.datasets import MNIST

class network(nn.Module):

    def __init__(self, state_list, state_space_parameters):
        super(network, self).__init__()
    	self.state_list = state_list
        self.state_space_parameters = state_space_parameters
    	# print('Training in PyTorch:')
    	# print(((self.state_list)[0]).layer_type)

        ''' uncomment to include dropout '''

    	# total_drop_no = 0
    	# for state in self.state_list:                            # uncomment to include dropout
     #        if state.layer_type == 'dropout':
     #        	total_drop_no += 1
    	feature_list = []
        classifier_list = []

        ''' uncomment to include dropoout '''
    	# if total_drop_no != 0:
    	# 	dropout_val = 0.5/total_drop_no		#uncomment to include dropout				# linearly increasing dropout value from 0 to 0.5 
    	# else:
    	# 	dropout_val = None													#  as mentioned in state_string_utils.py
    	
        ''' uncomment to include pool and dropput '''
        # conv_no = pool_no = fc_no = relu_no = drop_no = 0
        conv_no = relu_no = batchNorm_no = fc_no = 0 
        feature = 1
        in_channel = 0
        out_channel = (self.state_space_parameters).input_channel
        no_feature = (self.state_space_parameters).input_channel*(((self.state_space_parameters).image_size)**2)
        global final_no_feature
        final_no_feature = no_feature
        print('***')
        defeature_list = []
    	for state_no, state in enumerate(self.state_list):
            if state_no == len(self.state_list)-1:
                break
            if state.layer_type == 'fc':
                feature = 0
            if feature == 1:
                if state.layer_type == 'conv':
                    conv_no += 1
                    in_channel = out_channel
                    out_channel = state.filter_depth
                    no_feature = ((state.image_size)**2)*(out_channel)
                    final_no_feature = no_feature
                    ''' uncommenting starts '''
                    # feature_list.append(('conv' + str(conv_no), nn.Conv2d(in_channel, out_channel, \
                    #                     state.filter_size, stride = state.stride, padding = \
                    #                     (state.filter_size - 1)/2)))  # complete this
                    ''' uncommenting ends '''

                    feature_list.append(('conv' + str(conv_no), nn.Conv2d(in_channel, out_channel, \
                    state.filter_size, stride = state.stride)))  # complete this
                    defeature_list.append((out_channel, in_channel, state.filter_size))
                    batchNorm_no += 1
                    feature_list.append(('batchNorm' + str(batchNorm_no), nn.BatchNorm2d(out_channel)))
                    relu_no += 1
                    feature_list.append(('relu' + str(relu_no), nn.ReLU(inplace = True)))


                ''' uncomment to include pool '''
                # elif state.layer_type == 'pool':
                #     in_channel = out_channel
                #     no_feature = ((state.image_size)**2)*(out_channel)
                #     final_no_feature = no_feature
                #     pool_no += 1
                #     feature_list.append(('pool' + str(pool_no), nn.MaxPool2d(state.filter_size, state.stride))) # complete this
                
                ''' uncomment to include dropout '''
                # elif state.layer_type == 'dropout':
                #     drop_no += 1
                #     in_channel = out_channel
                #     no_feature = ((state.image_size)**2)*(out_channel)
                #     final_no_feature = no_feature
                #     feature_list.append(('dropout' + str(drop_no), nn.Dropout2d(p =drop_no*dropout_val)))
            else:
                if state.layer_type == 'fc':
                    fc_no += 1
                    in_channel = out_channel
                    in_feature = no_feature
                    no_feature = (state.fc_size)
                    out_channel = 1
                    classifier_list.append(('fc' + str(fc_no), nn.Linear(in_feature, no_feature))) # complete this
                
                ''' uncomment to include dropout '''
                # elif state.layer_type == 'dropout':
                #     drop_no += 1
                #     in_channel = out_channel
                #     out_channel = 1
                #     classifier_list.append(('dropout' + str(drop_no), nn.Dropout(p =drop_no*dropout_val)))
        classifier_list.append(('fc' + str(fc_no+1), nn.Linear(no_feature, state_space_parameters.output_states)))
        classifier_list.append(('sm1', nn.LogSoftmax()))
        self.features_list = nn.Sequential(collections.OrderedDict(feature_list))
        self.classifiers_list = nn.Sequential(collections.OrderedDict(classifier_list))
        feature_list_last2removed = []
        for i in range(len(feature_list)-2):
            feature_list_last2removed.append(feature_list[i])
        self.features_list_last2removed = nn.Sequential(collections.OrderedDict(feature_list_last2removed))
        defeature2_list = []
        convT_no = 0
        for i in range(len(defeature_list) - 1):
            index = len(defeature_list) - 1 - i 
            convT_no += 1
            defeature2_list.append(('convT' + str(convT_no), nn.ConvTranspose2d(defeature_list[index][0], \
                                    defeature_list[index][1], defeature_list[index][2])))
            batchNorm_no += 1
            defeature2_list.append(('batchNorm' + str(batchNorm_no), nn.BatchNorm2d(defeature_list[index][1])))
            relu_no += 1
            defeature2_list.append(('relu' + str(relu_no), nn.ReLU(inplace = True)))
        if len(defeature_list)>0:
            convT_no += 1
            defeature2_list.append(('convT' + str(convT_no), nn.ConvTranspose2d(defeature_list[0][0], \
                                    defeature_list[0][1], defeature_list[0][2])))
            defeature2_list.append(('sigmoid', nn.Sigmoid()))
        self.defeatures_list = nn.Sequential(collections.OrderedDict(defeature2_list))
        self.conv_no = conv_no
        if self.conv_no >= 2:
            # print(self.conv_no == convT_no)
            self.features_list_last2removed_uptoPenultimate = nn.Sequential(collections.OrderedDict(feature_list_last2removed[:-1]))
            # print(feature_list_last2removed)
            # print(defeature2_list[:3])

            self.defeatures_list_uptoFirst = nn.Sequential(collections.OrderedDict(feature_list_last2removed + defeature2_list[:3]))
    def forward(self, x):
        ''' uncomment for normal cnn''' 
        # x = self.features_list(x)
        # x = x.view(x.size(0), final_no_feature)
        # x = self.classifiers_list(x)
        # return x
        ''' uncommenting ends '''
        if self.conv_no >= 2:
            mapConv_untilPenultimate = self.features_list_last2removed_uptoPenultimate(x)
            dataForFeature = mapConv_untilPenultimate.data
            no_feature = dataForFeature.size(0)*dataForFeature.size(1)*dataForFeature.size(2)*dataForFeature.size(3)
            mapDeConv_afterFirst = self.defeatures_list_uptoFirst(x)
            return (2, mapConv_untilPenultimate, mapDeConv_afterFirst, no_feature)
        elif self.conv_no == 1:
            temp = self.features_list_last2removed(x)
            output = self.defeatures_list(temp)
            dataForFeature = output.data
            no_feature = dataForFeature.size(0)*dataForFeature.size(1)*dataForFeature.size(2)*dataForFeature.size(3)
            return (1, x, output, no_feature)
        else:
            return (0)
class network2(nn.Module):

    def __init__(self, state_list, state_space_parameters):
        super(network2, self).__init__()
        self.state_list = state_list
        self.state_space_parameters = state_space_parameters
        # print('Training in PyTorch:')
        # print(((self.state_list)[0]).layer_type)
        total_drop_no = 0
        for state in self.state_list:
            if state.layer_type == 'dropout':
                total_drop_no += 1
        feature_list = []
        classifier_list = []
        if total_drop_no != 0:
            dropout_val = 0.5/total_drop_no                     # linearly increasing dropout value from 0 to 0.5 
        else:
            dropout_val = None                                                  #  as mentioned in state_string_utils.py
        conv_no = pool_no = fc_no = relu_no = drop_no = 0
        feature = 1
        in_channel = 0
        out_channel = (self.state_space_parameters).input_channel
        no_feature = (self.state_space_parameters).input_channel*(((self.state_space_parameters).image_size)**2)
        global final_no_feature
        final_no_feature = no_feature
        print('***')
        for state_no, state in enumerate(self.state_list):
            if state_no == len(self.state_list)-1:
                break
            if state.layer_type == 'fc':
                feature = 0
            if feature == 1:
                if state.layer_type == 'conv':
                    conv_no += 1
                    in_channel = out_channel
                    out_channel = state.filter_depth
                    no_feature = ((state.image_size)**2)*(out_channel)
                    final_no_feature = no_feature
                    feature_list.append(('conv' + str(conv_no), nn.Conv2d(in_channel, out_channel, \
                                        state.filter_size, stride = state.stride)))  # complete this
                    relu_no += 1
                    feature_list.append(('relu' + str(relu_no), nn.ReLU(inplace = True)))
                elif state.layer_type == 'pool':
                    in_channel = out_channel
                    no_feature = ((state.image_size)**2)*(out_channel)
                    final_no_feature = no_feature
                    pool_no += 1
                    feature_list.append(('pool' + str(pool_no), nn.MaxPool2d(state.filter_size, state.stride))) # complete this
                elif state.layer_type == 'dropout':
                    drop_no += 1
                    in_channel = out_channel
                    no_feature = ((state.image_size)**2)*(out_channel)
                    final_no_feature = no_feature
                    feature_list.append(('dropout' + str(drop_no), nn.Dropout2d(p =drop_no*dropout_val)))
            else:
                if state.layer_type == 'fc':
                    fc_no += 1
                    in_channel = out_channel
                    in_feature = no_feature
                    no_feature = (state.fc_size)
                    out_channel = 1
                    classifier_list.append(('fc' + str(fc_no), nn.Linear(in_feature, no_feature))) # complete this
                elif state.layer_type == 'dropout':
                    drop_no += 1
                    in_channel = out_channel
                    out_channel = 1
                    classifier_list.append(('dropout' + str(drop_no), nn.Dropout(p =drop_no*dropout_val)))
        classifier_list.append(('fc' + str(fc_no+1), nn.Linear(no_feature, state_space_parameters.output_states)))
        classifier_list.append(('sm1', nn.LogSoftmax()))
        self.features_list = nn.Sequential(collections.OrderedDict(feature_list))
        self.classifiers_list = nn.Sequential(collections.OrderedDict(classifier_list))

    def forward(self, x):
        x = self.features_list(x)
        x = x.view(x.size(0), final_no_feature)
        x = self.classifiers_list(x)
        return x
best_prec1 = 0.
def train_val_net(state_list, state_space_parameters, data_path):
    global best_prec1
    best_prec1 = 0.
    model = network(state_list, state_space_parameters)
    print(model)

    ''' Uncomment for normal training '''
    # for param in model.features_list.parameters():                 #training only the classifier, comment otherwise
    #     param.requires_grad = False

    ''' Uncommenting ends '''

    # for param in model.parameters():
    #     print('1..\n')
    #     if param.requires_grad == True:
    #         print('1\n')
    # print('2\n')
    # model.features = torch.nn.DataParallel(model.features)               ### Uncomment when cuda is used
    # model.cuda()
    # criterion = nn.NLLLoss().cuda()
    criterion = nn.MSELoss(size_average = False)                                               ### Uncomment when cuda is used
    # cudnn.benchmark = True
                 ### Yet to do the Xavier initialization ###

    ''' Uncomment for normal training '''
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), # model.parameters()#uncomment if whole network being trained
    #                        lr = state_space_parameters.training_lr,\
    #                        betas = (state_space_parameters.beta1, state_space_parameters.beta2), \
    #                        eps = state_space_parameters.eps, weight_decay = state_space_parameters.weight_decay_rate)
    ''' Uncommenting ends '''

    traindir = os.path.join(data_path, 'train/')
    valdir = os.path.join(data_path, 'test/')

                        ### Don't Normalize
    # normalize = transforms.Normalize(mean=[0.46626906767843135, 0.3553785852011765, 0.353770305034902],    #mean
    #                                  std= [0.11063050470941177, 0.10724310918039215, 0.1115941032545098])  # and std for MNIST

    # train_dataset = datasets.ImageFolder(
    #     traindir,
    #     transforms.Compose([
    #         # transforms.RandomHorizontalFlip(),                   # No mirroring
    #         transforms.ToTensor(),
    #         # normalize,                                           # No normalization
    #     ]))
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size = state_space_parameters.train_batch_size, shuffle = True, \
    #     num_workers = state_space_parameters.workers)
    #     # pin_memory = True, \
    #     # sampler = state_space_parameters.train_sampler)

    # val_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder(valdir, transforms.Compose([
    #         transforms.ToTensor(),
    #         # normalize,                                           # No normalization
    #     ])),
    #     batch_size = state_space_parameters.eval_batch_size, shuffle=False, \
    #     num_workers = state_space_parameters.workers)
    #     # pin_memory=True)

                        ###Using Martin's dataloaders instead

    mnist = MNIST(False, state_space_parameters.image_size, state_space_parameters.train_batch_size,\
                  state_space_parameters.eval_batch_size, state_space_parameters.workers, traindir, valdir)
    train_loader = mnist.train_loader
    val_loader = mnist.val_loader

    ''' Uncomment for normal training ''' 
    # start_lr = state_space_parameters.training_lr
    # train_flag = True
    # epoch = 0
    # restart = 0
    # while epoch!=state_space_parameters.end_epoch:
    #     epoch += 1
    #     adjust_learning_rate(optimizer, epoch, start_lr)
    #     train(train_loader, model, criterion, optimizer, epoch)
    #     prec1 = validate(val_loader, model, criterion)
    #     if restart<5 and prec1<(state_space_parameters.acc_threshold*100):
    #         print('Restarting.')
    #         restart += 1
    #         epoch = 0
    #         start_lr = start_lr*0.4     # reduction by 0.4
    #         continue
    #     elif prec1<(100.0*state_space_parameters.acc_threshold):
    #         print('Quitting before completing training.')
    #         train_flag = False
    #         break
    #     is_best = prec1 > best_prec1
    #     print('Epoch no:{}'.format(epoch))
    #     best_prec1 = max(prec1, best_prec1)

    # last_val = prec1
    # best_val = best_prec1
    ''' Uncommenting ends '''
    total_loss = 0.
    if model.conv_no >= 1:
        for i, (input, target) in enumerate(train_loader):
            input_var = torch.autograd.Variable(input, volatile=True)
            target_var = torch.autograd.Variable(target, volatile=True)
            tupl = model(input_var)
            total_loss += criterion(tupl[1], tupl[2]).data /float(tupl[3])
        for i, (input, target) in enumerate(val_loader):
            input_var = torch.autograd.Variable(input, volatile=True)
            target_var = torch.autograd.Variable(target, volatile=True)
            tupl = model(input_var)
            total_loss += criterion(tupl[1], tupl[2]).data /float(tupl[3])
        print(total_loss)
        loss_inverse = 1.0/float(total_loss[0])
        loss = total_loss
        computeLoss_flag = True
        return (loss_inverse, loss, computeLoss_flag)
    else:
        return (-float('inf'), 0., False )

def train_val_net2(state_list, state_space_parameters, data_path):
    global best_prec1
    best_prec1 = 0.
    model = network2(state_list, state_space_parameters)
    print(model)
    # for param in model.features_list.parameters():                 #training only the classifier, comment otherwise
    #     param.requires_grad = False
    # for param in model.parameters():
    #     print('1..\n')
    #     if param.requires_grad == True:
    #         print('1\n')
    # print('2\n')
    model = torch.nn.DataParallel(model)                                  ### Uncomment when cuda is used
    model.cuda()
    criterion = nn.NLLLoss().cuda()
    # criterion = nn.NLLLoss()                                               ### Uncomment when cuda is used
    cudnn.benchmark = True
                 ### Yet to do the Xavier initialization ###

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), # model.parameters()#uncomment if whole network being trained
                           lr = state_space_parameters.training_lr,\
                           betas = (state_space_parameters.beta1, state_space_parameters.beta2), \
                           eps = state_space_parameters.eps, weight_decay = state_space_parameters.weight_decay_rate)

    traindir = os.path.join(data_path, 'train/')
    valdir = os.path.join(data_path, 'test/')

                        ### Don't Normalize
    # normalize = transforms.Normalize(mean=[0.46626906767843135, 0.3553785852011765, 0.353770305034902],    #mean
    #                                  std= [0.11063050470941177, 0.10724310918039215, 0.1115941032545098])  # and std for MNIST

    # train_dataset = datasets.ImageFolder(
    #     traindir,
    #     transforms.Compose([
    #         # transforms.RandomHorizontalFlip(),                   # No mirroring
    #         transforms.ToTensor(),
    #         # normalize,                                           # No normalization
    #     ]))
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size = state_space_parameters.train_batch_size, shuffle = True, \
    #     num_workers = state_space_parameters.workers)
    #     # pin_memory = True, \
    #     # sampler = state_space_parameters.train_sampler)

    # val_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder(valdir, transforms.Compose([
    #         transforms.ToTensor(),
    #         # normalize,                                           # No normalization
    #     ])),
    #     batch_size = state_space_parameters.eval_batch_size, shuffle=False, \
    #     num_workers = state_space_parameters.workers)
    #     # pin_memory=True)

                        ###Using Martin's dataloaders instead

    mnist = MNIST(False, state_space_parameters.image_size, state_space_parameters.train_batch_size,\
                  state_space_parameters.eval_batch_size, state_space_parameters.workers, traindir, valdir)
    train_loader = mnist.train_loader
    val_loader = mnist.val_loader
    start_lr = state_space_parameters.training_lr
    train_flag = True
    epoch = 0
    restart = 0
    while epoch!=state_space_parameters.end_epoch:
        epoch += 1
        adjust_learning_rate(optimizer, epoch, start_lr)
        train(train_loader, model, criterion, optimizer, epoch)
        prec1 = validate(val_loader, model, criterion)
        if restart<5 and prec1<(state_space_parameters.acc_threshold*100):
            print('Restarting.')
            restart += 1
            epoch = 0
            start_lr = start_lr*0.4     # reduction by 0.4
            continue
        elif prec1<(100.0*state_space_parameters.acc_threshold):
            print('Quitting before completing training.')
            train_flag = False
            break
        is_best = prec1 > best_prec1
        print('Epoch no:{}'.format(epoch))
        best_prec1 = max(prec1, best_prec1)

    last_val = prec1
    best_val = best_prec1
    return [best_val, last_val, train_flag]
                ### Validate after this
def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        target = target.cuda(async=True)                          ### Uncomment when cuda is used                       
        input_var = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target).cuda()
        output = model(input_var)
        # print(type(target_var), type(output))
        loss = criterion(output, target_var)

        prec1, prec3 = accuracy(output.data, target, topk=(1,3))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top3.update(prec3[0], input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        # print('yo')

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)                              ### Uncomment when cuda is used
        input_var = torch.autograd.Variable(input, volatile=True).cuda()
        target_var = torch.autograd.Variable(target, volatile=True).cuda()

        output = model(input_var)
        loss = criterion(output, target_var)

        prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top3.update(prec3[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print(' * Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f}'
          .format(top1 = top1, top3 = top3))

    return top1.avg

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def adjust_learning_rate(optimizer, epoch, training_lr):
    lr = training_lr * (0.2**(epoch//5))  # reduction by 0.2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr




