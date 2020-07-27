import fire
import os
import time
import torch
from torchvision import datasets, transforms
import torch.nn as nn

class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
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


def train_epoch(model, loader, optimizer, epoch, n_epochs, print_freq=1):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()

    # Model on train mode
    model.train()

    end = time.time()
    for batch_idx, (input, target) in enumerate(loader):
        # Create vaiables
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # compute output
        output = model(input)
        loss = torch.nn.functional.cross_entropy(output, target)

        # measure accuracy and record loss
        batch_size = target.size(0)
        _, pred = output.data.cpu().topk(1, dim=1)
        error.update(torch.ne(pred.squeeze(), target.cpu()).float().sum().item() / batch_size, batch_size)
        losses.update(loss.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print stats
        if batch_idx % print_freq == 0:
            res = '\t'.join([
                'Epoch: [%d/%d]' % (epoch + 1, n_epochs),
                'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                'Error %.4f (%.4f)' % (error.val, error.avg),
            ])
            print(res)

    # Return summary statistics
    return batch_time.avg, losses.avg, error.avg


def test_epoch(model, loader, print_freq=1, is_test=True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()

    # Model on eval mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            # Create vaiables
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = torch.nn.functional.cross_entropy(output, target)

            # measure accuracy and record loss
            batch_size = target.size(0)
            _, pred = output.data.cpu().topk(1, dim=1)
            error.update(torch.ne(pred.squeeze(), target.cpu()).float().sum().item() / batch_size, batch_size)
            losses.update(loss.item(), batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print stats
            if batch_idx % print_freq == 0:
                res = '\t'.join([
                    'Test' if is_test else 'Valid',
                    'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                    'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                    'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                    'Error %.4f (%.4f)' % (error.val, error.avg),
                ])
                print(res)

    # Return summary statistics
    return batch_time.avg, losses.avg, error.avg


def train(model, train_set, valid_set, test_set, save, n_epochs=300,
          batch_size=64, lr=0.1, wd=0.0001, momentum=0.9, seed=None, model_type="", prune=False):
    if seed is not None:
        torch.manual_seed(seed)

    train_prec_log = []
    test_prec_log = []

    # Data loaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               pin_memory=(torch.cuda.is_available()), num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                              pin_memory=(torch.cuda.is_available()), num_workers=0)
    if valid_set is None:
        valid_loader = None
    else:
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False,
                                                   pin_memory=(torch.cuda.is_available()), num_workers=0)
    # Model on cuda
    if torch.cuda.is_available():
        model = model.cuda()
    
    if prune:
        if model_type=='normal': #do fusion of normal models
            weights_dict = {}
            bias_dict = {}
            bmodel = None
            save=save[:-1] #strip default version of 0 from save folder name
            print(save)
            #populate weight and bias dicts from 4 normal models
            for d in [0,1,2,3]:
                model.load_state_dict(torch.load(os.path.join(save+str(d), 'model.th'))['state_dict'])
                # model.cuda()
                for id,module in enumerate(model.modules()):
                    if(type(module) == nn.Conv2d or type(module) == nn.Linear ):
                        if id in weights_dict:
                            weights_dict[id].append(module.weight.data.clone())
                            if module.bias is not None:
                                bias_dict[id].append(module.bias.data.clone())
                        else:
                            weights_dict[id]=[]
                            bias_dict[id]=[]
                            weights_dict[id].append(module.weight.data.clone())
                            if module.bias is not None:
                                bias_dict[id].append(module.bias.data.clone())
            #create the target model by fusing the weights and bias dicts into a single model
            for id,module in enumerate(model.modules()):
                if(type(module) == nn.Conv2d or type(module) == nn.Linear ):

                    var_inv_sum = 0
                    for i in range(4):
                        var_inv_sum +=  1/weights_dict[id][i].var()

                    module.weight.data.fill_(0)               
                    for i in range(4):
                        vrinv = (1/weights_dict[id][i].var())/var_inv_sum
                        module.weight.data+= vrinv*weights_dict[id][i]

                    if module.bias is not None:
                        module.bias.data.fill_(0)
                        for i in range(4):
                            vrinv = (1/weights_dict[id][i].var())/var_inv_sum
                            module.bias.data+= vrinv*bias_dict[id][i]
                            
            test_results = test_epoch(
                model=model,
                loader=test_loader,
                is_test=True
            )
            _, _, test_error = test_results
            
            save+="fused" #create new folder for fused data
            if not os.path.exists(save):
                os.makedirs(save)
            with open(os.path.join(save, 'fused.txt'), 'w') as f:
                f.write('%0.5f\n' % (test_error))
                f.close()
            print('Final test error: %.4f' % test_error)
        
        else:
            # Final test of model on test set
            model.load_state_dict(torch.load(os.path.join(save, 'model.th'))['state_dict'])
            model.cuda()
            # if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                # model = torch.nn.DataParallel(model).cuda()
            test_results = test_epoch(
                model=model,
                loader=test_loader,
                is_test=True
            )
            _, _, test_error = test_results
            with open(os.path.join(save, 'noprune.txt'), 'w') as f:
                f.write('%0.5f\n' % (test_error))
                f.close()
            print('Final test error: %.4f' % test_error)

            #if model_type != "normal":

            cnt_=0
            for m in model.modules():
                if hasattr(m, "domms"):
                    # print("Apply inv variance")
                    m.domms = False
                    m.apply_weights_pruning()
                    cnt_+=1
            print("CNT:",cnt_)

            test_results = test_epoch(
                model=model,
                loader=test_loader,
                is_test=True
            )
            _, _, test_error = test_results
            with open(os.path.join(save, 'prune.txt'), 'w') as f:
                f.write('%0.5f\n' % (test_error))
                f.close()
            print('Final test error pruning: %.4f' % test_error)
    else:
    # Wrap model for multi-GPUs, if necessary
        model_wrapper = model
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model_wrapper = torch.nn.DataParallel(model).cuda()

        # Optimizer
        optimizer = torch.optim.SGD(model_wrapper.parameters(), lr=lr, momentum=momentum, nesterov=True, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs],
                                                         gamma=0.1, last_epoch=-1)

        # optionally resume from a checkpoint
        if os.path.exists(os.path.join(save, 'epoch.txt')):

            print("=> loading checkpoint '{}'".format(os.path.join(save, 'chk.th')))
            checkpoint = torch.load(os.path.join(save, 'model.th'))
            start_epoch = checkpoint['epoch']
            best_error = checkpoint['best_error']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(os.path.join(save, 'model.th'), checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(os.path.join(save, 'chk.th')))                                             
            # Train model
            best_error = 1
            start_epoch = 0

        # Start log
        with open(os.path.join(save, 'results.csv'), 'w') as f:
            f.write('epoch,train_loss,train_error,valid_loss,valid_error,test_error\n')
        if start_epoch >0:
            for i in range(start_epoch):
                scheduler.step()

        for epoch in range(start_epoch,n_epochs):
            _, train_loss, train_error = train_epoch(
                model=model_wrapper,
                loader=train_loader,
                optimizer=optimizer,
                epoch=epoch,
                n_epochs=n_epochs,
            )
            scheduler.step()
            _, valid_loss, valid_error = test_epoch(
                model=model_wrapper,
                loader=valid_loader if valid_loader else test_loader,
                is_test=(not valid_loader)
            )


            # Determine if model is the best
            if valid_loader:
                if valid_error < best_error:
                    best_error = valid_error
                    print('New best error: %.4f' % best_error)
                    # torch.save(model.state_dict(), os.path.join(save, 'model.dat'))
                    torch.save({
                    'state_dict': model.state_dict(),
                    'best_error': best_error,
                    'best_loss':valid_loss,
                    'epoch': epoch + 1,
                    } ,os.path.join(save, 'model.th'))
            # else:
                # torch.save(model.state_dict(), os.path.join(save, 'model.dat'))

            # Log results
            with open(os.path.join(save, 'results.csv'), 'a') as f:
                f.write('%03d,%0.6f,%0.6f,%0.5f,%0.5f,\n' % (
                    (epoch + 1),
                    train_loss,
                    train_error,
                    valid_loss,
                    valid_error,
                ))

            if os.path.exists(os.path.join(save, 'train_log.txt')):
                append_write = 'a' # append if already exists
            else:
                append_write = 'w' # make a new file if not
            highscore = open(os.path.join(save, 'train_log.txt'),append_write)
            highscore.write(str(train_loss)+"\t"+str(train_error) + '\n')
            highscore.close()

            if os.path.exists(os.path.join(save, 'test_log.txt')):
                append_write = 'a' # append if already exists
            else:
                append_write = 'w' # make a new file if not
            highscore = open(os.path.join(save, 'test_log.txt'),append_write)
            highscore.write(str(valid_loss)+"\t"+str(valid_error) + '\n')
            highscore.close()

            torch.save({
            'state_dict': model.state_dict(),
            'best_error': best_error,
            'best_loss':valid_loss,
            'epoch': epoch + 1,
            } ,os.path.join(save, 'chk.th'))
            with open(os.path.join(save, 'epoch.txt'), 'w') as handle:
                handle.write(str(epoch))

        # Final test of model on test set
        model.load_state_dict(torch.load(os.path.join(save, 'model.th'))['state_dict'])
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model).cuda()
        test_results = test_epoch(
            model=model,
            loader=test_loader,
            is_test=True
        )
        _, _, test_error = test_results
        with open(os.path.join(save, 'results.csv'), 'a') as f:
            f.write(',,,,,%0.5f\n' % (test_error))
        print('Final test error: %.4f' % test_error)


def demo(data, save, depth=100, growth_rate=12, efficient=True, valid_size=5000,
         n_epochs=300, batch_size=64, seed=None, model_type="", dataset='cifar10', version=0, Mense=4, prune=False):
    """
    A demo to show off training of efficient DenseNets.
    Trains and evaluates a DenseNet-BC on CIFAR-10.

    Args:
        data (str) - path to directory where data should be loaded from/downloaded
            (default $DATA_DIR)
        save (str) - path to save the model to (default /tmp)

        depth (int) - depth of the network (number of convolution layers) (default 40)
        growth_rate (int) - number of features added per DenseNet layer (default 12)
        efficient (bool) - use the memory efficient implementation? (default True)

        valid_size (int) - size of validation set
        n_epochs (int) - number of epochs for training (default 300)
        batch_size (int) - size of minibatch (default 256)
        seed (int) - manually set the random seed (default None)
    """
    # parser.add_argument('--model_type', dest='model_type',
                        # help='ien normal maxout base',
                        # default='ien', type=str)       
    # parser.add_argument('-ver', '--version', default=0, type=int, == seed here
                        # help='Version')    
    # parser.add_argument('-m', '--Mense', default=4, type=int,
                        # help='number of ensembles ')   
                        
    if model_type == "normal":
        from models.densenet.densenet import DenseNet 
    elif model_type == "ien":
        from models.densenet.densenet_ien import DenseNet 
    elif model_type == "maxout":
        from models.densenet.densenet_maxout import DenseNet 
    elif model_type == "ien_nn":
        from models.densenet.densenet_ien_nn import DenseNet 
        
    print(save)
    #save = save+"_"+str(growth_rate)+"_"+str(depth)+"_"+str(model_type)+"_"+str(seed)+"_"+str(Mense)+"_" 
    save=os.path.join(save, "densenet_"+str(growth_rate)+"_"+str(depth)+"_"+str(model_type)+"_"+str(seed)+"_"+str(Mense)+"_"+str(version))

    # Get densenet configuration
    if (depth - 4) % 3:
        raise Exception('Invalid depth')
    block_config = [(depth - 4) // 6 for _ in range(3)]

    # Data transforms
    mean = [0.5071, 0.4867, 0.4408]
    stdv = [0.2675, 0.2565, 0.2761]
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])
    
    # Datasets
    if dataset == 'cifar10':
        num_classes=10
        train_set = datasets.CIFAR10(data, train=True, transform=train_transforms, download=True)
        test_set = datasets.CIFAR10(data, train=False, transform=test_transforms, download=False)
    elif dataset == 'cifar100':
        num_classes=100
        train_set = datasets.CIFAR100(data, train=True, transform=train_transforms, download=True)
        test_set = datasets.CIFAR100(data, train=False, transform=test_transforms, download=False)
    


    if valid_size:
        if dataset == 'cifar10':
            valid_set = datasets.CIFAR10(data, train=True, transform=test_transforms)
        elif dataset == 'cifar100':
            valid_set = datasets.CIFAR100(data, train=True, transform=test_transforms)
        indices = torch.randperm(len(train_set))
        train_indices = indices[:len(indices) - valid_size]
        valid_indices = indices[len(indices) - valid_size:]
        train_set = torch.utils.data.Subset(train_set, train_indices)
        valid_set = torch.utils.data.Subset(valid_set, valid_indices)
    else:
        valid_set = None
    

    
    # Models
    if model_type == "normal":

        model = DenseNet(
            growth_rate=growth_rate,
            block_config=block_config,
            num_init_features=growth_rate*2,
            num_classes=num_classes,
            small_inputs=True,
            efficient=efficient,
        )
    else:
        model = DenseNet(
            growth_rate=growth_rate,
            block_config=block_config,
            num_init_features=growth_rate*2,
            num_classes=num_classes,
            small_inputs=True,
            efficient=efficient,M=Mense
        )
    #print(model)
    
    # Print number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print("Total parameters: ", num_params)

    # Make save directory
    if not os.path.exists(save):
        os.makedirs(save)
    if not os.path.isdir(save):
        raise Exception('%s is not a dir' % save)

    # Train the model
    train(model=model, train_set=train_set, valid_set=valid_set, test_set=test_set, save=save,
          n_epochs=n_epochs, batch_size=batch_size, seed=seed, model_type=model_type, prune=prune)
    print('Done!')


"""
A demo to show off training of efficient DenseNets.
Trains and evaluates a DenseNet-BC on CIFAR-10.

Try out the efficient DenseNet implementation:
python demo.py --efficient True --data <path_to_data_dir> --save <path_to_save_dir>

Try out the naive DenseNet implementation:
python demo.py --efficient False --data <path_to_data_dir> --save <path_to_save_dir>

Other args:
    --depth (int) - depth of the network (number of convolution layers) (default 40)
    --growth_rate (int) - number of features added per DenseNet layer (default 12)
    --n_epochs (int) - number of epochs for training (default 300)
    --batch_size (int) - size of minibatch (default 256)
    --seed (int) - manually set the random seed (default None)
    --model_type - normal ien maxout ien_nn
    --Mense - Number of ensembles
"""
#CUDA_VISIBLE_DEVICES=0 python demo.py --data ./data --depth 40 --growth_rate 12 --seed 9999 --model_type normal --Mense 4 --save ./testX --efficient True 

#CUDA_VISIBLE_DEVICES=0 python  demo.py --data ./data --depth 100 --growth_rate 12 --seed 9999 --model_type maxout --Mense 4 --save ./testX --efficient True 
if __name__ == '__main__':
    fire.Fire(demo)
