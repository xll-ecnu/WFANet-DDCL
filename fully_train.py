import argparse
import logging
import random
import sys
import imageio
from PIL import Image
from torch import optim
from torch.utils.data.distributed import DistributedSampler
#from utils.distributed import DistributedSampler
from torch.utils.data.sampler import BatchSampler
#from utils.sampler import BatchSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import torch.backends.cudnn as cudnn
import os
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
from torchvision.utils import make_grid
from tqdm import tqdm
import torch.multiprocessing as mp
from ptb.utils import ptb_f
import torch
from dataloaders.dataset import (data_process,  Resize_2d,
                                 ToTensor, TwoStreamBatchSampler
                                 )
from networks.net_factory import net_factory_3d
from utils import ramps
from val import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--semi', type=str,
                    default=True, help='semi or fully')
parser.add_argument('--root_path', type=str,
                    default='./data/paired10', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='WFANet', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='WFANet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=3000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--image_size', type=list,  default=[224, 224],
                    help='image size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
# label and unlabel
parser.add_argument('--paired_bs', type=int, default=2,
                    help='labeled_batch_size per gpu')
parser.add_argument('--paired_num', type=int, default=10,
                    help='paired data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0,1])
parser.add_argument('--multi_gpu', action='store_true', default=False
                    )
parser.add_argument('--port', type=int, default=12361)
args = parser.parse_args()

device = args.device
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in args.gpu_ids])
print(os.environ["CUDA_VISIBLE_DEVICES"])

def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://127.0.0.1:{args.port}',
        world_size=world_size,
        rank=rank
    )

def cleanup():
    dist.destroy_process_group()

def device_config(args):
    try:
        if not args.multi_gpu:
            # Single GPU
            if args.device == 'mps':
                args.device = torch.device('mps')
            else:
                args.device = torch.device(f"cuda:{args.gpu_ids[0]}")
        else:
            args.nodes = 1
            args.ngpus_per_node = len(args.gpu_ids)
            args.world_size = args.nodes * args.ngpus_per_node

    except RuntimeError as e:
        print(e)

def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(args.seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(args):
    snapshot_path = "../model/{}_{}/{}".format(
        args.exp, args.paired_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    semi = args.semi
    base_lr = args.base_lr
    train_data_path = args.root_path
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    def create_model(ema=False):
        # Network definition
        net = net_factory_3d(net_type=args.model)
        #print('net:',net)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    if args.multi_gpu and semi:
        model = create_model()
        ema_model = create_model(ema=True)
        model = DDP(model, device_ids=[args.rank], output_device=args.rank,find_unused_parameters=True)
        #ema_model = DDP(ema_model, device_ids=[args.rank], output_device=args.rank)
    elif args.multi_gpu and not semi:
        model = create_model()
        model = DDP(model, device_ids=[args.rank], output_device=args.rank,find_unused_parameters=True)
        #print('model:',model)
        #ema_model = create_model(ema=True)
    else:
        model = create_model()
        ema_model = create_model(ema=True)


    db_train = data_process(base_dir=train_data_path,
                         split='train',
                         transform=transforms.Compose([
                             #RandomRotFlip(),
                             #CenterCrop(args.patch_size),
                             Resize_2d(args.image_size),
                             ToTensor(),
                         ]))
    paired_lenth, unpaired_lenth = db_train.get_data_lenth()
    image_list = db_train.get_image_list()
    print('paired_lenth',paired_lenth)
    print('unpaired_lenth',unpaired_lenth)
    print('image_list',image_list)
    print('lenth of image_list',len(image_list))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    paired_idxs = list(range(0, paired_lenth))
    unpaired_idxs = list(range(paired_lenth, unpaired_lenth))

    batch_sampler = TwoStreamBatchSampler(
        paired_idxs, unpaired_idxs, batch_size, batch_size-args.paired_bs)
    #print(batch_sampler)
    #if args.multi_gpu:
    #batch_sampler = DistributedSampler(db_train)
    #train_batch_sampler = BatchSampler(batch_sampler, batch_size, drop_last=False)
    # else:
    train_batch_sampler = batch_sampler

    trainloader = DataLoader(db_train, batch_sampler=train_batch_sampler,
                             num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()
    if semi:
        ema_model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)

    criterion_L1 = nn.L1Loss()
    criterion_L2 = nn.MSELoss()
    #criterion_L2 = criterion_L2.cuda()
    ce_loss = CrossEntropyLoss()
    #dice_loss = losses.DiceLoss(2)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            print('i_batch:',i_batch)
            i3_batch, i7_batch = sampled_batch['paired_3T'], sampled_batch['paired_7T']
            #print('i3_batch.shape',i3_batch.shape)
            #print('i7_batch.shape',i7_batch.shape)
            for k in range(len(i7_batch)):
                image_copy = (torch.squeeze(i3_batch[k])).cpu().numpy()
                image_copy = np.uint8((image_copy*255))
                #image_copy = image_copy.transpose(1,2,0)
                imageio.imwrite('./test/{}_image_{}.png'.format(i_batch, k), image_copy)
                label_copy = i7_batch[k].cpu().numpy()
                label_copy = np.uint8((label_copy*255))
                imageio.imwrite('./test/{}_label_{}.png'.format(i_batch, k), label_copy)
            i7_batch = torch.unsqueeze(i7_batch, 1)
            # print('i7_batch.shape',i7_batch.shape)
            i3_batch, i7_batch = i3_batch.cuda(), i7_batch.cuda()
            unlabeled_i3_batch = i3_batch[args.paired_bs:]
            #print('unlabeled_i3_batch.shape', unlabeled_i3_batch.shape)
            noise = torch.clamp(torch.randn_like(
               unlabeled_i3_batch) * 0.1, -0.2, 0.2)

            ema_inputs_n1 = unlabeled_i3_batch + noise
            ema_inputs_n2 = ptb_f(unlabeled_i3_batch)
            #print('ema_inputs_n1.shape', ema_inputs_n1.shape)
            #print('ema_inputs_n2.shape', ema_inputs_n2.shape)

            model.train()

            outputs = model(i3_batch)
            #print('outputs.shape', outputs.shape)
            with torch.no_grad():
               ema_output_n1 = ema_model(ema_inputs_n1)
               ema_output_n2 = ema_model(ema_inputs_n2)

            supervised_loss = criterion_L1(torch.squeeze(outputs[:args.paired_bs]), torch.squeeze(i7_batch[:args.paired_bs]))

            consistency_weight = get_current_consistency_weight(iter_num//150)

            consistency_loss_1 = torch.mean(
                (outputs[args.paired_bs:] - ema_output_n1) ** 2)
            consistency_loss_2 = torch.mean(
               (outputs[args.paired_bs:] - ema_output_n2) ** 2)
            consistency_loss = 0.4 * consistency_loss_1 + 0.6 * consistency_loss_2

            loss = supervised_loss +  2 * consistency_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_supervised', supervised_loss, iter_num)
            writer.add_scalar('info/consistency_loss',
                             consistency_loss, iter_num)
            writer.add_scalar('info/consistency_weight',
                             consistency_weight, iter_num)
            logging.info(
                'iteration %d : loss : %f, loss_supervised: %f' %
                (iter_num, loss.item(), supervised_loss.item()))
            writer.add_scalar('loss/loss', loss, iter_num)

            if iter_num % 10 == 0:
                image = i3_batch[0, 0:1, :, :].permute(
                    0, 1, 2).repeat(3, 1, 1)

                grid_image = make_grid(image, 5, normalize=True)

                writer.add_image('train/Image', grid_image, iter_num)

                image = outputs[0, 0:1, :, :].permute(
                    0, 1, 2).repeat(3, 1, 1)

                grid_image = make_grid(image, 5, normalize=False)

                writer.add_image('train/Predicted_label',
                                 grid_image, iter_num)


                image = i7_batch[0, 0:1, :, :].permute(
                    0, 1, 2).repeat(3, 1, 1)

                grid_image = make_grid(image, 5, normalize=False)

                writer.add_image('train/Groundtruth_label',
                                 grid_image, iter_num)

            if iter_num > 0 and iter_num % 2 == 0:
                model.eval()
                avg_metric = test_all_case(
                    model, args.root_path, snapshot_path, test_list="val.txt", output_size=args.image_size)
                print(avg_metric)
                if avg_metric[:, 0].mean() > best_performance:
                    best_performance = avg_metric[:, 0].mean()
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_ssim_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                writer.add_scalar('info/val_ssim',
                                  avg_metric[0, 0], iter_num)
                writer.add_scalar('info/val_psnr',
                                  avg_metric[0, 1], iter_num)
                logging.info(
                    'iteration %d : ssim : %f psnr : %f' % (iter_num, avg_metric[0, 0].mean(), avg_metric[0, 1].mean()))
                model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"

def main(args):
    mp.set_sharing_strategy('file_system')
    device_config(args)
    if args.multi_gpu:
        mp.spawn(
            main_worker,
            nprocs=args.world_size,
            args=(args, )
        )
    else:
        random.seed(2023)
        np.random.seed(2023)
        torch.manual_seed(2023)
        # Train
        train(args)


def main_worker(rank, args):
    setup(rank, args.world_size)
    print('rank:',rank)
    print('ngpus_per_node:',args.ngpus_per_node)
    torch.cuda.set_device(rank)
    args.num_workers = int(args.num_workers / args.ngpus_per_node)
    args.device = torch.device(f"cuda:{rank}")
    args.rank = rank

    init_seeds(2023 + rank)
    train(args)
    cleanup()


if __name__ == "__main__":
    snapshot_path = "../model/{}_{}/{}".format(
        args.exp, args.paired_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    main(args)
