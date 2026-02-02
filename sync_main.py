# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
from collections import OrderedDict
import importlib
import json
import os
import shutil
import sys
import time
import random
import numpy as np
import psutil
import math
import collections

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter
import torch.distributed.pipelining as pp

from data_utils import ShakespeareDataset, WikiTextDataset, OpenWebTextDataset, BookCorpusDataset, DataUtil
from transformers import AutoTokenizer

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data_dir', '-dd', type=str, default='~/data',
                    help='path to dataset')
parser.add_argument('--dataset_name', '-d', type=str,
                    help='name of dataset')
parser.add_argument('--distributed_backend', type=str,
                    help='distributed backend to use (gloo|nccl)')
parser.add_argument('--module', '-m', required=True,
                    help='name of module that contains model and tensor_shapes definition')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--num_microbatches', default=None, type=int,
                    help="Number of microbatches per batch")
parser.add_argument('--eval-batch-size', default=100, type=int,
                    help='eval mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_policy', default='step', type=str,
                    help='policy for controlling learning rate')
parser.add_argument('--lr_decay_interval', default=30, type=int,
                    help='LR decay interval when policy is step')
parser.add_argument('--lr_warmup', action='store_true',
                    help='Warmup learning rate first 3 epochs')
parser.add_argument('--lr_warmup_epochs', default=3, type=int,
                    help='Warmup learning rate first 3 epochs')
parser.add_argument('--lr_warmup_init', default=1e-7, type=float,
                    help='Initial learning rate for warmup')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0.01, type=float,
                    metavar='W', help='weight decay (default: 0.01)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--fp16', action='store_true',
                    help='train model in fp16 precision')
parser.add_argument('--loss_scale', type=float, default=1,
                    help='static loss scale, positive power of 2 to improve fp16 convergence')
parser.add_argument('--master_addr', default=None, type=str,
                    help="IP address of master (machine with rank 0)")
parser.add_argument('--master_port', default="12345", type=str,
                    help="Port of master (machine with rank 0)")                    
parser.add_argument('--config_path', default=None, type=str,
                    help="Path of configuration file")
parser.add_argument('--no_input_pipelining', action='store_true',
                    help="No pipelining of inputs")
parser.add_argument('--rank', default=None, type=int,
                    help="Rank of worker")
parser.add_argument('--local_rank', default=0, type=int,
                    help="Local rank of worker")
parser.add_argument('--forward_only', action='store_true',
                    help="Run forward pass only")
parser.add_argument('--num_minibatches', default=None, type=int,
                    help="Number of minibatches to run")
parser.add_argument('--num_eval_minibatches', default=None, type=int,
                    help="Number of minibatches to run for validation")
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--checkpoint_dir', default='', type=str, metavar='PATH',
                    help='path to directory to save checkpoints')
parser.add_argument('--checkpoint_dir_not_nfs', action='store_true',
                    help='checkpoint dir is not on a shared NFS server')
parser.add_argument('-s', '--synthetic_data', action='store_true',
                    help="Use synthetic data")
parser.add_argument('-v', '--verbose_frequency', default=0, type=int, metavar='N',
                    help="Log verbose information")
parser.add_argument('--num_ranks_in_server', default=1, type=int,
                    help="number of gpus per machine")
# enable-recompute -- to make sure tensor versions are correct for backward!
# forward will be computed again before backward
parser.add_argument('--recompute', action='store_true',
                    help='Recompute tensors in backward pass')
# Macrobatching reduces the number of weight versions to save,
# by not applying updates every minibatch.
parser.add_argument('--macrobatch', action='store_true',
                    help='Macrobatch updates to save memory')
## Pipamare related changes
# Only async backprop, no weight stashing
parser.add_argument('--async_backprop', action='store_true',
                    help='Async backpropogation as in PipeMare, no other correction')
# Only LR correction in the optimizer step, with weight stashing
# if used together with ascyn-backprop, no weight stashing
parser.add_argument('--lr_correction', action='store_true',
                    help='LR scaling for delayed updates, no other correction')
parser.add_argument('--lr_correction_epoch', default=7, type=int,
                    help='LR scaling max epoch for delayed updates')
parser.add_argument('--pipemare', action='store_true',
                    help='Full PipeMare, including async updates, lr and weight corrctions')
# reporting
parser.add_argument('--log_tb', action='store_true',
                    help='Log to tensorboard')
parser.add_argument('--tb_dir', default='/async-pp/runs/', type=str,
                    help='Tensorboard directory')
parser.add_argument('--exp_name', default='', type=str,
                    help='Experiment name')
parser.add_argument('--save_weights_to_disk', action='store_true',
                    help='Save weights to disk')
parser.add_argument('--stash_to_cpu', action='store_true',
                    help='Stash weights to CPU')
# second order gradient correction
parser.add_argument('--second_order_correction', action='store_true',
                    help='Second order gradient correction')
# use this to specify the hessian approximation method
parser.add_argument('--hessian_approx', default=None, type=str,
                    help='Hessian approximation method ["diagonal", "fisher", "hessian-free"]')
# discount the weights based on the variance of gradient estimates, 
# as in Delay-Compensated async-sgd paper
parser.add_argument('--adaptive_weight', action='store_true',
                    help='Adaptive weight for the second order correction')
parser.add_argument('--grad_estimation_decay', default=0.04, type=float,
                    help='Decay rate for the gradient estimation')
parser.add_argument('--grad_est_correction', action='store_true',
                    help='Downweight the gradient estimation for large delay, follow pipemare style')
parser.add_argument('--grad_est_correction_epoch', default=7, type=int,
                    help='Downweight the gradient estimation for large delay, follow pipemare style, analogous to lr_correction_epoch')
parser.add_argument('--clip_grad', default=None, type=float,
                    help='Clip gradient norm using this value')
parser.add_argument('--deterministic', action='store_true',
                    help='Deterministic training, results are reproducible')
parser.add_argument('--optimizer', default="adamw", type=str,
                    help='Optimizer ["adamw", "sgd"]')
parser.add_argument('--fft_correction', action='store_true',
                    help='Use FFT to correct gradient')
parser.add_argument('--grad_history_len', default=5, type=int,
                    help='Length of gradient history for FFT correction')
# use this to specify the correction method
parser.add_argument('--correction_method', default=None, type=str,
                    help='Correction method ["fft", "poly", "second_order"]')
parser.add_argument('--poly_order', default=2, type=int,
                    help='Order of polynomial for polynomial fit correction')
parser.add_argument('--use_weights_and_time', action='store_true',
                    help='Use weights and time for polynomial fit correction')
parser.add_argument('--adaptive_correction', action='store_true',
                    help='Adaptive correction method for polynomial fit correction, disable if fit-error increases')
# model
parser.add_argument('--block_size', default=256, type=int,
                    metavar='N', help='block size (default: 256)')
parser.add_argument('--n_embd', default=384, type=int,
                    metavar='N', help='embedding dimension (default: 384)')
parser.add_argument('--n_head', default=6, type=int,
                    metavar='N', help='number of heads (default: 6)')
parser.add_argument('--n_layer', default=6, type=int,
                    metavar='N', help='number of layers (default: 6)')
# sync schedule
parser.add_argument('--sync_schedule', default='gpipe', type=str,
                    help='Sync schedule ["gpipe", "1f1b"]')

best_loss = 100
best_fit_err = 100
_tb = None
RANDOM_SEED = 1337

# Helper methods.
def is_first_stage():
    return args.stage is None or (args.stage == 0)

def is_last_stage():
    return args.stage is None or (args.stage == (args.num_stages-1))

def seed_torch(deterministic=False):
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED) # if you are using multi-GPU.
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

def init_fn(worker_id):
   random.seed(RANDOM_SEED + worker_id)
   np.random.seed(RANDOM_SEED + worker_id)

def reshaped_cross_entropy(outputs, targets):
    loss_fn = nn.CrossEntropyLoss()
    sz = targets.numel()
    outputs = outputs.reshape(sz, -1)
    targets = targets.reshape(-1)
    return loss_fn(outputs, targets)

def main():
    global args, best_loss, _tb, best_fit_err
    args = parser.parse_args()
    args.data_dir = os.path.expanduser(args.data_dir)   # to handle ~

    torch.cuda.set_device(args.local_rank)

    # for reproducibility
    seed_torch(args.deterministic)

    #  and create datasets
    if args.dataset_name == "shakespeare":
        train_dataset = ShakespeareDataset(args.data_dir, train=True, block_size=args.block_size)
        val_dataset = ShakespeareDataset(args.data_dir, train=False, block_size=args.block_size)
        vocab_size = train_dataset.vocab_size
    elif args.dataset_name == "wikitext-103-v1":
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as padding token
        train_dataset = WikiTextDataset(args.data_dir, tokenizer=tokenizer, train=True, block_size=args.block_size)
        val_dataset = WikiTextDataset(args.data_dir, tokenizer=tokenizer, train=False, block_size=args.block_size)
        vocab_size = tokenizer.vocab_size
    elif args.dataset_name == "openwebtext":
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as padding token
        train_dataset = OpenWebTextDataset(args.data_dir, tokenizer=tokenizer, train=True, block_size=args.block_size)
        val_dataset = OpenWebTextDataset(args.data_dir, tokenizer=tokenizer, train=False, block_size=args.block_size)
        vocab_size = tokenizer.vocab_size
    elif args.dataset_name == "bookcorpus":
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as padding token
        train_dataset = BookCorpusDataset(args.data_dir, tokenizer=tokenizer, train=True, block_size=args.block_size)
        val_dataset = BookCorpusDataset(args.data_dir, tokenizer=tokenizer, train=False, block_size=args.block_size)
        vocab_size = tokenizer.vocab_size
    else:
        raise Exception("Invalid dataset name")

    # define loss function (criterion)
    criterion = reshaped_cross_entropy
    
    # create stages of the model
    module = importlib.import_module(args.module)
    args.arch = module.arch()
    if args.arch == "gptn":
        model = module.model(criterion, vocab_size=vocab_size, block_size=args.block_size, 
                         n_embd=args.n_embd, n_head=args.n_head, n_layer=args.n_layer)
    else:
        model = module.model(criterion, vocab_size=vocab_size, block_size=args.block_size)

    args.nparams = float(sum(sum(p.numel() for p in s.parameters()) for s, _, _ in model[:-1])) / 1e6
    print(f"#Params: {args.nparams:.2f} M")

    # determine shapes of all tensors in passed-in model
    input_size = [args.batch_size, args.block_size]
    training_tensor_shapes = {"input0": input_size, "target": input_size}
    dtypes = {"input0": torch.int64, "target": torch.int64}
    inputs_module_destinations = {"input": 0}
    target_tensor_names = {"target"}
    for (stage, inputs, outputs) in model[:-1]:  # Skip last layer (loss).
        input_tensors = []
        for input in inputs:
            input_tensor = torch.zeros(tuple(training_tensor_shapes[input]),
                                       dtype=dtypes[input])
            input_tensors.append(input_tensor)
        with torch.no_grad():
            output_tensors = stage(*tuple(input_tensors))
        if not type(output_tensors) is tuple:
            output_tensors = [output_tensors]
        for output, output_tensor in zip(outputs,
                                         list(output_tensors)):
            training_tensor_shapes[output] = list(output_tensor.size())
            dtypes[output] = output_tensor.dtype

    assert args.batch_size == args.eval_batch_size, f"Batch size {args.batch_size} and eval batch size {args.eval_batch_size} must be the same"

    configuration_maps = {
        'module_to_stage_map': None,
        'stage_to_rank_map': None,
        'stage_to_depth_map': None
    }
    if args.config_path is not None:
        json_config_file = json.load(open(args.config_path, 'r'))
        configuration_maps['module_to_stage_map'] = json_config_file.get("module_to_stage_map", None)
        configuration_maps['stage_to_rank_map'] = json_config_file.get("stage_to_rank_map", None)
        configuration_maps['stage_to_rank_map'] = {
            int(k): v for (k, v) in configuration_maps['stage_to_rank_map'].items()}
        configuration_maps['stage_to_depth_map'] = json_config_file.get("stage_to_depth_map", None)
    else:
        # create sequential stage map
        configuration_maps['module_to_stage_map'] = [i for i in range(args.n_layer)] + [args.n_layer-1]
        configuration_maps['stage_to_rank_map'] = {i: [i] for i in range(args.n_layer)}

    ## setup pipeline
    assert len(configuration_maps['module_to_stage_map']) == len(model)
    assert args.rank is not None

    stage_to_module_map = collections.defaultdict(list)
    for module in range(len(configuration_maps['module_to_stage_map'])):
        stage_to_module_map[configuration_maps['module_to_stage_map'][module]].append(module)

    rank_to_stage_map = {}
    for stage in configuration_maps['stage_to_rank_map']:
        for rank in configuration_maps['stage_to_rank_map'][stage]:
            rank_to_stage_map[rank] = stage

    # Now, use this mapping to determine the modules contained in
    # each stage.
    assert 0 <= args.rank < len(rank_to_stage_map)
    args.num_ranks = len(rank_to_stage_map)
    args.num_stages = len(stage_to_module_map)
    args.stage = rank_to_stage_map[args.rank]
    assert args.num_ranks == args.num_stages
    args.macro_batch_size = args.batch_size * args.num_microbatches
    args.num_minibatches = args.num_minibatches // args.num_microbatches
    args.num_eval_minibatches = args.num_eval_minibatches // args.num_microbatches

    # Initialize the distributed environment.
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = str(args.master_port)
    dist.init_process_group(backend=args.distributed_backend, rank=args.rank, world_size=args.num_ranks)
    print("Finished initializing process group; backend: %s, rank: %d, "
              "world_size: %d" % (args.distributed_backend, args.rank, args.num_ranks))
    
    ## setup pipeline
    ## only supports each stage having one module, except the last stage
    assert len(stage_to_module_map[args.stage]) == 1 or is_last_stage()
    module = stage_to_module_map[args.stage][0]
    stage_input_eg = [torch.zeros(tuple(training_tensor_shapes[input]), dtype=dtypes[input], device='cuda') for input in model[module][1]]
    stage_output_eg = [torch.zeros(tuple(training_tensor_shapes[output]), dtype=dtypes[output], device='cuda') for output in model[module][2]]
    pp_stage = pp.PipelineStage(model[module][0], args.rank, args.num_ranks, torch.device('cuda'), stage_input_eg, stage_output_eg)
    # loss_fn is handled separately in schedule!

    if args.sync_schedule == 'gpipe':
        schedule = pp.ScheduleGPipe(pp_stage, args.num_microbatches, loss_fn=criterion)
    elif args.sync_schedule == '1f1b':
        schedule = pp.Schedule1F1B(pp_stage, args.num_microbatches, loss_fn=criterion)
    else:
        raise Exception("Invalid sync schedule")

    exp_name = args.exp_name + str(time.time())
    # do it here as stages are set above
    if args.log_tb and is_last_stage():
        _tb = SummaryWriter(log_dir=os.path.join(args.tb_dir, exp_name))
        _tb.add_text('config', json.dumps(vars(args), sort_keys=True, indent=4))

    # if specified, resume from checkpoint
    if args.resume:
        checkpoint_file_path = "%s.%d.pth.tar" % (args.resume, args.stage)
        assert os.path.isfile(checkpoint_file_path)
        print("=> loading checkpoint '{}'".format(checkpoint_file_path))
        checkpoint = torch.load(checkpoint_file_path)
        args.start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        pp_stage.submod.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}) (best_epoch {}, best_loss {})"
                .format(checkpoint_file_path, checkpoint['epoch'], checkpoint['best_epoch'], best_loss))

    if args.optimizer == "adamw":
        optimizer = optim.AdamW(pp_stage.submod.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(pp_stage.submod.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "nadamw":
        optimizer = optim.NAdam(pp_stage.submod.parameters(), lr=args.lr, weight_decay=args.weight_decay, decoupled_weight_decay=True)
    else:
        raise Exception("Invalid optimizer")

    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])

    ## data loader
    distributed_sampler = False
    train_sampler = None
    val_sampler = None
    if configuration_maps['stage_to_rank_map'] is not None:
        num_ranks_in_first_stage = len(configuration_maps['stage_to_rank_map'][0])
        if num_ranks_in_first_stage > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, num_replicas=num_ranks_in_first_stage,
                rank=args.rank)
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset, num_replicas=num_ranks_in_first_stage,
                rank=args.rank)
            distributed_sampler = True

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.macro_batch_size, 
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True,
        worker_init_fn=init_fn)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.macro_batch_size, 
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=True,
        worker_init_fn=init_fn)

    du = DataUtil(train_loader, val_loader)

    # if checkpoint is loaded, start by running validation
    if args.resume:
        assert args.start_epoch > 0
        validate(val_loader, schedule, args.start_epoch-1, du)

    for epoch in range(args.start_epoch, args.epochs):
        if distributed_sampler:
            train_sampler.set_epoch(epoch)

        # train or run forward pass only for one epoch
        if args.forward_only:
            validate(val_loader, schedule, epoch, du)
        else:
            train(train_loader, schedule, optimizer, epoch, du)

            # evaluate on validation set
            loss = validate(val_loader, schedule, epoch, du)
            if not is_last_stage(): loss = 100

            should_save_checkpoint = True   #args.checkpoint_dir_not_nfs or r.rank_in_stage == 0

            # remember best prec@1 and save checkpoint
            # saving best checkpoint for all stages is a pain, because we need to communicate with all other ranks!
            if best_loss > loss:
                best_loss = loss               

            # also save the current checkpoint
            if args.checkpoint_dir and should_save_checkpoint:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': pp_stage.submod.state_dict(),
                    'best_loss': best_loss,
                    'best_epoch': epoch+1,
                    'optimizer' : optimizer.state_dict(),
                }, args.checkpoint_dir, args.stage)

            if _tb is not None:
                _tb.add_scalar('best_loss/val', best_loss, epoch)
                _tb.flush()

    if _tb is not None:
        _tb.close()

    dist.destroy_process_group()
    

def train(train_loader, schedule, optimizer, epoch, du):
    batch_time = AverageMeter()
    losses = AverageMeter()

    n = min(args.num_minibatches, len(train_loader.dataset) // args.macro_batch_size)
    # loader_iter = iter(train_loader)
    
    # switch to train mode
    # schedule._stage.submod.train()
    
    epoch_start_time = time.time()
    end = time.time()

    for i in range(n):
        # cleanup, since val is done in train mode
        optimizer.zero_grad()
        
        # perform forward pass
        # input, target = next(loader_iter)
        input, target = du.get_batch()
        if is_first_stage():
            schedule.step(input.cuda(non_blocking=True))
        elif is_last_stage():
            loss = []
            schedule.step(target=target.cuda(non_blocking=True), losses=loss)
            losses.update(sum([l.item() for l in loss])/len(loss), len(loss))
        else:
            schedule.step()

        # Adjust learning rate
        adjust_learning_rate(optimizer, epoch, args.epochs, args.lr_policy, i, n, 
                             args.lr_decay_interval, args.lr_correction, args.lr_correction_epoch)

        if is_last_stage():
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            epoch_time = (end - epoch_start_time) / 3600.0
            full_epoch_time = (epoch_time / float(i+1)) * float(n)

            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Epoch time [hr]: {epoch_time:.3f} ({full_epoch_time:.3f})\t'
                      'Memory: {memory:.3f} ({cached_memory:.3f}) ({cpu_memory:.3f})\t'
                      'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                       epoch, i, n, batch_time=batch_time,
                       epoch_time=epoch_time, full_epoch_time=full_epoch_time,
                       loss=losses,
                       memory=(float(torch.cuda.memory_allocated()) / 10**9),
                       cached_memory=(float(torch.cuda.memory_reserved()) / 10**9),
                       cpu_memory=psutil.virtual_memory().percent))

                sys.stdout.flush()
        else:
            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\tMemory: {memory:.3f} ({cached_memory:.3f})'.format(
                       epoch, i, n, memory=(float(torch.cuda.memory_allocated()) / 10**9),
                       cached_memory=(float(torch.cuda.memory_reserved()) / 10**9)))
                sys.stdout.flush()

        # step
        if args.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(schedule._stage.submod.parameters(), max_norm=args.clip_grad)
        optimizer.step()
        # cleanup
        optimizer.zero_grad()

    # wait for all helper threads to complete
    dist.barrier()

    print("Epoch %d: %.3f seconds" % (epoch, time.time() - epoch_start_time))
    print("Epoch start time: %.3f, epoch end time: %.3f" % (epoch_start_time, time.time()))

    if _tb is not None:
        _tb.add_scalar('loss/train', losses.avg, epoch)
        _tb.add_scalar('cpu_mem/train', psutil.virtual_memory().percent, epoch)
        _tb.flush()


def validate(val_loader, schedule, epoch, du):
    batch_time = AverageMeter()
    losses = AverageMeter()

    n = min(args.num_eval_minibatches, len(val_loader.dataset) // args.macro_batch_size)
    # loader_iter = iter(val_loader)
    
    # switch to eval mode
    # schedule._stage.submod.eval()
    
    epoch_start_time = time.time()
    end = time.time()

    # with torch.no_grad():
    for i in range(n):
        # perform forward pass
        # input, target = next(loader_iter)
        input, target = du.get_batch(eval=True)

        if is_first_stage():
            schedule.step(input.cuda(non_blocking=True))
        elif is_last_stage():
            loss = []
            schedule.step(target=target.cuda(non_blocking=True), losses=loss)
            losses.update(sum([l.item() for l in loss])/len(loss), len(loss))
        else:
            schedule.step()

        if is_last_stage():
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}][{1}/{2}]\t'
                        'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Memory: {memory:.3f} ({cached_memory:.3f})\t'
                        'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                        epoch, i, n, batch_time=batch_time, loss=losses,
                        memory=(float(torch.cuda.memory_allocated()) / 10**9),
                        cached_memory=(float(torch.cuda.memory_reserved()) / 10**9)))
                sys.stdout.flush()

    if is_last_stage():
        print(' * Loss {loss.avg:.4f}'.format(loss=losses))
        if _tb is not None:
            _tb.add_scalar('loss/val', losses.avg, epoch)
            _tb.flush()

    # wait for all helper threads to complete
    dist.barrier()

    print('Epoch %d: %.3f seconds' % (epoch, time.time() - epoch_start_time))
    print("Epoch start time: %.3f, epoch end time: %.3f" % (epoch_start_time, time.time()))

    return losses.avg


def save_checkpoint(state, checkpoint_dir, stage, isbest=False):
    assert os.path.isdir(checkpoint_dir)
    checkpoint_file_path = os.path.join(checkpoint_dir, "checkpoint-best.%d.pth.tar" % stage) if isbest else os.path.join(checkpoint_dir, "checkpoint.%d.pth.tar" % stage)
    torch.save(state, checkpoint_file_path)
    print("Saved checkpoint to %s" % checkpoint_file_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
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


def adjust_learning_rate(optimizer, epoch, total_epochs, lr_policy, step, epoch_length, 
                         decay_interval=30, lr_correction=False, lr_correction_epoch=8):
    """ Adjusts learning rate based on stage, epoch, and policy.

    Gets learning rate for stage from runtime and adjusts based on policy.

    Supported LR policies:
         - step
         - polynomial decay
         - exponential decay
    """
    stage_base_lr = args.lr

    if args.lr_warmup and epoch < args.lr_warmup_epochs:
        lr = args.lr_warmup_init + (stage_base_lr - args.lr_warmup_init) * float(1 + step + epoch*epoch_length)/(args.lr_warmup_epochs*epoch_length)

    else:
        if lr_policy == "step":
            lr = stage_base_lr * (0.1 ** (epoch // decay_interval))
        elif lr_policy == "polynomial":
            power = 2.0
            lr = stage_base_lr * ((1.0 - (float(epoch) / float(total_epochs))) ** power)
        elif lr_policy == "exponential_decay":
            decay_rate = 0.97
            lr = stage_base_lr * (decay_rate ** (float(epoch) / float(total_epochs)))
        elif lr_policy == "cosine":
            it = step + epoch*epoch_length
            warmup_it = args.lr_warmup_epochs*epoch_length if args.lr_warmup else 0
            decay_ratio = float(it - warmup_it) / float(total_epochs*epoch_length - warmup_it)
            min_lr = 0.1 * stage_base_lr
            lr = min_lr + (stage_base_lr - min_lr) * (0.5 * (1 + math.cos(math.pi * decay_ratio)))
        else:
            raise NotImplementedError
    
    if lr_correction and optimizer.num_versions > 1: # PipeMare type LR correction when there is delay
        lr = lr / ((optimizer.num_versions-1) ** 
                   (1 - min(float(1 + step + epoch*epoch_length)/(lr_correction_epoch*epoch_length), 1)))


    if step % 100 == 0:
        print("Epoch: %d Step %d \tLearning rate: %f" % (epoch, step, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
        import traceback
        print(traceback.format_exc())
        print("Killing all python processes...")
        sys.stdout.flush()
        os.system("pkill -f sync_main.py")
