# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import importlib
import json
import os
import sys
import time
import random
import numpy as np
import psutil
import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter

from data_utils import ShakespeareDataset, WikiTextDataset, OpenWebTextDataset, BookCorpusDataset, DataUtil
from transformers import AutoTokenizer

sys.path.append("..")
from runtime import runtime
from optim import adamw
from optim import nadamw

parser = argparse.ArgumentParser(description='AsyncPP Language Model Training')
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
parser.add_argument('--momentum_decay', default=0.004, type=float,
                    help='momentum decay for nadamw')
parser.add_argument('--adaptive_momentum', action='store_true',
                    help='Adaptive momentum')
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
# Only LR correction in the optimizer step, with weight stashing
# if used together with ascyn-backprop, no weight stashing
parser.add_argument('--lr_correction', action='store_true',
                    help='LR scaling for delayed updates, no other correction')
parser.add_argument('--lr_correction_epoch', default=7, type=int,
                    help='LR scaling max epoch for delayed updates')
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
parser.add_argument('--clip_grad', default=None, type=float,
                    help='Clip gradient norm using this value')
parser.add_argument('--deterministic', action='store_true',
                    help='Deterministic training, results are reproducible')
parser.add_argument('--optimizer', default="adamw", type=str,
                    help='Optimizer ["adamw", "nadamw"]')
# model
parser.add_argument('--block_size', default=256, type=int,
                    metavar='N', help='block size (default: 256)')
parser.add_argument('--n_embd', default=384, type=int,
                    metavar='N', help='embedding dimension (default: 384)')
parser.add_argument('--n_head', default=6, type=int,
                    metavar='N', help='number of heads (default: 6)')
parser.add_argument('--n_layer', default=6, type=int,
                    metavar='N', help='number of layers (default: 6)')

best_loss = 100
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

def main():
    global args, best_loss, _tb
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
    criterion = nn.CrossEntropyLoss()
    
    # create stages of the model
    module = importlib.import_module(args.module)
    args.arch = module.arch()
    if args.arch == "gptn":
        model = module.model(criterion, vocab_size=vocab_size, block_size=args.block_size, 
                         n_embd=args.n_embd, n_head=args.n_head, n_layer=args.n_layer)
    else:
        raise Exception("Invalid model name")

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

    eval_tensor_shapes = {}
    for key in training_tensor_shapes:
        eval_tensor_shapes[key] = tuple(
            [args.eval_batch_size] + training_tensor_shapes[key][1:])
        training_tensor_shapes[key] = tuple(
            training_tensor_shapes[key])

    configuration_maps = {
        'module_to_stage_map': None,
        'stage_to_rank_map': None,
        'stage_to_depth_map': None
    }
    if args.config_path is not None:
        with open(args.config_path, 'r') as f:
            json_config_file = json.load(f)
        configuration_maps['module_to_stage_map'] = json_config_file.get("module_to_stage_map", None)
        configuration_maps['stage_to_rank_map'] = json_config_file.get("stage_to_rank_map", None)
        configuration_maps['stage_to_rank_map'] = {
            int(k): v for (k, v) in configuration_maps['stage_to_rank_map'].items()}
        configuration_maps['stage_to_depth_map'] = json_config_file.get("stage_to_depth_map", None)
    else:
        # create sequential stage map
        configuration_maps['module_to_stage_map'] = [i for i in range(args.n_layer)] + [args.n_layer-1]
        configuration_maps['stage_to_rank_map'] = {i: [i] for i in range(args.n_layer)}

    r = runtime.StageRuntime(
        model=model, distributed_backend=args.distributed_backend,
        fp16=args.fp16, loss_scale=args.loss_scale,
        training_tensor_shapes=training_tensor_shapes,
        eval_tensor_shapes=eval_tensor_shapes,
        training_tensor_dtypes=dtypes,
        inputs_module_destinations=inputs_module_destinations,
        target_tensor_names=target_tensor_names,
        configuration_maps=configuration_maps,
        master_addr=args.master_addr, master_port=args.master_port,
        rank=args.rank, local_rank=args.local_rank,
        num_ranks_in_server=args.num_ranks_in_server,
        verbose_freq=args.verbose_frequency,
        model_type=runtime.LANGUAGE_MODELING,
        enable_recompute=args.recompute)

    # stage needed to determine if current stage is the first stage
    # num_stages needed to determine if current stage is the last stage
    # num_ranks needed to determine number of warmup_minibatches in case of pipelining
    args.stage = r.stage
    args.num_stages = r.num_stages
    args.num_ranks = r.num_ranks

    args.optim_save_dir = None
    if args.save_weights_to_disk:
        args.optim_save_dir = os.path.join(args.checkpoint_dir, f"weight_stash_{args.rank}")
        os.makedirs(args.optim_save_dir, exist_ok=True) # one per rank

    exp_name = args.exp_name + str(time.time())
    # do it here as stages are set above
    if args.log_tb and is_last_stage():
        _tb = SummaryWriter(log_dir=os.path.join(args.tb_dir, exp_name))
        _tb.add_text('config', json.dumps(vars(args), sort_keys=True, indent=4))


    # define optimizer
    if args.no_input_pipelining:
        num_versions = 1
    else:
        # number of versions is the total number of machines following the current
        # stage, shared amongst all replicas in this stage
        ## todo: check if this is correct when DP is enabled!!
        num_versions = r.num_warmup_minibatches + 1
    print(f"## Stage: {args.stage}, Num versions: {num_versions}")

    # if specified, resume from checkpoint
    if args.resume:
        checkpoint_file_path = "%s.%d.pth.tar" % (args.resume, r.stage)
        assert os.path.isfile(checkpoint_file_path)
        print("=> loading checkpoint '{}'".format(checkpoint_file_path))
        checkpoint = torch.load(checkpoint_file_path, weights_only=False)
        args.start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        r.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}) (best_epoch {}, best_loss {})"
                .format(checkpoint_file_path, checkpoint['epoch'], checkpoint['best_epoch'], best_loss))

    if args.adaptive_momentum:
        args.momentum = args.momentum + (num_versions - 1) * (0.99 - args.momentum) / args.num_stages
        assert args.momentum <= 1.0 and args.momentum >= 0.0

    if args.optimizer == "adamw":
        optimizer = adamw.AdamWWithWeightStashing(r.modules(), r.master_parameters,
                                          r.model_parameters, loss_scale=args.loss_scale,
                                          num_versions=num_versions,
                                          lr=args.lr, betas=(args.momentum, 0.999),
                                          weight_decay=args.weight_decay,
                                          verbose_freq=args.verbose_frequency,
                                          macrobatch=args.macrobatch,
                                          clip_grad=args.clip_grad, save_dir=args.optim_save_dir,
                                          stash_to_cpu=args.stash_to_cpu)
    elif args.optimizer == "nadamw":
        optimizer = nadamw.NAdamWithWeightStashing(r.modules(), r.master_parameters,
                                          r.model_parameters, loss_scale=args.loss_scale,
                                          num_versions=num_versions,
                                          lr=args.lr, betas=(args.momentum, 0.999), momentum_decay=args.momentum_decay,
                                          weight_decay=args.weight_decay,
                                          verbose_freq=args.verbose_frequency,
                                          macrobatch=args.macrobatch, 
                                          clip_grad=args.clip_grad, save_dir=args.optim_save_dir,
                                          stash_to_cpu=args.stash_to_cpu)
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
        train_dataset, batch_size=args.batch_size, 
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True,
        worker_init_fn=init_fn)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.eval_batch_size, 
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=True,
        worker_init_fn=init_fn)

    du = DataUtil(train_loader, val_loader)

    # if checkpoint is loaded, start by running validation
    if args.resume:
        assert args.start_epoch > 0
        validate(val_loader, r, args.start_epoch-1, du)

    for epoch in range(args.start_epoch, args.epochs):
        if distributed_sampler:
            train_sampler.set_epoch(epoch)

        # train or run forward pass only for one epoch
        if args.forward_only:
            validate(val_loader, r, epoch, du)
        else:
            train(train_loader, r, optimizer, epoch, du)

            # evaluate on validation set
            loss = validate(val_loader, r, epoch, du)
            if not is_last_stage(): loss = 100

            should_save_checkpoint = args.checkpoint_dir_not_nfs or r.rank_in_stage == 0

            # remember best prec@1 and save checkpoint
            # saving best checkpoint for all stages is a pain, because we need to communicate with all other ranks!
            if best_loss > loss:
                best_loss = loss               

            # also save the current checkpoint
            if args.checkpoint_dir and should_save_checkpoint:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': r.state_dict(),
                    'best_loss': best_loss,
                    'best_epoch': epoch+1,
                    'optimizer' : optimizer.state_dict(),
                }, args.checkpoint_dir, r.stage)

            if _tb is not None:
                _tb.add_scalar('best_loss/val', best_loss, epoch)
                _tb.flush()

    if _tb is not None:
        _tb.close()


def train(train_loader, r, optimizer, epoch, du):
    batch_time = AverageMeter()
    losses = AverageMeter()
    # switch to train mode
    n = r.num_iterations(loader_size=len(train_loader.dataset) // args.batch_size)
    if args.num_minibatches is not None:
        n = min(n, args.num_minibatches)
    r.train(n)
    r.set_batch_sampler(du.get_batch if is_first_stage() else None, is_eval=False)

    # reset weight stashes
    optimizer.initialize_queue()

    end = time.time()
    epoch_start_time = time.time()

    if args.no_input_pipelining:
        num_warmup_minibatches = 0
    else:
        num_warmup_minibatches = r.num_warmup_minibatches

    if args.verbose_frequency > 0:
        print("Letting in %d warm-up minibatches" % num_warmup_minibatches)
        print("Running training for %d minibatches" % n)

    # start num_warmup_minibatches forward passes
    for i in range(num_warmup_minibatches):
        r.run_forward()

    for i in range(n - num_warmup_minibatches):
        # perform forward pass
        r.run_forward()

        # Adjust learning rate
        adjust_learning_rate(optimizer, epoch, args.epochs, r, args.lr_policy, i, n, 
                             args.lr_decay_interval, args.lr_correction, args.lr_correction_epoch)

        if is_last_stage():
            # measure accuracy and record loss
            output, target, loss = r.output, r.target, r.loss
            losses.update(loss.item(), output.size(0))

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

        # perform backward pass
        if args.fp16:
            r.zero_grad()
        else:
            optimizer.zero_grad()
        # load weights for backprop
        optimizer.load_old_params()
        r.run_backward()
        # load weights for optimizer step
        optimizer.load_new_params()
        optimizer.step()

    # finish remaining backward passes
    for i in range(num_warmup_minibatches):
        optimizer.zero_grad()
        # load weights for backprop
        optimizer.load_old_params()
        r.run_backward()
        # load weights for optimizer step
        optimizer.load_new_params()
        optimizer.step()

    # wait for all helper threads to complete
    r.wait()

    print("Epoch %d: %.3f seconds" % (epoch, time.time() - epoch_start_time))
    print("Epoch start time: %.3f, epoch end time: %.3f" % (epoch_start_time, time.time()))

    if _tb is not None:
        _tb.add_scalar('loss/train', losses.avg, epoch)
        _tb.add_scalar('cpu_mem/train', psutil.virtual_memory().percent, epoch)
        _tb.flush()


def validate(val_loader, r, epoch, du):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    n = r.num_iterations(loader_size=len(val_loader.dataset) // args.eval_batch_size)
    if args.num_eval_minibatches is not None:
        n = min(n, args.num_eval_minibatches)
    r.eval(n)
    r.set_batch_sampler(du.get_batch if is_first_stage() else None, is_eval=True)

    end = time.time()
    epoch_start_time = time.time()

    if args.no_input_pipelining:
        num_warmup_minibatches = 0
    else:
        num_warmup_minibatches = r.num_warmup_minibatches

    if args.verbose_frequency > 0:
        print("Letting in %d warm-up minibatches" % num_warmup_minibatches)
        print("Running validation for %d minibatches" % n)

    with torch.no_grad():
        for i in range(num_warmup_minibatches):
            r.run_forward()

        for i in range(n - num_warmup_minibatches):
            # perform forward pass
            r.run_forward()
            r.run_ack()

            if is_last_stage():
                output, target, loss = r.output, r.target, r.loss

                # measure and record loss
                losses.update(loss.item(), output.size(0))
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

        for i in range(num_warmup_minibatches):
            r.run_ack()

        # wait for all helper threads to complete
        r.wait()

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


def adjust_learning_rate(optimizer, epoch, total_epochs, r, lr_policy, step, epoch_length, 
                         decay_interval=30, lr_correction=False, lr_correction_epoch=8):
    """ Adjusts learning rate based on stage, epoch, and policy.

    Gets learning rate for stage from runtime and adjusts based on policy.

    Supported LR policies:
         - step
         - polynomial decay
         - exponential decay
    """
    stage_base_lr = r.get_adjusted_learning_rate(base_lr=args.lr)

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
        os.system("pkill -f main_with_runtime.py")
