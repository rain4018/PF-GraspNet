import argparse
import datetime
import json
import logging
import os
import sys
import math
import cv2
import tensorboardX
import torch
import torch.optim as optim
import torch.utils.data

from torchsummary import summary
from utils.data.cbrgd_data import CBRGDDataset
from hardware.device import get_device
from inference.models.AGRNet import AGRNet
from inference.post_process import post_process_output
from utils.dataset_processing import evaluation
from utils.visualisation.gridshow import gridshow


def cosine_attn_weight(epoch, max_epoch=30):
    if epoch >= max_epoch:
        return 1.0
    progress = epoch / max_epoch
    return 0.5 * (1 - math.cos(math.pi * progress)) * (1.0 - 0.05) + 0.05


def parse_args():
    parser = argparse.ArgumentParser(description='Train network')

    # Network
    parser.add_argument('--input-size', type=int, default=300,
                        help='Input image size for the network')
    parser.add_argument('--use-depth', type=int, default=1,
                        help='Use Depth image for training (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1,
                        help='Use RGB image for training (1/0)')
    parser.add_argument('--use-dropout', type=int, default=1,
                        help='Use dropout for training (1/0)')
    parser.add_argument('--dropout-prob', type=float, default=0.1,
                        help='Dropout prob for training (0-1)')
    parser.add_argument('--channel-size', type=int, default=32,
                        help='Internal channel size for the network')
    parser.add_argument('--iou-threshold', type=float, default=0.25,
                        help='Threshold for IOU matching')

    # Datasets
    parser.add_argument('--dataset-path', type=str,
                        help='Path to dataset')
    parser.add_argument('--ds-shuffle', action='store_true', default=False,
                        help='Shuffle the dataset')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Dataset workers')
    parser.add_argument('--scene-id', type=int, help='scene-id')

    # Training
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs')
    parser.add_argument('--batches-per-epoch', type=int, default=1000,
                        help='Batches per Epoch')
    parser.add_argument('--optim', type=str, default='adam',
                        help='Optmizer for the training. (adam or SGD)')

    # Logging etc.
    parser.add_argument('--description', type=str, default='',
                        help='Training description')
    parser.add_argument('--logdir', type=str, default='logs/',
                        help='Log directory')
    parser.add_argument('--vis', action='store_true',
                        help='Visualise the training process')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')
    parser.add_argument('--random-seed', type=int, default=4018,
                        help='Random seed for numpy')

    args = parser.parse_args()
    return args


def validate(net, device, val_data):
    """
    Run validation.
    :param net: Network
    :param device: Torch device
    :param val_data: Validation Dataset
    :return: Successes, Failures and Losses
    """
    net.eval()

    results = {
        'correct5': 0,
        'failed5': 0,
        'loss': 0,
        'losses': {

        }
    }

    ld = len(val_data)

    with torch.no_grad():
        for x, y, didx, rot in val_data:
            xc = x.to(device)
            yc = [yy.to(device) for yy in y]
            lossd = net.compute_loss(xc, yc)

            loss = lossd['loss']

            results['loss'] += loss.item() / ld
            for ln, l in lossd['losses'].items():
                if ln not in results['losses']:
                    results['losses'][ln] = 0
                results['losses'][ln] += l.item() / ld

            q_out, ang_out, w_out = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                        lossd['pred']['sin'], lossd['pred']['width'])

            s5 = evaluation.calculate_iou_match(q_out, ang_out,
                                                val_data.dataset.get_gtbb(didx, rot[0]),
                                                no_grasps=5,
                                                grasp_width=w_out,
                                                )

            results['correct5'] += s5['correct']
            results['failed5'] += s5['failed']

    return results


def train(epoch, net, device, train_data, optimizer, batches_per_epoch, vis=False):
    """
    Run one training epoch
    :param epoch: Current epoch
    :param net: Network
    :param device: Torch device
    :param train_data: Training Dataset
    :param optimizer: Optimizer
    :param batches_per_epoch:  Data batches to train on
    :param vis:  Visualise training progress
    :return:  Average Losses for Epoch
    """
    results = {
        'loss': 0,
        'losses': {
        }
    }

    net.train()

    batch_idx = 0
    # Use batches per epoch to make training on different sized datasets (cornell/jacquard) more equivalent.
    while batch_idx <= batches_per_epoch:
        for x, y, _, _ in train_data:
            batch_idx += 1
            if batch_idx >= batches_per_epoch:
                break

            xc = x.to(device)
            yc = [yy.to(device) for yy in y]
            lossd = net.compute_loss(xc, yc)

            loss = lossd['loss']

            if batch_idx % 100 == 0:
                logging.info('Epoch: {}, Batch: {}, Loss: {:0.4f}'.format(epoch, batch_idx, loss.item()))

            results['loss'] += loss.item()
            for ln, l in lossd['losses'].items():
                if ln not in results['losses']:
                    results['losses'][ln] = 0
                results['losses'][ln] += l.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Display the images
            if vis:
                imgs = []
                n_img = min(4, x.shape[0])
                for idx in range(n_img):
                    imgs.extend([x[idx,].numpy().squeeze()] + [yi[idx,].numpy().squeeze() for yi in y] + [
                        x[idx,].numpy().squeeze()] + [pc[idx,].detach().cpu().numpy().squeeze() for pc in
                                                      lossd['pred'].values()])
                gridshow('Display', imgs,
                         [(xc.min().item(), xc.max().item()), (0.0, 1.0), (0.0, 1.0), (-1.0, 1.0),
                          (0.0, 1.0)] * 2 * n_img,
                         [cv2.COLORMAP_BONE] * 10 * n_img, 10)
                cv2.waitKey(2)

    results['loss'] /= batch_idx
    for l in results['losses']:
        results['losses'][l] /= batch_idx

    return results


def run():
    args = parse_args()

    # Set-up output directories
    dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
    net_desc = '{}_{}'.format(dt, '_'.join(args.description.split()))

    save_folder = os.path.join(args.logdir, net_desc)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    tb = tensorboardX.SummaryWriter(save_folder)

    # Save commandline args
    if args is not None:
        params_path = os.path.join(save_folder, 'commandline_args.json')
        with open(params_path, 'w') as f:
            json.dump(vars(args), f)

    # Initialize logging
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        filename="{0}/{1}.log".format(save_folder, 'log'),
        format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)

    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)

    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    # Get the compute device
    device = get_device(args.force_cpu)

    # Load Dataset
    logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    train_dataset = CBRGDDataset(args.dataset_path,
                      output_size=args.input_size,
                      ds_rotate=args.ds_rotate,
                      random_rotate=True,
                      include_depth=args.use_depth,
                      include_rgb=args.use_rgb,
                      scene_id=args.scene_id)

    val_dataset = CBRGDDataset(args.dataset_path,
                            output_size=args.input_size,
                            ds_rotate=args.ds_rotate,
                            random_rotate=False,
                            include_depth=args.use_depth,
                            include_rgb=args.use_rgb,
                            split='val',
                            scene_id=args.scene_id)


    logging.info('Training size: {}'.format(train_dataset.length))
    logging.info('Validation size: {}'.format(val_dataset.length))

    train_data = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True
    )
    val_data = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=args.num_workers,
    )
    logging.info('Done')

    # Load the network
    logging.info('Loading Network...')
    input_channels = 1 * args.use_depth + 3 * args.use_rgb
    net = AGRNet(
        input_channels=input_channels,
        dropout=args.use_dropout,
        prob=args.dropout_prob,
        channel_size=args.channel_size
    )

    net = net.to(device)
    logging.info('Done')

    if args.optim.lower() == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=0.001)
    elif args.optim.lower() == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    else:
        raise NotImplementedError('Optimizer {} is not implemented'.format(args.optim))

    # Print model architecture.
    summary(net, (input_channels, args.input_size, args.input_size))
    f = open(os.path.join(save_folder, 'arch.txt'), 'w')
    sys.stdout = f
    summary(net, (input_channels, args.input_size, args.input_size))
    sys.stdout = sys.__stdout__
    f.close()

    best_iou = 0.0
    for epoch in range(args.epochs):
        net.attn_weight = cosine_attn_weight(epoch)

        logging.info('Beginning Epoch {:02d}'.format(epoch))
        train_results = train(epoch, net, device, train_data, optimizer, args.batches_per_epoch, vis=args.vis)

        # Log training losses to tensorboard
        tb.add_scalar('loss/train_loss', train_results['loss'], epoch)
        for n, l in train_results['losses'].items():
            tb.add_scalar('train_loss/' + n, l, epoch)

        # Run Validation
        logging.info('Validating...')
        test_results = validate(net, device, val_data)

        total5 = test_results['correct5'] + test_results['failed5']
        if total5 == 0:
            acc5 = 0.0
        else:
            acc5 = test_results['correct5'] / total5
        logging.info('top5：%d/%d = %f' % (test_results['correct5'], total5, acc5))

        # Log validation results to tensorbaord
        tb.add_scalar('loss/top_5', acc5, epoch)
        tb.add_scalar('loss/val_loss', test_results['loss'], epoch)
        for n, l in test_results['losses'].items():
            tb.add_scalar('val_loss/' + n, l, epoch)
        tb.add_scalar('train/attn_weight', net.attn_weight, global_step=epoch)

        # Save best performing network
        if acc5 > best_iou or epoch == 0 or (epoch % 10) == 0:
            torch.save(net, os.path.join(save_folder, 'epoch_%02d_top5_%0.4f' %(epoch, acc5)))
            best_iou = acc5


if __name__ == '__main__':
    run()
