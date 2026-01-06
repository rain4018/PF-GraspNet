import argparse
import logging
import torch.utils.data

from inference.post_process import post_process_output
from hardware.device import get_device
from utils.data import get_dataset
from utils.dataset_processing import evaluation, grasp
from utils.visualisation.plot import save_results

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate networks')

    # Network
    parser.add_argument('--network', metavar='N', type=str, nargs='+',
                        help='Path to saved networks to evaluate')
    parser.add_argument('--input-size', type=int, default=300,
                        help='Input image size for the network')

    # Dataset
    parser.add_argument('--dataset', type=str,
                        help='Dataset Name ("cornell" or "jaquard")')
    parser.add_argument('--dataset-path', type=str,
                        help='Path to dataset')
    parser.add_argument('--use-depth', type=int, default=1,
                        help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1,
                        help='Use RGB image for evaluation (1/0)')
    parser.add_argument('--augment', action='store_true',
                        help='Whether data augmentation should be applied')
    parser.add_argument('--split', type=float, default=0.9,
                        help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-shuffle', action='store_true', default=False,
                        help='Shuffle the dataset')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Dataset workers')
    parser.add_argument('--scene-id', type=int,
                        help='target-id')

    # Evaluation
    parser.add_argument('--n-grasps', type=int, default=5,
                        help='Number of grasps to consider per image')
    parser.add_argument('--iou-threshold', type=float, default=0.25,
                        help='Threshold for IOU matching')
    parser.add_argument('--iou-eval', action='store_true',
                        help='Compute success based on IoU metric.')

    # Misc.
    parser.add_argument('--vis', action='store_true',
                        help='Visualise the network output')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')
    parser.add_argument('--random-seed', type=int, default=4018,
                        help='Random seed for numpy')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    # Get the compute device
    device = get_device(args.force_cpu)

    # Load Dataset
    logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    Dataset = get_dataset(args.dataset)

    val_dataset = Dataset(args.dataset_path,
                          output_size=args.input_size,
                          ds_rotate=args.ds_rotate,
                          include_depth=args.use_depth,
                          include_rgb=args.use_rgb,
                          split='val',
                          scene_id=args.scene_id)

    logging.info('Validation size: {}'.format(val_dataset.length))

    val_data = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=args.num_workers
    )
    logging.info('Done')

    for network in args.network:
        logging.info('\nEvaluating model {}'.format(network))

        # Load Network
        net = torch.load(network)

        results = {'correct': 0, 'failed': 0}

        with torch.no_grad():
            for idx, (x, y, didx, rot) in enumerate(val_data):
                xc = x.to(device)
                yc = [yi.to(device) for yi in y]
                lossd = net.compute_loss(xc, yc)

                q_img, ang_img, width_img = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                                lossd['pred']['sin'], lossd['pred']['width'])

                if args.iou_eval:
                    s = evaluation.calculate_iou_match(q_img, ang_img, val_data.dataset.get_gtbb(didx, rot[0]),
                                                       no_grasps=args.n_grasps,
                                                       grasp_width=width_img,
                                                       threshold=args.iou_threshold
                                                       )

                    results['correct'] += s['correct']
                    results['failed'] += s['failed']

                if args.vis:
                    save_results(
                        rgb_img=val_data.dataset.get_rgb(didx, rot, normalise=False),
                        depth_img=val_data.dataset.get_depth(didx, rot),
                        grasp_q_img=q_img,
                        grasp_angle_img=ang_img,
                        no_grasps=args.n_grasps,
                        grasp_width_img=width_img
                    )

        if args.iou_eval:
            logging.info('IOU Results: %d/%d = %f' % (results['correct'],
                                                      results['correct'] + results['failed'],
                                                      results['correct'] / (results['correct'] + results['failed'])))

        del net
        torch.cuda.empty_cache()
