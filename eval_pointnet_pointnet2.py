import os
import logging
import argparse
import importlib

import torch
import numpy as np
from tqdm import tqdm

from pointnet_pointnet2.PathPlanDataLoader import PathPlanDataset


classes = ['other free points', 'optimal path points']
NUM_CLASSES = len(classes)
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--dim', type=int, default=2, help='environment dimension: 2 or 3.')
    parser.add_argument('--model', type=str, default='pointnet2', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', default=100, type=int, help='Epoch to run [default: 100]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int, default=2048, help='Point Number [default: 4096]')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--random_seed', type=int, default=None)
    parser.add_argument('--save_inference', action='store_true', default=False, help='save inference results')
    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)
    
    model_name = args.model+'_'+str(args.dim)+'d'
    experiment_dir = os.path.join('results/model_training', model_name)
    checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')
    log_dir = os.path.join(experiment_dir, 'logs')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    args = parse_args()
    if args.dim != 2 and args.dim != 3:
        raise ValueError('Invalid dimension: %s.' % args.dim)
    if args.random_seed is not None:
        print("Setting random seed to {0}".format(args.random_seed))
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
    else:
        print("Random seed not set")
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, model_name))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)
    log_string("saving to "+experiment_dir)
    
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size

    env_type = 'random_'+str(args.dim)+'d'
    print("env_type: ", env_type)

    TRAIN_DATASET = PathPlanDataset(dataset_filepath='data/'+env_type+'/train.npz')
    weights = torch.Tensor(TRAIN_DATASET.labelweights).cuda()

    TEST_DATASET = PathPlanDataset(dataset_filepath='data/'+env_type+'/test.npz')
    testDataLoader = torch.utils.data.DataLoader(
        TEST_DATASET,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=10,
        drop_last=False,
    )
    log_string("The number of test data is: %d" % len(testDataLoader))

    MODEL = importlib.import_module('pointnet_pointnet2.models.'+args.model)
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    checkpoint = torch.load(experiment_dir+'/checkpoints/best_'+model_name+'.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()
    
    criterion = MODEL.get_loss().cuda()
    if args.save_inference:
        token_list = []
        pred_list = []
        xyz_list = []
        features_list = []
        labels_list = []
        scores_list = []
    
    with torch.no_grad():
        num_batches = len(testDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        labelweights = np.zeros(NUM_CLASSES)
        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
        classifier = classifier.eval()

        for i, batch in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
            # pc_xyz, pc_features, pc_labels = batch # (b, N, 3), (b, N, 3), (b, N)
            # pc_xyz, pc_features, pc_labels, token = batch
            pc_xyz_raw, pc_xyz, pc_features, pc_labels, token = batch
            
            pc_xyz = pc_xyz.data.numpy()
            pc_xyz = torch.Tensor(pc_xyz)
            points = torch.cat([pc_xyz, pc_features], dim=2) # (b, N, 6)
            points, target = points.float().cuda(), pc_labels.long().cuda()

            points = points.transpose(2, 1)

            seg_pred, trans_feat = classifier(points)
            pred_val = seg_pred.contiguous().cpu().data.numpy()
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

            batch_label = target.cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]
            scores = np.exp(pred_val)/(np.exp(pred_val).sum(-1)[:,:,np.newaxis]) # (1,2048,2)
            
            loss = criterion(seg_pred, target, trans_feat, weights)
            loss_sum += loss
            pred_val = np.argmax(pred_val, 2)
            correct = np.sum((pred_val == batch_label))
            total_correct += correct
            total_seen += (BATCH_SIZE * NUM_POINT)
            tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
            labelweights += tmp

            for l in range(NUM_CLASSES):
                total_seen_class[l] += np.sum((batch_label == l))
                total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))

            if args.save_inference:
                token_list.append(token)
                pred_list.append(pred_val)
                xyz_list.append(pc_xyz_raw)
                features_list.append(pc_features)
                labels_list.append(pc_labels)
                scores_list.append(scores)

        labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
        mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float32) + 1e-6))
        log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
        log_string('eval point avg class IoU: %f' % (mIoU))
        log_string('eval point accuracy: %f' % (total_correct / float(total_seen)))
        log_string('eval point avg class acc: %f' % (
            np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float32) + 1e-6))))

        iou_per_class_str = '------- IoU --------\n'
        for l in range(NUM_CLASSES):
            iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
                seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), labelweights[l - 1],
                total_correct_class[l] / float(total_iou_deno_class[l]))

        log_string(iou_per_class_str)
        log_string('Eval mean loss: %.3f' % (loss_sum / num_batches))
        log_string('Eval accuracy: %.3f' % (total_correct / float(total_seen)))
        log_string('Best Optimal Path IoU: %.3f' % (total_correct_class[1] / float(total_iou_deno_class[1])))

        if args.save_inference:
            results = {}
            results['label'] = torch.cat(labels_list, dim=0).data.numpy()
            results['features'] = torch.cat(features_list, dim=0).data.numpy()
            results['xyz'] = torch.cat(xyz_list, dim=0).data.numpy()
            results['pred'] = np.concatenate(pred_list, axis=0) #(N, 2048)
            results['token'] = token_list
            results['score'] = np.concatenate(scores_list, axis=0) #(N, 2048, 2)
            os.makedirs('visualization', exist_ok=True)
            torch.save(results, 'visualization/'+model_name+'_inference_results.pt')


if __name__ == '__main__':
    args = parse_args()
    main(args)
