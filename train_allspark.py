import argparse
import logging
import os
import pprint

import numpy as np
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from dataset.semi import SemiDataset
from dataset.semi_weak import SemiDatasetWeak
from train_baseline_sup import evaluate
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed
from model.model_helper import ModelBuilder
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
# parser.add_argument('--port', default=None, type=int, required=False)

def entropy_loss(pred, batch_size, entropy_bool):
    if not entropy_bool:
        return torch.tensor(0)

    prob_x = torch.softmax(pred, dim=1)
    entropy_unlabeled_map = torch.sum(-prob_x * torch.log(prob_x + 1e-8), dim=1)
    sum_entropy_per_batch = torch.sum(entropy_unlabeled_map, dim=(1, 2))
    total_entropy_sum = torch.sum(sum_entropy_per_batch)

    total_pixels = entropy_unlabeled_map.size(1) * entropy_unlabeled_map.size(2)

    loss_entropy = total_entropy_sum * (1/total_pixels) * (1/batch_size)
    return loss_entropy

def mask_reconstruction_loss(true_label_weak_masked, pseudo_label_weak_masked, pred_x_masked, pred_u_masked, mask, mask_reconstruction_bool):
    if not mask_reconstruction_bool:
        return torch.tensor(0)
    true_label_weak_masked_ = true_label_weak_masked.cpu() * (1 - mask.squeeze(dim=1))
    pseudo_label_weak_masked_ = pseudo_label_weak_masked.cpu() * (1 - mask.squeeze(dim=1))
    pred_x_masked_ = pred_x_masked.cpu() * (1 - mask)
    pred_u_masked_ = pred_u_masked.cpu() * (1 - mask)


    # plt.imshow(mask[0,0,:,:])
    # plt.show()
    #
    # true_label_weak_masked_np = true_label_weak_masked[0].cpu().detach().numpy()
    # pseudo_label_weak_masked_np = pseudo_label_weak_masked[0].cpu().detach().numpy()
    # pred_x_masked_np = pred_x_masked[0, 0, :].cpu().detach().numpy()
    # pred_u_masked_np = pred_u_masked[0, 0, :].cpu().detach().numpy()

    # Plot side by side

    # fig, axes = plt.subplots(2, 2)
    #
    # axes[0, 0].imshow(true_label_weak_masked_np, cmap='viridis')
    # axes[0, 0].set_title('True Label Weak Masked')
    #
    # axes[0, 1].imshow(pseudo_label_weak_masked_np, cmap='viridis')
    # axes[0, 1].set_title('Pseudo Label Weak Masked')
    #
    # axes[1, 0].imshow(pred_x_masked_np, cmap='viridis')
    # axes[1, 0].set_title('Pred X Masked')
    #
    # axes[1, 1].imshow(pred_u_masked_np, cmap='viridis')
    # axes[1, 1].set_title('Pred U Masked')
    #
    # # Hide the axes
    # for ax in axes.flatten():
    #     ax.axis('off')
    #
    # plt.show()
    #
    # true_label_weak_masked_np = true_label_weak_masked_[0].cpu().detach().numpy()
    # pseudo_label_weak_masked_np = pseudo_label_weak_masked_[0].cpu().detach().numpy()
    # pred_x_masked_np = pred_x_masked_[0, 0, :].cpu().detach().numpy()
    # pred_u_masked_np = pred_u_masked_[0, 0, :].cpu().detach().numpy()
    #
    # # Plot side by side
    #
    # fig, axes = plt.subplots(2, 2)
    #
    # axes[0, 0].imshow(true_label_weak_masked_np, cmap='viridis')
    # axes[0, 0].set_title('True Label Weak Masked')
    #
    # axes[0, 1].imshow(pseudo_label_weak_masked_np, cmap='viridis')
    # axes[0, 1].set_title('Pseudo Label Weak Masked')
    #
    # axes[1, 0].imshow(pred_x_masked_np, cmap='viridis')
    # axes[1, 0].set_title('Pred X Masked')
    #
    # axes[1, 1].imshow(pred_u_masked_np, cmap='viridis')
    # axes[1, 1].set_title('Pred U Masked')
    #
    # # Hide the axes
    # for ax in axes.flatten():
    #     ax.axis('off')
    #
    # plt.show()
    # exit()

    loss = nn.CrossEntropyLoss(ignore_index=255).cuda()
    loss_mask_reconstruction_l = loss(pred_x_masked_, true_label_weak_masked_)
    loss_mask_reconstruction_u = loss(pred_u_masked_, pseudo_label_weak_masked_)
    return (loss_mask_reconstruction_l + loss_mask_reconstruction_u) / 2

def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    # rank, world_size = setup_distributed(port=args.port)
    rank = 0
    world_size = 1

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        
        save_path = args.save_path
        os.makedirs(save_path, exist_ok=True)
        writer = SummaryWriter(save_path)


    cudnn.enabled = True
    cudnn.benchmark = True

    model = ModelBuilder(cfg['model'])
    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                     {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                      'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)

    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    local_rank = 0
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) #OG
    model.cuda()

    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], #OG
    #                                                   output_device=local_rank, find_unused_parameters=False)

    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    criterion_u = nn.CrossEntropyLoss(**cfg['criterion_u']['kwargs']).cuda(local_rank)
    
    trainset_u = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                             cfg['crop_size'], args.unlabeled_id_path)
    trainset_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                             cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
    trainset_u_weak = SemiDatasetWeak(cfg['dataset'], cfg['data_root'], 'train_u',
                            cfg['crop_size'], args.unlabeled_id_path)
    trainset_l_weak = SemiDatasetWeak(cfg['dataset'], cfg['data_root'], 'train_l',
                            cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u_weak.ids))
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')

    # trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True)
    trainloader_l_weak = DataLoader(trainset_l_weak, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True)
    # trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True)
    trainloader_u_weak = DataLoader(trainset_u_weak, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True)
    # valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False)

    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best = 0.0
    best_epoch = 0
    epoch = -1

    if os.path.exists(os.path.join(save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(save_path, 'latest.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        best_epoch = checkpoint['best_epoch']
        previous_best = checkpoint['previous_best']

        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)

    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f} in Epoch {:}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best, best_epoch))

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_u = AverageMeter()

        # trainloader_l.sampler.set_epoch(epoch)
        # trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u, trainloader_l_weak, trainloader_u_weak)

        model.decoder.set_SMem_status(epoch=epoch, isVal=False)

        for i, ((img_x, mask_x), img_u_s, (img_x_weak, mask_x_weak), img_u_s_weak) in enumerate(loader):

            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_s = img_u_s.cuda()

            with torch.no_grad():
                model.eval()
                pred_u_pseudo = model(img_u_s).detach()
                model.decoder.set_pseudo_prob_map(pred_u_pseudo)
                pseudo_label = pred_u_pseudo.argmax(dim=1)
                model.decoder.set_pseudo_label(pseudo_label)


            model.train()
            loss_mask = 0
            if cfg["multi-tasks"]["mask"]:
                mask = torch.randint(0, 2, (img_u_s_weak.shape[0], 1, img_u_s_weak.shape[2], img_u_s_weak.shape[3]))
                img_u_weak_masked = img_u_s_weak * mask
                img_x_weak_masked = img_x_weak * mask
                num_lb_masked, num_ulb_masked = img_x_weak_masked.shape[0], img_u_weak_masked.shape[0]

                preds_masked = model(torch.cat((img_x_weak_masked, img_u_weak_masked)).cuda())
                pred_x_masked, pred_u_masked = preds_masked.split([num_lb_masked, num_ulb_masked])

                pseudo_label_weak_masked = model(img_u_s_weak.cuda()).detach()
                model.decoder.set_pseudo_prob_map(pseudo_label_weak_masked)
                pseudo_label_weak_masked = pseudo_label_weak_masked.argmax(dim=1)
                model.decoder.set_pseudo_label(pseudo_label_weak_masked)

                loss_mask = mask_reconstruction_loss(mask_x_weak, pseudo_label_weak_masked, pred_x_masked,
                                                     pred_u_masked, mask, cfg["multi-tasks"]["mask"]["bool"])
                del preds_masked
                del pred_x_masked, pred_u_masked
                del pseudo_label_weak_masked


            num_lb, num_ulb = img_x.shape[0], img_u_s.shape[0]
            preds = model(torch.cat((img_x, img_u_s)))
            pred_x, pred_u = preds.split([num_lb, num_ulb])

            loss_entropy = entropy_loss(pred_u, cfg["batch_size"], cfg["multi-tasks"]["entropy"]["bool"])

            loss_x = criterion_l(pred_x, mask_x)
            loss_u = criterion_u(pred_u, pseudo_label) + cfg["multi-tasks"]["entropy"]["loss_factor"] * loss_entropy + cfg["multi-tasks"]["mask"]["loss_factor"] * loss_mask
            
            loss = (loss_x + loss_u) / 2.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_u.update(loss_u.item())

            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']

            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss_x.item(), iters)
                writer.add_scalar('train/loss_u', loss_u.item(), iters)
                writer.add_scalar('train/loss_entropy', loss_entropy.item(), iters)
                writer.add_scalar('train/loss_mask', loss_mask.item(), iters)

            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, Loss u: {:.3f}'
                            .format(i, total_loss.avg, total_loss_x.avg, total_loss_u.avg))
        model.decoder.set_SMem_status(epoch=epoch, isVal=True)
        eval_mode = 'sliding_window' if (cfg['dataset'] == 'cityscapes' or cfg['dataset'] == 'potsdam') else 'original'
        mIoU, iou_class, loss_val = evaluate(model, valloader, eval_mode, cfg, criterion_l)

        if rank == 0:
            writer.add_scalar('train/loss_val', loss_val.item(), epoch)
            logger.info(f"Validation loss: {loss_val.item()}")
            for (cls_idx, iou) in enumerate(iou_class):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                            'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
            logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format(eval_mode, mIoU))

            writer.add_scalar('eval/mIoU', mIoU, epoch)
            for i, iou in enumerate(iou_class):
                writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)

        is_best = mIoU > previous_best
        previous_best = max(mIoU, previous_best)
        if is_best:
            best_epoch = epoch
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_epoch': best_epoch,
                'previous_best': previous_best,
            }
            torch.save(checkpoint, os.path.join(save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(save_path, 'best.pth'))


if __name__ == '__main__':
    main()
