import time
import os
import numpy as np
import torch

from toolbox import utils, metrics, plotter


def train(args, train_loader, model, criterion, optimizer, logger, epoch,
          eval_score=None, print_freq=10, tb_writer=None):

    # switch to train mode
    model.train()
    meters = logger.reset_meters('train')
    meters_params = logger.reset_meters('hyperparams')
    meters_params['learning_rate'].update(optimizer.param_groups[0]['lr'])
    end = time.time()

    for i, (input, target_class, name) in enumerate(train_loader):
        # print(f'{i} - {input.size()} - {target_class.size()}')
        batch_size = input.size(0)

        # measure data loading time
        meters['data_time'].update(time.time() - end, n=batch_size)

        input, target_class = input.to(
            args.device).requires_grad_(), target_class.to(args.device)
        output = model(input)

        loss = criterion(output, target_class)

        meters['loss'].update(loss.data.item(), n=batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        if eval_score is not None:
            acc1, pred, label = eval_score(output, target_class)
            meters['acc1'].update(acc1, n=batch_size)
            meters['confusion_matrix'].update(
                pred.squeeze(), label.type(torch.LongTensor))

        # measure elapsed time
        meters['batch_time'].update(time.time() - end, n=batch_size)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'LR {lr.val:.2e}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=meters['batch_time'],
                      data_time=meters['data_time'], lr=meters_params['learning_rate'], loss=meters['loss'], top1=meters['acc1']))

        if True == args.short_run:
            if 12 == i:
                print(' --- running in short-run mode: leaving epoch earlier ---')
                break

    if args.tensorboard:
        tb_writer.add_scalar('acc1/train', meters['acc1'].avg, epoch)
        tb_writer.add_scalar('loss/train', meters['loss'].avg, epoch)
        tb_writer.add_scalar(
            'learning rate', meters_params['learning_rate'].val, epoch)

    logger.log_meters('train', n=epoch)
    logger.log_meters('hyperparams', n=epoch)


def validate(args, val_loader, model, criterion, logger, epoch, eval_score=None, print_freq=10, tb_writer=None):

    # switch to evaluate mode
    model.eval()
    meters = logger.reset_meters('val')
    end = time.time()
    hist = np.zeros((args.num_classes, args.num_classes))
    res_list = {}
    grid_pred = None
    with torch.no_grad():
        for i, (input, target_class, name) in enumerate(val_loader):
            batch_size = input.size(0)

            meters['data_time'].update(time.time()-end, n=batch_size)

            label = target_class.numpy()

            input, target_class = input.to(
                args.device).requires_grad_(), target_class.to(args.device)

            output = model(input)

            loss = criterion(output, target_class)
            meters['loss'].update(loss.data.item(), n=batch_size)

            # measure accuracy and record loss
            if eval_score is not None:
                acc1, pred, buff_label = eval_score(output, target_class)
                meters['acc1'].update(acc1, n=batch_size)
                meters['confusion_matrix'].update(
                    pred.squeeze(), buff_label.type(torch.LongTensor))

                _, pred = torch.max(output, 1)

                for idx, curr_name in enumerate(name):
                    res_list[curr_name.item()] = [pred[idx].item(),
                                                  target_class[idx].item()]

                pred = pred.to('cpu').data.numpy()
                hist += metrics.fast_hist(pred.flatten(),
                                          label.flatten(), args.num_classes)
                mean_ap = round(np.nanmean(
                    metrics.per_class_iu(hist)) * 100, 2)
                meters['mAP'].update(mean_ap, n=batch_size)

            # measure elapsed time
            end = time.time()
            meters['batch_time'].update(time.time() - end, n=batch_size)

            # save samples from first mini-batch for qualitative visualization
            if i == 0:
                pass
                # utils.save_res_grid

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {score.val:.3f} ({score.avg:.3f})'.format(
                          i, len(val_loader), batch_time=meters['batch_time'], loss=meters['loss'],
                          score=meters['acc1']), flush=True)

            if True == args.short_run:
                if 12 == i:
                    print(' --- running in short-run mode: leaving epoch earlier ---')
                    break

    acc, acc_cls, mean_iu, fwavacc = metrics.evaluate(hist)
    meters['acc_class'].update(acc_cls)
    meters['meanIoU'].update(mean_iu)
    meters['fwavacc'].update(fwavacc)

    print(' * Validation set: Average loss {:.4f}, Accuracy {:.3f}%, Accuracy per class {:.3f}%, meanIoU {:.3f}%, \
            fwavacc {:.3f}% \n'.format(meters['loss'].avg, meters['acc1'].avg, meters['acc_class'].val,
                                       meters['meanIoU'].val, meters['fwavacc'].val))

    logger.log_meters('val', n=epoch)
    if args.verbose:
        print(res_list)
    utils.save_res_list(res_list, os.path.join(
        args.res_dir, 'val_results_list_ep{}.json'.format(epoch)))

    if args.tensorboard:
        tb_writer.add_scalar('acc1/val', meters['acc1'].avg, epoch)
        tb_writer.add_scalar('loss/val', meters['loss'].avg, epoch)
        tb_writer.add_scalar('mAP/val', meters['mAP'].avg, epoch)
        tb_writer.add_scalar('acc_class/val', meters['acc_class'].val, epoch)
        tb_writer.add_scalar('meanIoU/val', meters['meanIoU'].val, epoch)
        tb_writer.add_scalar('fwavacc/val', meters['fwavacc'].val, epoch)

    return meters, meters['loss'].avg, res_list


def test(args, eval_data_loader, model, criterion, epoch, eval_score=None,
         output_dir='pred', has_gt=True, tb_writer=None, print_freq=10):

    model.eval()
    meters = metrics.make_meters(args.num_classes)
    end = time.time()
    hist = np.zeros((args.num_classes, args.num_classes))
    res_list = {}
    scores = {}
    with torch.no_grad():
        for i, (input, target_class, name) in enumerate(eval_data_loader):
            # print(input.size())
            batch_size = input.size(0)
            meters['data_time'].update(time.time()-end, n=batch_size)

            label = target_class.numpy()
            input, target_class = input.to(
                args.device).requires_grad_(), target_class.to(args.device)

            output = model(input)

            loss = criterion(output, target_class)

            meters['loss'].update(loss.data.item(), n=batch_size)

            # measure accuracy and record loss
            if eval_score is not None:
                acc1, pred, buff_label = eval_score(output, target_class)
                meters['acc1'].update(acc1, n=batch_size)
                meters['confusion_matrix'].update(
                    pred.squeeze(), buff_label.type(torch.LongTensor))

                _, pred = torch.max(output, 1)

                for idx, curr_name in enumerate(name):
                    res_list[curr_name.item()] = [pred[idx].item(),
                                                  target_class[idx].item()]
                    scores[curr_name.item()] = np.exp(output[idx][0].item())

                pred = pred.to('cpu').data.numpy()
                hist += metrics.fast_hist(pred.flatten(),
                                          label.flatten(), args.num_classes)
                mean_ap = round(np.nanmean(
                    metrics.per_class_iu(hist)) * 100, 2)
                meters['mAP'].update(mean_ap, n=batch_size)

            end = time.time()
            meters['batch_time'].update(time.time() - end, n=batch_size)

            end = time.time()
            if i % print_freq == 0:
                print('Testing: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {score.val:.3f} ({score.avg:.3f})'.format(
                          i, len(eval_data_loader), batch_time=meters['batch_time'], loss=meters['loss'],
                          score=meters['acc1']), flush=True)

            if True == args.short_run:
                if 12 == i:
                    print(' --- running in short-run mode: leaving epoch earlier ---')
                    break

    if eval_score is not None:
        acc, acc_cls, mean_iu, fwavacc = metrics.evaluate(hist)
        meters['acc_class'].update(acc_cls)
        meters['meanIoU'].update(mean_iu)
        meters['fwavacc'].update(fwavacc)

        print(' * Test set: Average loss {:.4f}, Accuracy {:.3f}%, Accuracy per class {:.3f}%, meanIoU {:.3f}%, fwavacc {:.3f}% \n'.format(
            meters['loss'].avg, meters['acc1'].avg, meters['acc_class'].val, meters['meanIoU'].val, meters['fwavacc'].val))

    metrics.save_meters(meters, os.path.join(
        output_dir, 'test_results_ep{}.json'.format(epoch)), epoch)
    utils.save_res_list(res_list, os.path.join(
        output_dir, 'test_results_list_ep{}.json'.format(epoch)))

    # TODO: add class names
    cm = np.array(meters["confusion_matrix"].value())
    plotter.plot_confusion_matrix(cm, os.path.join(
        output_dir, 'norm_cm_ep{}.png'.format(epoch)), normalize=True, tb_writer=tb_writer)
    plotter.plot_confusion_matrix(cm, os.path.join(
        output_dir, 'cm_ep{}.png'.format(epoch)))

    if args.num_classes == 2:
        prob_scores = [i for i in scores.values()]
        ground_truth = [i[1] for i in res_list.values()]
        plotter.plot_roc_curve(ground_truth, prob_scores, os.path.join(
            output_dir, 'roc_ep{}.png'.format(epoch)), tb_writer=tb_writer)
        tb_writer.add_pr_curve('pr_curve', np.array(
            ground_truth), np.array(prob_scores))
