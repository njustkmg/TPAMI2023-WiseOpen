import logging
import time
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from utils import AverageMeter, ova_loss,\
    save_checkpoint, test, test_ood, test_upper,\
    exclude_dataset_via_var, cal_Lu, exclude_dataset_via_loss

logger = logging.getLogger(__name__)



def train(args, labeled_trainloader, unlabeled_dataset, test_loader, val_loader,
          ood_loaders, model, optimizer, ema_model, scheduler, best_acc, best_acc_val, best_all_acc, best_acc_roc, best_roc,
          var_list, loss_list, sel_list, unique_labeled_trainloader, dist_align):
    if args.amp:
        from apex import amp

    labels = None
    best_aupr_in_ood_dic = {}
    best_aupr_out_ood_dic = {}
    best_roc_ood_dic = {}
    overall_valid = 0
    close_valid = 0
    unk_valid = 0
    roc_valid = 0
    roc_softm_valid = 0
    aupr_in_valid = 0
    aupr_out_valid = 0
    un_all_cnt = len(unlabeled_dataset)

    grad_dims = []
    for param in model.parameters():
        grad_dims.append(param.data.numel())

    logger.info("global best_acc: {:.4f} | best_acc_val: {:.4f} | best_all_acc: {:.4f} | best_acc_roc: {:.4f} | best_roc: {:.4f}".format(
        best_acc, best_acc_val, best_all_acc, best_acc_roc, best_roc 
    ))
    test_accs = []

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
    labeled_iter = iter(labeled_trainloader)
    default_out = "Epoch: {epoch}/{epochs:4}. " \
                  "LR: {lr:.6f}. " \
                  "Lx: {loss_x:.4f}. " 
    output_args = vars(args)
    default_out += " MB: {loss_mb:.4f}."
    default_out += " OP: {loss_op:.4f}."
    default_out += " UI: {loss_ui:.4f}."

    unlabeled_dataset_sel = copy.deepcopy(unlabeled_dataset)

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    unlabeled_trainloader = DataLoader(unlabeled_dataset,
                                           sampler = train_sampler(unlabeled_dataset),
                                           batch_size = args.batch_size * args.mu,
                                           num_workers = args.num_workers,
                                           drop_last = True)
    unlabeled_iter = iter(unlabeled_trainloader)

    print('unlabeled_trainloader: ',len(unlabeled_dataset))

    for epoch in range(args.start_epoch, args.epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_mb = AverageMeter()
        losses_ui = AverageMeter()
        losses_op = AverageMeter()
        p_mask_probs = AverageMeter()
        q_mask_probs = AverageMeter()
        end = time.time()
        output_args["epoch"] = epoch
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step),
                         disable=args.local_rank not in [-1, 0], ncols=150, ascii=True)

        if args.sel:
            if args.var_th and epoch % args.varth_interval==0:
                all_selected, selected_mask_df, var_all, l_list, labels, th =\
                    exclude_dataset_via_var(args, unique_labeled_trainloader, grad_dims, 
                                            unlabeled_dataset_sel, model, dist_align)
                var_list.append(var_all)
            elif args.loss_th:
                all_selected, selected_mask_df, l_list, labels, th =\
                    exclude_dataset_via_loss(args, unique_labeled_trainloader, grad_dims, 
                                            unlabeled_dataset_sel, model, dist_align)
            loss_list.append(l_list)
            print('loss list size: ({},{})'.format(len(loss_list), len(loss_list[0])))
            sel_list.append(selected_mask_df)
            unlabeled_dataset.init_index()
            unlabeled_dataset.set_index(all_selected)
            args.writer.add_scalar('var/1.th', th, epoch)
            args.writer.add_scalar('data/for_classifier', len(unlabeled_dataset)/un_all_cnt, epoch)
                


        print(len(unlabeled_dataset))
        model.train()
        for batch_idx in range(args.eval_step):
            ## Data loading

            try:
                inputs_x, targets_x = labeled_iter.__next__()
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x = labeled_iter.__next__()

            try:
                (inputs_u_w, inputs_u_s), target_god = unlabeled_iter.__next__()
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s), target_god = unlabeled_iter.__next__()
            target_god = target_god.to(args.device)

            data_time.update(time.time() - end)

            num_lb = inputs_x.shape[0]
            num_ulb = inputs_u_w.shape[0]

            ## 2023.10.20
            inputs = torch.cat([inputs_x, inputs_u_w, inputs_u_s], 0).to(args.device)
            targets_x = targets_x.to(args.device)

            ## Feed data
            logits, logits_mb, logits_open = model(inputs)

            logits_x_lb = logits[:num_lb]
            logits_mb_x_lb = logits_mb[:num_lb]

            ## supervied loss
            L_x = F.cross_entropy(logits_x_lb, targets_x, reduction='mean')
            L_mb = ova_loss(logits_mb_x_lb, targets_x)

            ## unsupervied loss
            L_ui, L_op, p_mask, q_mask = cal_Lu(args, logits[num_lb:], logits_open[num_lb:], logits_mb[num_lb:], num_ulb, dist_align, update=True)

            p_mask_probs.update(p_mask.mean().item())
            q_mask_probs.update(q_mask.mean().item())

            if args.no_op:
                loss = L_x + L_mb + L_ui
            else:
                loss = L_x + L_mb + L_ui + L_op * args.lmbd_op
            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            losses.update(loss.item())
            losses_x.update(L_x.item())
            losses_mb.update(L_mb.item())
            losses_op.update(L_op.item())
            losses_ui.update(L_ui.item())

            output_args["batch"] = batch_idx
            output_args["loss_x"] = losses_x.avg
            output_args["loss_mb"] = losses_mb.avg
            output_args["loss_op"] = losses_op.avg
            output_args["loss_ui"] = losses_ui.avg
            output_args["lr"] = [group["lr"] for group in optimizer.param_groups][0]


            optimizer.step()
            if args.opt != 'adam':
                scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()
            batch_time.update(time.time() - end)
            end = time.time()

            if not args.no_progress:
                p_bar.set_description(default_out.format(**output_args))
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:
            val_acc = test(args, val_loader, test_model, epoch, val=True)
            test_loss, test_acc_close, test_overall, \
            test_unk, test_roc , test_roc_softm, test_id, \
            test_aupr_in, test_aupr_out \
                = test(args, test_loader, test_model, epoch)
            aupr_in_ood_dic = {}
            aupr_out_ood_dic = {}
            roc_ood_dic = {}
            if args.ood_test:
                for ood in ood_loaders.keys():
                    roc_ood, aupr_in_ood, aupr_out_ood = test_ood(args, test_id, ood_loaders[ood], test_model)
                    aupr_in_ood_dic[ood] = aupr_in_ood
                    aupr_out_ood_dic[ood] = aupr_out_ood
                    roc_ood_dic[ood] = roc_ood
                    logger.info("ROC vs {ood}: {roc}".format(ood=ood, roc=roc_ood))
                    logger.info("AUPR(in) vs {ood}: {aupr_in}".format(ood=ood, aupr_in=aupr_in_ood))
                    logger.info("AUPR(out) vs {ood}: {aupr_out}".format(ood=ood, aupr_out=aupr_out_ood))

            args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
            args.writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
            args.writer.add_scalar('train/2.train_loss_mb', losses_mb.avg, epoch)
            args.writer.add_scalar('train/3.train_loss_op', losses_op.avg, epoch)
            args.writer.add_scalar('train/4.train_loss_ui', losses_ui.avg, epoch)
            args.writer.add_scalar('train/5.mask', p_mask_probs.avg, epoch)
            args.writer.add_scalar('train/6.mask', q_mask_probs.avg, epoch)

            args.writer.add_scalar('test/1.test_acc', test_acc_close, epoch)
            args.writer.add_scalar('test/2.test_loss', test_loss, epoch)
            args.writer.add_scalar('test/3.roc', test_roc, epoch)
            args.writer.add_scalar('val/1.acc', val_acc, epoch)

            args.writer.add_scalar('hyper/1.lr', [group["lr"] for group in optimizer.param_groups][0], epoch)
            args.writer.add_scalar('hyper/2.p_cutoff', args.p_cutoff, epoch)
            args.writer.add_scalar('hyper/3.q_cutoff', args.q_cutoff, epoch)

            is_best = val_acc > best_acc_val
            is_all_best = test_overall > best_all_acc
            is_roc_best = test_roc > best_roc
            is_ac_best = (test_roc*100*0.5 + test_acc_close*0.5) > best_acc_roc
            # best_acc_val = max(val_acc, best_acc_val)
            best_acc_val = max(val_acc, best_acc_val)
            best_all_acc = max(test_overall, best_all_acc)
            best_roc = max(test_roc, best_roc)
            best_acc_roc = max((test_roc*0.5 + test_acc_close*0.5), best_acc_roc)
            if is_best:
                close_valid = test_acc_close
                overall_valid = test_overall
                unk_valid = test_unk
                roc_valid = test_roc
                roc_softm_valid = test_roc_softm
                aupr_in_valid = test_aupr_in
                aupr_out_valid = test_aupr_out
                best_aupr_out_ood_dic = aupr_out_ood_dic
                best_aupr_in_ood_dic = aupr_in_ood_dic
                best_roc_ood_dic = roc_ood_dic
            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc close': test_acc_close,
                'acc overall': test_overall,
                'unk': test_unk,
                'best_acc': best_acc,
                'best_acc_val':best_acc_val,
                'best_all_acc':best_all_acc,
                'best_roc':best_roc,
                'best_acc_roc':best_acc_roc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'var_list': var_list, 
                'loss_list': loss_list, 
                'sel_list': sel_list,
                'labels':labels,
                'dist_align':dist_align,
            }, is_best, args.out)
            if is_all_best:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model_to_save.state_dict(),
                    'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                    'acc close': test_acc_close,
                    'acc overall': test_overall,
                    'unk': test_unk,
                    'best_acc': best_acc,
                    'best_acc_val':best_acc_val,
                    'best_all_acc':best_all_acc,
                    'best_roc':best_roc,
                    'best_acc_roc':best_acc_roc,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'var_list': var_list, 
                    'loss_list': loss_list, 
                    'sel_list': sel_list,
                    'labels':labels,
                    'dist_align':dist_align,
                }, False, args.out, filename='all_best.pth.tar')
            if is_roc_best:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model_to_save.state_dict(),
                    'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                    'acc close': test_acc_close,
                    'acc overall': test_overall,
                    'unk': test_unk,
                    'best_acc': best_acc,
                    'best_acc_val':best_acc_val,
                    'best_all_acc':best_all_acc,
                    'best_roc':best_roc,
                    'best_acc_roc':best_acc_roc,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'var_list': var_list, 
                    'loss_list': loss_list, 
                    'sel_list': sel_list,
                    'labels':labels,
                    'dist_align':dist_align,
                }, False, args.out, filename='roc_best.pth.tar')
            if is_ac_best:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model_to_save.state_dict(),
                    'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                    'acc close': test_acc_close,
                    'acc overall': test_overall,
                    'unk': test_unk,
                    'best_acc': best_acc,
                    'best_acc_val':best_acc_val,
                    'best_all_acc':best_all_acc,
                    'best_roc':best_roc,
                    'best_acc_roc':best_acc_roc,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'var_list': var_list, 
                    'loss_list': loss_list, 
                    'sel_list': sel_list,
                    'labels':labels,
                    'dist_align':dist_align,
                }, False, args.out, filename='acc_roc_best.pth.tar')
            test_accs.append(test_acc_close)
            logger.info('Best val closed acc: {:.3f}'.format(best_acc_val))
            logger.info('Valid closed acc: {:.3f}'.format(close_valid))
            logger.info('Valid overall acc: {:.3f}'.format(overall_valid))
            logger.info('Valid unk acc: {:.3f}'.format(unk_valid))
            logger.info('Valid roc: {:.2f}'.format(roc_valid*100))
            logger.info('Valid roc soft: {:.3f}'.format(roc_softm_valid))
            logger.info('Valid aupr_in: {:.2f}'.format(aupr_in_valid*100))
            logger.info('Valid aupr_out: {:.2f}'.format(aupr_out_valid*100))
            logger.info('Mean top-1 acc: {:.3f}\n'.format(
                np.mean(test_accs[-20:])))
            
            for ood in best_roc_ood_dic.keys():
                logger.info("Valid ROC vs {ood}: {roc}".format(ood=ood, roc=best_roc_ood_dic[ood]))
                logger.info("Valid AUPR(in) vs {ood}: {aupr_in}".format(ood=ood, aupr_in=best_aupr_in_ood_dic[ood]))
                logger.info("Valid AUPR(out) vs {ood}: {aupr_out}".format(ood=ood, aupr_out=best_aupr_out_ood_dic[ood]))
                    
    if args.local_rank in [-1, 0]:
        args.writer.close()
