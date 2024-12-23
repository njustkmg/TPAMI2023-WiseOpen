import logging
from utils import test, test_ood, test_upper

logger = logging.getLogger(__name__)
best_acc = 0
best_acc_val = 0
def eval_model(args, labeled_trainloader, unlabeled_dataset, test_loader, val_loader,
          ood_loaders, model, ema_model):
    if args.amp:
        from apex import amp
    global best_acc
    global best_acc_val

    model.eval()
    if args.use_ema:
        test_model = ema_model.ema
    else:
        test_model = model
    epoch = 0
    if args.local_rank in [-1, 0]:
        val_acc = test(args, val_loader, test_model, epoch, val=True)
        if args.upper_bound:
            test_loss, test_overall, close_valid = test_upper(args, test_loader, test_model, epoch)
        else:
            test_loss, close_valid, test_overall, \
            test_unk, test_roc, test_roc_softm, test_id, \
            test_aupr_in, test_aupr_out \
                = test(args, test_loader, test_model, epoch)   

            unk_valid = test_unk
            roc_valid = test_roc
            roc_softm_valid = test_roc_softm 
            # for ood in ood_loaders.keys():
            #     args.no_progress = 1
            #     roc_ood, aupr_in_ood, aupr_out_ood = test_ood(args, test_id, ood_loaders[ood], test_model)
            #     logger.info("ROC vs {ood}: {roc:.2f}".format(ood=ood, roc=roc_ood*100))
            #     logger.info("AUPR(in) vs {ood}: {aupr_in:.2f}".format(ood=ood, aupr_in=aupr_in_ood*100))
            #     logger.info("AUPR(out) vs {ood}: {aupr_out:.2f}".format(ood=ood, aupr_out=aupr_out_ood*100))

        overall_valid = test_overall
        
        logger.info('validation closed acc: {:.2f}'.format(val_acc))
        logger.info('test closed acc: {:.2f}'.format(close_valid))
        logger.info('test overall acc: {:.2f}'.format(overall_valid))
        if not args.upper_bound:
            logger.info('test unk acc: {:.2f}'.format(unk_valid*100))
            logger.info('test roc: {:.2f}'.format(roc_valid*100))
            logger.info('test roc soft: {:.2f}'.format(roc_softm_valid*100))
            logger.info('test AUPR(in): {:.2f}'.format(test_aupr_in*100))
            logger.info('test AUPR(out): {:.2f}'.format(test_aupr_out*100))
    if args.local_rank in [-1, 0]:
        args.writer.close()
