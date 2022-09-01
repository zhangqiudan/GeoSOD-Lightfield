import argparse
import os

from dataset_model import get_loader
from solver_model import Solver


def main(config):
    if config.mode == 'train':
        train_loader, dataset = get_loader(config.batch_size, num_thread=config.num_thread)
        run = "model"
        if not os.path.exists("%s/run-%s" % (config.save_fold, run)):
            os.mkdir("%s/run-%s" % (config.save_fold, run))
            os.mkdir("%s/run-%s/logs" % (config.save_fold, run))
            os.mkdir("%s/run-%s/models_save" % (config.save_fold, run))
        config.save_fold = "%s/run-%s" % (config.save_fold, run)
        train = Solver(train_loader, None, config)
        train.train()
    elif config.mode == 'test':
        test_loader, dataset = get_loader(config.test_batch_size, mode='test', num_thread=config.num_thread,
                                          test_mode=config.test_mode, sal_mode=config.sal_mode)

        test = Solver(None, test_loader, config, dataset.save_folder())
        test.test(test_mode=config.test_mode)
    else:
        raise IOError("illegal input!!!")


if __name__ == '__main__':

    resnet_path = './weights/resnet50_caffe.pth'

    parser = argparse.ArgumentParser()

    # Hyper-parameters
    parser.add_argument('--n_color', type=int, default=3)

    parser.add_argument('--cuda', type=bool, default=True)

    # Training settings
    parser.add_argument('--resnet', type=str, default=resnet_path)
    parser.add_argument('--epoch', type=int, default=50)  # 12, now x3
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--num_thread', type=int, default=0)
    parser.add_argument('--load_bone', type=str, default='')
    parser.add_argument('--save_fold', type=str, default='./')
    parser.add_argument('--epoch_save', type=int, default=1)  # 2, now x3
    parser.add_argument('--epoch_show', type=int, default=1)
    parser.add_argument('--pre_trained', type=str, default=None)

    # Testing settings
    parser.add_argument('--model', type=str, default='./run-model/models.pth')
    parser.add_argument('--test_fold', type=str, default='./results/test')
    parser.add_argument('--test_mode', type=int, default=1)
    parser.add_argument('--sal_mode', type=str, default='e')
    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--visdom', type=bool, default=False)

    config = parser.parse_args()

    if not os.path.exists(config.save_fold): os.mkdir(config.save_fold)
    main(config)
