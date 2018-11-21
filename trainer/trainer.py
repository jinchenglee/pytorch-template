import numpy as np
import time
import torch
from torchvision.utils import make_grid
from base import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, optimizer, resume, config,
                 data_loader, num_classes=3, valid_data_loader=None, lr_scheduler=None, train_logger=None):
        super(Trainer, self).__init__(model, loss, metrics, optimizer, resume, config, train_logger)
        self.config = config
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = self.batch_size
        self.num_classes = num_classes

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros((len(self.metrics), self.num_classes))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
            self.writer.add_scalar(metric.__name__+'_car', acc_metrics[i][0])
            self.writer.add_scalar(metric.__name__+'_none', acc_metrics[i][1])
            self.writer.add_scalar(metric.__name__+'_ped', acc_metrics[i][2])
            self.writer.add_scalar(metric.__name__+'_rider', acc_metrics[i][3])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        time_start = time.time()

        self.model.train()
    
        total_loss = 0
        total_metrics = np.zeros((len(self.metrics), self.num_classes))
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            self.writer.add_scalar('loss', loss.item())
            # TODO: do loss calculation on GPU? And piece-wise? To reduce traffic between GPUs.
            total_loss += loss.item()
            # print("_train: target=", target, ", output=", output, ", total_loss=", total_loss)
            total_metrics += self._eval_metrics(output, target)

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    self.data_loader.n_samples,
                    100.0 * batch_idx / len(self.data_loader),
                    loss.item()))
                if batch_idx % (100 * self.log_step):
                    # Save only first 4 images of batch
                    # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
                    self.writer.add_image('input', make_grid(data.cpu()[:4], nrow=1, normalize=True))
                    self.writer.add_image('pred', make_grid(output.cpu()[:4], nrow=1, normalize=True))
                    self.writer.add_image('target', make_grid(target.cpu()[:4], nrow=1, normalize=True))

        log = {
            'loss': total_loss / len(self.data_loader),
            'metrics': (total_metrics / len(self.data_loader)).tolist()
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        time_end = time.time()
        print("\n_train_epoch() time spent:", time_end - time_start, "s")

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros((len(self.metrics), self.num_classes))
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.loss(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.writer.add_scalar('loss', loss.item())
                total_val_loss += loss.item()
                # print("_val: target=", target, ", output=", output, ", total_loss=", total_val_loss)
                total_val_metrics += self._eval_metrics(output, target)

                # Less freq. image dumpping for faster validation
                if batch_idx % self.log_step == 0:
                    # Save only first 4 images of batch
                    # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
                    self.writer.add_image('input', make_grid(data.cpu()[:4], nrow=1, normalize=True))
                    self.writer.add_image('pred', make_grid(output.cpu()[:4], nrow=1, normalize=True))
                    self.writer.add_image('target', make_grid(target.cpu()[:4], nrow=1, normalize=True))

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }
