from network.basic import LossModel, SingleBaseTester, SingleBaseTrainer


class PairLoss(LossModel):
    def __init__(self, options, verbose=False):
        super(PairLoss, self).__init__(options, verbose)

    def forward(self, component_pairs):
        self._loss = self.handle(*component_pairs)
        self.loss_collect = [x.item() for x in self.handle.collect()]
        self.loss = self._loss.item()
        self.loss_collect = [self.loss] + self.loss_collect
        self.perf.iter_record(self.loss_collect, self.verbose_notify)
        return self.loss


class ImagePairTester(SingleBaseTester):
    def __init__(self, options, process_func, model=None, verbose=True):
        super().__init__(options, model, verbose)
        self.loss_enabled = self._check_loss_enabled(options)
        self.loss_handle = PairLoss(options, False) if self.loss_enabled else None
        self.pred_result = None
        self.process = process_func

    @staticmethod
    def _check_loss_enabled(opt):
        if opt['loss'] == 'none' or opt['loss'] == 'None' or opt['loss'] is None:
            print('No loss function designated for testing procedure.')
            return False
        else:
            return True

    def iter_test(self, train_pairs):
        pred, loss = self.process(train_pairs, self.model, self.loss_handle, self.device, False)
        self.pred_result = pred
        return pred, loss

    def prediction(self, train_pairs):
        return self.iter_test(train_pairs)

    def report_handle(self):
        return self.loss_handle


class ImagePairTrainer(SingleBaseTrainer):
    def __init__(self, options, process_func, model=None, verbose=True):
        super().__init__(options, model, verbose)
        self.loss_handle = PairLoss(options, False)
        self.pred = None
        self.loss = 0
        self.process = process_func

    def iter_batch(self, epoch, batch, train_pairs):
        pred, loss = self.process(train_pairs, self.model, self.loss_handle, self.device, True)
        self.pred = pred
        self.loss = loss
        return self.pred, self.loss

    def batch_update(self, batch, notify=False):
        self.optimizer.zero_grad()
        self.loss_handle.backward()
        self.optimizer.step()
        return self.loss_handle.report_status(notify)

    def epoch_update(self, epoch, notify=False):
        if self.scheduler is not None:
            self.scheduler.step()
        self.last_epoch = epoch

    def prediction(self, train_pairs):
        pred, loss = self.process(train_pairs, self.model, self.loss_handle, self.device, False)
        self.pred = pred
        self.loss = loss
        return self.pred, self.loss

    def report_handle(self):
        return self.loss_handle
