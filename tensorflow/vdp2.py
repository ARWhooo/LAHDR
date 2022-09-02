import transplant
import utils as utl
import os
import numpy as np


class VDPLogger:
    def __init__(self, save_dir, report_iterations, norm=False):
        self.bpe = report_iterations
        self.logger = utl.PerformanceHelper(['HDR-VDP2'])
        self.cnt = 0
        self.save_dir = save_dir
        self.matlab = transplant.Matlab(jvm=False, desktop=False)
        self.matlab.addpath('../hdrvdp-2.2.1')
        self.per_norm = norm
        if norm:
            self.logfile = 'HDR-VDP2_normed.txt'
        else:
            self.logfile = 'HDR-VDP2.txt'

    def log(self, pred, gt, epoch, bt):
        vdp2 = self.metric(np.squeeze(pred), np.squeeze(gt))
        self.logger.iter_record([vdp2], True)
        utl.save_to_file(os.path.join(self.save_dir, self.logfile),
                         'Epoch %02d, Batch %02d: %f' % (epoch, bt, vdp2))
        self.cnt += 1
        if self.cnt == self.bpe:
            rcd_str = self.logger.report_all_averages(True)
            utl.save_to_file(os.path.join(self.save_dir, self.logfile), rcd_str)
            self.cnt = 0
            self.logger.flush()

    def metric(self, inp, lbl, maxval=1.0, trans_func=None):
        if self.per_norm:
            inp = inp / inp.max()
            lbl = lbl / lbl.max()
        else:
            inp = inp / maxval
            lbl = lbl / maxval
        if trans_func is not None:
            inp = trans_func(inp, maxval)
            lbl = trans_func(lbl, maxval)
        return self.matlab.metric_vdp2(inp.astype(np.float32), lbl.astype(np.float32))

    def __del__(self):
        self.matlab.exit()
