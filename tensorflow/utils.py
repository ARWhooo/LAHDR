import tensorflow as tf
import numpy as np
import os, re
import logging, time


def save_to_file(filename, a):
    with open(filename, 'a') as f:
        f.writelines(a)
        f.write('\n')


def list_testing_pairs(ldr_dir, hdr_dir):
    t1 = os.listdir(ldr_dir)
    t2 = os.listdir(hdr_dir)
    reg_ldr = re.compile(".*.(jpg|png|jpeg|bmp)")
    reg_hdr = re.compile(".*.(pfm|exr|hdr|dng)")
    ldr_list = [f.split('.') for f in t1 if reg_ldr.match(f)]
    hdr_list = [f.split('.') for f in t2 if reg_hdr.match(f)]
    l_list = []
    h_list = []
    ldr_list.sort(key=(lambda x: x[0]))
    hdr_list.sort(key=(lambda x: x[0]))
    if len(ldr_list) != len(hdr_list):
        print("Error: dataset pairs count uneven.")
        exit(1)
    for i in range(len(ldr_list)):
        if ldr_list[i][0] != hdr_list[i][0]:
            print("Error: dataset picture names unpair.")
            exit(1)
        l_list.append(os.path.join(ldr_dir, ldr_list[i][0] + '.' + ldr_list[i][1]))
        h_list.append(os.path.join(hdr_dir, hdr_list[i][0] + '.' + hdr_list[i][1]))
    return l_list, h_list, len(l_list)


def get_train_save_parameters(scope):
    if scope is None:
        restore_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        list_all = tf.global_variables()
        bn_moving_vars = [g for g in list_all if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in list_all if 'moving_variance' in g.name]
        var_all = restore_vars + bn_moving_vars
    else:
        if not (isinstance(scope, list) or isinstance(scope, tuple)):
            scope = [scope]
        restore_vars = []
        var_all = []
        for sc in scope:
            rev = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=sc)
            list_all = tf.global_variables(scope=sc)
            bn_moving_vars = [g for g in list_all if 'moving_mean' in g.name]
            bn_moving_vars += [g for g in list_all if 'moving_variance' in g.name]
            vaa = rev + bn_moving_vars
            restore_vars += rev
            var_all += vaa
    return restore_vars, var_all


def get_update_ops(scope):
    if scope is None:
        ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    else:
        if not (isinstance(scope, list) or isinstance(scope, tuple)):
            scope = [scope]
        ops = []
        for sc in scope:
            ops += tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=sc)
    return ops


# This function must called when the global_step tensor is declared in the graph with the name 'global_step:0'
# Must be called after the init_op executed in the session
def load_latest_state(saver, sess, checkpoint_dir, batchnum, global_step):
    ckpt_state = tf.train.get_checkpoint_state(checkpoint_dir)
    start_epoch = 1
    start_batchcnt = 1
    if ckpt_state and ckpt_state.model_checkpoint_path:
        resume_path = tf.train.latest_checkpoint(checkpoint_dir)
        step = int(resume_path.split('-')[-1])
        start_epoch = step // batchnum + 1
        start_batchcnt = step % batchnum + 1
        global_step = tf.assign(global_step, step)
        step = sess.run(global_step)
        print('Loaded previous results from: %s, training step: %d.' % (resume_path, step))
        saver.restore(sess, resume_path)
    return start_epoch, start_batchcnt, global_step


def checkpoint_restore(path, scope, sess):
    ev_path = tf.train.latest_checkpoint(path)
    if ev_path is None:
        print('Could not find the latest checkpoint in {}'.format(path))
        exit(-1)
    else:
        print('Found model: %s' % ev_path)
    _, var_all = get_train_save_parameters(scope)
    saver = tf.train.Saver(var_all)
    saver.restore(sess, ev_path)


class LossesHelper:
    def __init__(self, loss_dec):
        self.count = len(loss_dec)
        self.loss_pool = []
        for i in range(self.count):
            self.loss_pool.append([])
        self.decs = loss_dec

    def iter_record(self, entries, display=True):
        if len(entries) != self.count:
            print('Not valid losses entry!')
            exit(-1)
        rcd_str = ''
        rcd_str2 = ''
        for i in range(self.count):
            self.loss_pool[i].append(entries[i])
            rcd_str += '%s: %f' % (self.decs[i], entries[i])
            rcd_str2 += '%s: %f' % (self.decs[i], entries[i])
            if i < (self.count - 1):
                rcd_str += ', '
                rcd_str2 += '\n'
            else:
                rcd_str += '.'
        if display:
            print(rcd_str2)
        return rcd_str

    def get_average(self, i):
        if i >= self.count or i < 0:
            print('Not valid index number!')
            exit(-1)
        m_result = np.mean(np.array(self.loss_pool[i]))
        return m_result

    def report_average(self, i, display=True):
        res = self.get_average(i)
        rcd_str = '%s: %f' % (self.decs[i], res)
        if display:
            print(rcd_str)
        return rcd_str

    def flush(self):
        for i in range(self.count):
            self.loss_pool[i].clear()

    def report_all_averages(self, display=True):
        rcd_str = ''
        for i in range(self.count):
            rcd_str += self.report_average(i, display)
            if i < (self.count - 1):
                rcd_str += ', '
            else:
                rcd_str += '.'
        return rcd_str


class PerformanceHelper(LossesHelper):
    def __init__(self, perf_desc):
        super(PerformanceHelper, self).__init__(perf_desc)

    def iter_record(self, entries, display=True):
        if len(entries) != self.count:
            print('Not valid losses entry!')
            exit(-1)
        rcd_str = ''
        for i in range(self.count):
            self.loss_pool[i].append(entries[i])
            rcd_str += '%s %.4f' % (self.decs[i], entries[i])
            if i < (self.count - 1):
                rcd_str += ', '
            else:
                rcd_str += '.'
        if display:
            print(rcd_str)
        return rcd_str

    def report_average(self, i, display=True):
        res = self.get_average(i)
        rcd_str = '%s %.4f' % (self.decs[i], res)
        if display:
            print(rcd_str)
        return rcd_str


def num_of_total_parameters(scope=None):
    total_parameters = 0
    if scope is None:
        variables = tf.trainable_variables()
    else:
        variables = tf.trainable_variables(scope=scope)
    for variable in variables:
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters


def print_num_of_total_parameters(scope=None, output_detail=False, output_to_logging=False):
    total_parameters = 0
    parameters_string = ''

    if scope is None:
        variables = tf.trainable_variables()
    else:
        variables = tf.trainable_variables(scope=scope)

    for variable in variables:
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
        if len(shape) == 1:
            parameters_string += ("%s %d, " % (variable.name, variable_parameters))
        else:
            parameters_string += ("%s %s=%d, " % (variable.name, str(shape), variable_parameters))

    if output_to_logging:
        if scope is not None:
            logging.info("Network: %s" % scope)
        if output_detail:
            logging.info(parameters_string)
        logging.info("Total %d variables, %s params" % (len(variables), "{:,}".format(total_parameters)))
    else:
        if scope is not None:
            print("Network: %s" % scope)
        if output_detail:
            print(parameters_string)
        print("Total %d variables, %s params" % (len(variables), "{:,}".format(total_parameters)))
    return total_parameters


def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('GFLOPs: {}, Trainable params: {}.'.format(flops.total_float_ops/1000000000.0, params.total_parameters))


def prepare_single_net_train(loss, learning_rate, train_decay, scope):
    global_step = tf.Variable(0, trainable=False, name='global_step')
    decay_type = train_decay['type']
    if decay_type == 'exponential':
        decay_step = train_decay['decay_step']
        start_lr = tf.train.exponential_decay(learning_rate, global_step, decay_step, 0.9, staircase=True)
    else:  # train_decay['type'] == 'None'
        start_lr = tf.convert_to_tensor(learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)
    g_vars, var_all = get_train_save_parameters(scope)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(start_lr).minimize(loss, global_step=global_step, var_list=g_vars)
    saver = tf.train.Saver(var_all)
    return train_op, saver, global_step, start_lr


class Model:
    def __init__(self, mdl, mdl_dir, mdl_name):
        if '-' in mdl_name:
            raise RuntimeError('Invalid model name: %s, "-" included.' % mdl_name)
        self.mdl = mdl
        self.loc = mdl_dir
        self.name = mdl_name
        self._save_hdl = None
        self._global_step = None
        self._lr = None
        self.train_op = None
        self.store_contents = None
        self.loss_collect = None
        self.loss_desc = None
        self.loss = 0
        self.training = False

    def get_parameters(self):
        return get_train_save_parameters(self._scope())

    def _scope(self):
        return self.name

    def _saver(self):
        if self._save_hdl is None:
            _, vars = self.get_parameters()
            self._save_hdl = tf.train.Saver(vars)
        return self._save_hdl

    def _net_step(self):
        if self._global_step is None:
            self._global_step = tf.Variable(0, trainable=False, name=self.name + '_global_step')
        return self._global_step

    def _setup_lr(self, learning_rate, train_decay, decay_step):
        if train_decay:
            start_lr = tf.train.exponential_decay(learning_rate, self._net_step(), decay_step, 0.9, staircase=True)
        else:
            start_lr = tf.convert_to_tensor(learning_rate)
        return start_lr

    def restore(self, sess, print_status=True):
        ev_path = tf.train.latest_checkpoint(self.loc)
        if ev_path is None:
            print('Could not find the latest checkpoint in {}'.format(self.loc))
            exit(-1)
        else:
            print('Found model: %s' % ev_path)
        self._saver().restore(sess, ev_path)
        return self.status(print_status)

    def restore_from_loc(self, sess, loc, print_status=True):
        ev_path = tf.train.latest_checkpoint(loc)
        if ev_path is None:
            print('Could not find the latest checkpoint in {}'.format(loc))
            exit(-1)
        else:
            print('Found model: %s' % ev_path)
        self._saver().restore(sess, ev_path)
        return self.status(print_status)

    def save(self, sess, global_step=None):
        loc = os.path.join(self.loc, self.name + '.ckpt')
        if global_step is None:
            global_step = self._net_step()
        self._saver().save(sess, loc, global_step=global_step)

    def final_save(self, sess):
        loc = os.path.join(self.loc, self.name + '_final.ckpt')
        self._saver().save(sess, loc, global_step=self._net_step())

    def _optim(self, loss, start_lr):
        train_vars, save_vars = self.get_parameters()
        global_step = self._net_step()
        self._save_hdl = tf.train.Saver(save_vars)
        with tf.control_dependencies(get_update_ops(self._scope())):
            train_op = tf.train.AdamOptimizer(start_lr).minimize(loss, global_step=global_step,
                                                                 var_list=train_vars)
        return train_op

    def optim(self, loss, start_lr, train_decay, decay_step, store_contents, loss_collect, loss_desc):
        self._lr = self._setup_lr(start_lr, train_decay, decay_step)
        self.train_op = self._optim(loss, self._lr)
        self.store_contents = store_contents
        self.loss_collect = loss_collect
        self.loss_desc = loss_desc
        self.loss = loss
        if not os.path.exists(self.loc):
            os.mkdir(self.loc)

    def train_loader(self):
        if self.train_op is None:
            raise RuntimeError('%s needs setup by optim() before the train_loader() call.' % self.name)
        return self.loss, self.train_op, self.loss_collect, self.loss_desc

    def test_loader(self):
        if self.train_op is None:
            raise RuntimeError('%s needs setup by optim() before the test_loader() call.' % self.name)
        return self.store_contents, self.loss, self.loss_collect

    def resume(self, sess, total_batchnum):
        global_step = self._net_step()
        epoch, batch, self._global_step = load_latest_state(self._saver(), sess, self.loc,
                                                        total_batchnum, global_step)
        return epoch, batch

    def status(self, print_status=True):
        if print_status:
            return print_num_of_total_parameters(scope=self.name)
        else:
            return num_of_total_parameters(scope=self.name)

    def __call__(self, inp, is_training):
        self.out = self.mdl(inp, name=self.name, is_training=is_training)
        self.training = is_training
        return self.out


def single_net_train_flags(logfile, test_interval, train_resume, print_batch, save_dir, chk_interval):
    return {
        'logfile': logfile,
        'test_interval': test_interval,
        'adaptive_resume': train_resume,
        'print_train_result': print_batch,
        'test_save_dir': save_dir,
        'checkpoint_interval': chk_interval,
    }


def single_net_train(sess, base_mdl, pre_loader, feeder, evaluations, dataset, testset, train_flags):
    _ = sess.run(tf.global_variables_initializer())
    if pre_loader is not None:
        pre_loader(sess, base_mdl)

    savefile_name = train_flags['logfile']
    test_interval = train_flags['test_interval']
    adaptive_resume = train_flags['adaptive_resume']
    print_train_result = train_flags['print_train_result']
    test_save_dir = train_flags['test_save_dir']
    checkpoint_interval = train_flags['checkpoint_interval']
    checkpoint_dir = base_mdl.loc

    params = base_mdl.status(True)
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    if not os.path.exists(os.path.join(checkpoint_dir, 'networks.log')):
        save_to_file(os.path.join(checkpoint_dir, 'networks.log'),
                     'Training network %s, location %s, parameters %d.' % (base_mdl.name, base_mdl.loc, params))
    if adaptive_resume:
        epoch, batch = base_mdl.resume(sess, dataset.get_total_batchnum())
    else:
        epoch = 1
        batch = 1
    sess.graph.finalize()
    try:
        while epoch <= dataset.get_total_epochs():
            loss, train_op, loss_collect, loss_desc = base_mdl.train_loader()
            lossHelp = LossesHelper(loss_desc)
            mean_loss = []
            train_time = []
            while batch <= dataset.get_total_batchnum():
                st_time = time.time()
                feeds = feeder(dataset, epoch)
                whole_loss, _, lossvals = sess.run([loss, train_op, loss_collect], feed_dict=feeds)
                if print_train_result:
                    rcd_str = 'Training: epoch %d, batch %d: whole loss: %f.' % \
                              (epoch, batch, whole_loss)
                    print(rcd_str)
                    lossHelp.iter_record(lossvals, True)
                else:
                    lossHelp.iter_record(lossvals, False)
                mean_loss.append(whole_loss)
                batch += 1
                train_time.append(time.time() - st_time)
            m_loss = np.mean(np.array(mean_loss))
            m_time = np.mean(np.array(train_time))
            rcd_str = 'Mean loss for epoch %d: %f, average time: %f, ' % (epoch, m_loss, m_time)
            print(rcd_str)
            save_to_file(savefile_name, rcd_str + lossHelp.report_all_averages())
            mean_loss.clear()
            lossHelp.flush()
            if (epoch % test_interval == 0) and testset is not None:
                bt = 1
                while bt <= testset.get_total_batchnum():
                    feeds = feeder(testset, epoch)
                    train_contents, loss, loss_collect = base_mdl.test_loader()
                    output, stage_losses, stage_collects = sess.run([train_contents, loss, loss_collect],
                                                                    feed_dict=feeds)
                    rcd_str = 'Testing epoch %d, batch %d: whole loss: %f. ' % (epoch, bt, stage_losses)
                    print(rcd_str)
                    rcd_str += lossHelp.iter_record(stage_collects, True)
                    save_to_file(savefile_name, rcd_str)
                    if evaluations is not None:
                        evaluations(output, test_save_dir, epoch, bt)
                    mean_loss.append(stage_losses)
                    bt += 1
                m_loss = np.mean(np.array(mean_loss))
                rcd_str = 'Mean loss for whole epoch: %f. ' % (m_loss)
                print(rcd_str)
                save_to_file(savefile_name, rcd_str + lossHelp.report_all_averages())
                mean_loss.clear()
                lossHelp.flush()
            if epoch % checkpoint_interval == 0:
                base_mdl.save(sess)
            epoch += 1
            batch = 1
    except KeyboardInterrupt:
        print("Whole training complete. %d epochs trained." % (epoch - 1))

    base_mdl.final_save(sess)


def multiple_net_train_flags(logfile, test_interval, train_resume, print_batch, save_dir, chk_interval, chk_dir):
    return {
        'logfile': logfile,
        'test_interval': test_interval,
        'adaptive_resume': train_resume,
        'print_train_result': print_batch,
        'test_save_dir': save_dir,
        'checkpoint_interval': chk_interval,
        'checkpoint_dir': chk_dir
    }


def multiple_net_train(sess, dataset, networks, pre_loader, feeder, train_loader, testset, test_loader, evaluations,
                       train_flags):
    _ = sess.run(tf.global_variables_initializer())
    if pre_loader is not None:
        pre_loader(sess, networks)

    for net in networks:
        if not net.training:
            net.restore(sess, True)
        else:
            net.status(True)

    savefile_name = train_flags['logfile']
    checkpoint_dir = train_flags['checkpoint_dir']
    test_interval = train_flags['test_interval']
    adaptive_resume = train_flags['adaptive_resume']
    print_train_result = train_flags['print_train_result']
    test_save_dir = train_flags['test_save_dir']
    checkpoint_interval = train_flags['checkpoint_interval']

    epoch = 1
    batch = 1
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    if not os.path.exists(os.path.join(checkpoint_dir, 'networks.log')):
        for net in networks:
            params = net.status(False)
            if net.training:
                save_to_file(os.path.join(checkpoint_dir, 'networks.log'),
                             'Training network %s, location %s, parameters %d.' % (net.name, net.loc, params))
            else:
                save_to_file(os.path.join(checkpoint_dir, 'networks.log'),
                             'Pre-trained network %s, location %s, parameters %d.' % (net.name, net.loc, params))
    if not os.path.exists(os.path.join(checkpoint_dir, 'saved_epoch.npy')):
        if adaptive_resume:
            print('Epoch storage "saved_epoch.npy" not found in %s, resume aborted.' % checkpoint_dir)
        adaptive_resume = False
    if adaptive_resume:
        epoch = np.load(os.path.join(checkpoint_dir, 'saved_epoch.npy'))
        for net in networks:
            if net.training:
                net.resume(sess, dataset.get_total_batchnum())
        print('Resume to epoch %d, batch %d.' % (epoch, batch))

    sess.graph.finalize()
    try:
        while epoch <= dataset.get_total_epochs():
            loss, train_op, loss_collect, loss_dec = train_loader(epoch, networks)
            lossHelp = LossesHelper(loss_dec)
            mean_loss = []
            train_time = []
            while batch <= dataset.get_total_batchnum():
                st_time = time.time()
                feeds = feeder(dataset, epoch)
                whole_loss, _, lossvals = sess.run([loss, train_op, loss_collect], feed_dict=feeds)
                if print_train_result:
                    rcd_str = 'Training: epoch %d, batch %d: whole loss: %f.' % \
                              (epoch, batch, whole_loss)
                    print(rcd_str)
                    lossHelp.iter_record(lossvals, True)
                else:
                    lossHelp.iter_record(lossvals, False)
                mean_loss.append(whole_loss)
                batch += 1
                train_time.append(time.time() - st_time)
            m_loss = np.mean(np.array(mean_loss))
            m_time = np.mean(np.array(train_time))
            rcd_str = 'Mean loss for epoch %d: %f, average time: %f. ' % (epoch, m_loss, m_time)
            print(rcd_str)
            save_to_file(savefile_name, rcd_str + lossHelp.report_all_averages())
            mean_loss.clear()
            lossHelp.flush()
            if (epoch % test_interval == 0) and testset is not None:
                bt = 1
                train_contents, loss, loss_collect = test_loader(epoch, networks)
                while bt <= testset.get_total_batchnum():
                    feeds = feeder(testset, epoch)
                    output, stage_losses, stage_collects = sess.run([train_contents, loss, loss_collect],
                                                                    feed_dict=feeds)
                    rcd_str = 'Testing epoch %d, batch %d: whole loss: %f. ' % (epoch, bt, stage_losses)
                    print(rcd_str)
                    rcd_str += lossHelp.iter_record(stage_collects, True)
                    save_to_file(savefile_name, rcd_str)
                    if evaluations is not None:
                        evaluations(output, test_save_dir, epoch, bt)
                    mean_loss.append(stage_losses)
                    bt += 1
                m_loss = np.mean(np.array(mean_loss))
                rcd_str = 'Mean loss for whole epoch: %f. ' % (m_loss)
                print(rcd_str)
                save_to_file(savefile_name, rcd_str + lossHelp.report_all_averages())
                mean_loss.clear()
                lossHelp.flush()
            if epoch % checkpoint_interval == 0:
                print('Saving networks, epoch %d.' % epoch)
                np.save(os.path.join(checkpoint_dir, 'saved_epoch.npy'), epoch)
                for net in networks:
                    net.save(sess)
            epoch += 1
            batch = 1
    except KeyboardInterrupt:
        print("Whole training complete. %d epochs trained." % (epoch - 1))

    for net in networks:
        net.final_save(sess)
