import tools
import data_roller
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import pickle
from tensorflow.python import debug as tf_debug
import model
import losses
import os
import numpy as np
import time
import config


class Train(object):
    def __init__(self, num_gpu, cfg, train_sequences, val_sequence, tensorboard_meta=False, start_epoch=0,
                 restore_file=None, restore_ekf_state_file=None, train_frames_range=None, val_frames_range=None):
        # configurations
        self.cfg = cfg
        self.num_gpu = num_gpu
        self.train_sequences = train_sequences
        self.val_sequence = val_sequence
        self.tensorboard_meta = tensorboard_meta
        self.results_dir_path = ""
        self.curr_dir_path = ""
        self.start_epoch = start_epoch
        self.restore_file = restore_file
        self.restore_ekf_state_file = restore_ekf_state_file
        self.train_frames_range = train_frames_range
        self.val_frames_range = val_frames_range

        # data managers
        self.train_data_gen = None
        self.val_data_gen = None

        # tf variables, ops, objects
        # tensors
        self.t_alpha = None
        self.t_lr = None
        self.t_inputs = None
        self.t_lstm_initial_state = None
        self.t_ekf_initial_state = None
        self.t_ekf_initial_covariance = None
        self.t_initial_poses = None
        self.t_is_training = None
        self.t_use_initializer = None
        self.t_se3_labels = None
        self.t_fc_labels = None
        self.t_lstm_states = None
        self.t_ekf_states = None
        self.t_ekf_covar_states = None
        self.t_se3_loss = None
        self.t_total_loss = None
        self.t_sequence_id = None
        self.t_epoch = None

        # ops
        self.op_trainer = None
        self.op_train_merged_summary = None
        self.op_train_image_summary = None
        self.op_val_merged_summary = None

        # objects
        self.tf_saver_checkpoint = None
        self.tf_saver_best = None
        self.tf_saver_restore = None
        self.best_val_path = None
        self.model_epoch_path = None

        self.tf_session = None
        self.tf_tb_writer = None

        if self.tensorboard_meta:
            self.tf_run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            self.tf_run_metadata = tf.RunMetadata()
        else:
            self.tf_run_options = None
            self.tf_run_metadata = None

        # ensure that the batch size is a multiple of num GPUs
        assert (type(self.cfg.batch_size) is int and
                type(self.num_gpu) is int and
                self.cfg.batch_size % self.num_gpu == 0)

        self.__log_files_and_configs()
        self.__build_model_inputs_and_labels()
        self.__build_model_and_summary()
        self.__init_tf_savers()
        self.__load_data_set()

    def train(self):
        self.__run_train()

    def __load_data_set(self):
        tools.printf("Loading training data...")
        self.train_data_gen = data_roller.StatefulRollerDataGen(self.cfg, config.dataset_path, self.train_sequences,
                                                                frames=self.train_frames_range)
        tools.printf("Loading validation data...")
        self.val_data_gen = data_roller.StatefulRollerDataGen(self.cfg, config.dataset_path, [self.val_sequence],
                                                              frames=self.val_frames_range)

    def __log_files_and_configs(self):
        self.results_dir_path = tools.create_results_dir("train_seq")
        self.curr_dir_path = os.path.dirname(os.path.realpath(__file__))
        tools.log_file_content(self.results_dir_path, [os.path.realpath(__file__),
                                                       os.path.join(self.curr_dir_path, "data_roller.py"),
                                                       os.path.join(self.curr_dir_path, "model.py"),
                                                       os.path.join(self.curr_dir_path, "losses.py"),
                                                       os.path.join(self.curr_dir_path, "train.py"),
                                                       os.path.join(self.curr_dir_path, "train_seq.py"),
                                                       os.path.join(self.curr_dir_path, "config.py")])
        tools.set_log_file(os.path.join(self.results_dir_path, "print_logs.txt"))
        config.print_configs(self.cfg)

    def __init_tf_savers(self):
        self.tf_saver_checkpoint = tf.train.Saver(max_to_keep=2)
        self.tf_saver_best = tf.train.Saver(max_to_keep=2)
        if self.cfg.dont_restore_init:
            varlist = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="cnn_layer") + \
                      tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="fc_layer") + \
                      tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="rnn_layer")
        else:
            varlist = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        self.tf_saver_restore = tf.train.Saver(var_list=varlist)
        self.best_val_path = os.path.join(self.results_dir_path, "best_val")
        self.model_epoch_path = self.results_dir_path

    def __build_model_inputs_and_labels(self):
        self.t_inputs, self.t_lstm_initial_state, self.t_initial_poses, self.t_imu_data, self.t_ekf_initial_state, \
        self.t_ekf_initial_covariance, self.t_is_training, self.t_use_initializer = model.seq_model_inputs(self.cfg)

        # 7 for translation + quat
        self.t_se3_labels = tf.placeholder(tf.float32, name="se3_labels",
                                           shape=[self.cfg.timesteps, self.cfg.batch_size, 7])

        # 6 for translation + rpy, labels not needed for covars
        self.t_fc_labels = tf.placeholder(tf.float32, name="se3_labels",
                                          shape=[self.cfg.timesteps, self.cfg.batch_size, 6])

        # between 0 and 1, larger favors fc loss
        self.t_alpha = tf.placeholder(tf.float32, name="alpha", shape=[])
        self.t_lr = tf.placeholder(tf.float32, name="se3_lr", shape=[])

    def __build_model_and_summary(self):
        # split the tensors
        with tf.variable_scope("tower_split"), tf.device("/cpu:0"):
            # splitted tensors
            ts_inputs = tf.split(self.t_inputs, self.num_gpu, 1)
            ts_lstm_initial_state = tf.split(self.t_lstm_initial_state, self.num_gpu, 2)
            ts_initial_poses = tf.split(self.t_initial_poses, self.num_gpu, 0)
            ts_imu_data = tf.split(self.t_imu_data, self.num_gpu, 1)
            ts_ekf_initial_state = tf.split(self.t_ekf_initial_state, self.num_gpu, 0)
            ts_ekf_initial_covar = tf.split(self.t_ekf_initial_covariance, self.num_gpu, 0)
            ts_se3_labels = tf.split(self.t_se3_labels, self.num_gpu, 1)
            ts_fc_labels = tf.split(self.t_fc_labels, self.num_gpu, 1)

            # list to store results
            ts_ekf_states = []
            ts_ekf_covar_states = []
            ts_lstm_states = []
            losses_keys = ["se3_loss", "se3_xyz_loss", "se3_quat_loss",
                           "fc_loss", "fc_xyz_loss", "fc_ypr_loss", "x_loss", "y_loss", "z_loss",
                           "total_loss"]
            ts_losses_dict = dict(zip(losses_keys, [[] for i in range(len(losses_keys))]))

        for i in range(0, self.num_gpu):
            device_setter = tf.train.replica_device_setter(ps_tasks=1,
                                                           ps_device='/job:localhost/replica:0/task:0/device:CPU:0',
                                                           worker_device='/job:localhost/replica:0/task:0/device:GPU:%d' % i)

            with tf.name_scope("tower_%d" % i), tf.device(device_setter):
                tools.printf("Building model...")

                fc_outputs, fc_covar, se3_outputs, lstm_states, ekf_states, ekf_covar_states, _, _, _ = \
                    model.build_seq_model(self.cfg, ts_inputs[i], ts_lstm_initial_state[i], ts_initial_poses[i],
                                          ts_imu_data[i], ts_ekf_initial_state[i], ts_ekf_initial_covar[i],
                                          self.t_is_training, get_activations=True,
                                          use_initializer=self.t_use_initializer,
                                          use_ekf=self.cfg.use_ekf, fc_labels=ts_fc_labels[i])

                # this returns lstm states as a tuple, we need to stack them
                lstm_states = tf.stack(lstm_states, 0)
                ts_lstm_states.append(lstm_states)
                ts_ekf_states.append(ekf_states)
                ts_ekf_covar_states.append(ekf_covar_states)

                with tf.variable_scope("loss"):
                    se3_loss, se3_xyz_loss, se3_quat_loss \
                        = losses.se3_losses(se3_outputs, ts_se3_labels[i], self.cfg.k_se3)
                    fc_loss, fc_xyz_loss, fc_ypr_loss, x_loss, y_loss, z_loss \
                        = losses.fc_losses(fc_outputs, fc_covar, ts_fc_labels[i], self.cfg.k_fc)
                    total_loss = (1 - self.t_alpha) * se3_loss + self.t_alpha * fc_loss

                    for k, v in ts_losses_dict.items():
                        v.append(locals()[k])

                tf.get_variable_scope().reuse_variables()

        with tf.variable_scope("tower_join"), tf.device("/cpu:0"):
            # join the lstm states
            self.t_lstm_states = tf.concat(ts_lstm_states, 2)
            for k, v in ts_losses_dict.items():
                ts_losses_dict[k] = tf.reduce_mean(v)

            self.t_ekf_states = tf.concat(ts_ekf_states, 0)
            self.t_ekf_covar_states = tf.concat(ts_ekf_covar_states, 0)

            self.t_total_loss = ts_losses_dict["total_loss"]
            self.t_se3_loss = ts_losses_dict["se3_loss"]

        tools.printf("Building optimizer...")
        with tf.variable_scope("optimizer", reuse=tf.AUTO_REUSE):
            if self.cfg.use_init and self.cfg.only_train_init:
                train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "initializer_layer")
            elif self.cfg.train_noise_covariance and self.cfg.static_nn:
                train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "imu_noise_params")
            else:
                train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

            self.op_trainer = tf.train.AdamOptimizer(learning_rate=self.t_lr). \
                minimize(self.t_total_loss, colocate_gradients_with_ops=True, var_list=train_vars)

        # tensorboard summaries
        tools.printf("Building tensorboard summaries...")
        with tf.device("/cpu:0"):
            self.t_sequence_id = tf.placeholder(dtype=tf.uint8, shape=[])
            self.t_epoch = tf.placeholder(dtype=tf.int32, shape=[])

        tf.summary.scalar("total_loss", ts_losses_dict["total_loss"])
        tf.summary.scalar("fc_loss", ts_losses_dict["fc_loss"])
        tf.summary.scalar("se3_loss", ts_losses_dict["se3_loss"])
        tf.summary.scalar("fc_xyz_loss", ts_losses_dict["fc_xyz_loss"])
        tf.summary.scalar("fc_ypr_loss", ts_losses_dict["fc_ypr_loss"])
        tf.summary.scalar("se3_xyz_loss", ts_losses_dict["se3_xyz_loss"])
        tf.summary.scalar("se3_quat_loss", ts_losses_dict["se3_quat_loss"])
        tf.summary.scalar("x_loss", ts_losses_dict["x_loss"])
        tf.summary.scalar("y_loss", ts_losses_dict["y_loss"])
        tf.summary.scalar("z_loss", ts_losses_dict["z_loss"])
        tf.summary.scalar("alpha", self.t_alpha)
        tf.summary.scalar("lr", self.t_lr)
        tf.summary.scalar("sequence_id", self.t_sequence_id)
        tf.summary.scalar("epoch", self.t_epoch)
        self.op_train_merged_summary = tf.summary.merge_all()

        activations = tf.get_collection(tf.GraphKeys.ACTIVATIONS)
        initial_layer = tf.summary.image("1st layer activations", tf.expand_dims(activations[0][:, 0, :, :], -1))
        final_layer = tf.summary.image("Last layer activations", tf.expand_dims(activations[1][:, 0, :, :], -1))
        self.op_train_image_summary = tf.summary.merge([initial_layer, final_layer])

        val_loss_sum = tf.summary.scalar("val_total_loss", ts_losses_dict["total_loss"])
        val_fc_sum = tf.summary.scalar("val_fc_losses", ts_losses_dict["fc_loss"])
        val_se3_sum = tf.summary.scalar("val_se3_losses", ts_losses_dict["se3_loss"])
        val_fc_xyz_loss = tf.summary.scalar("val_fc_xyz_loss", ts_losses_dict["fc_xyz_loss"])
        val_fc_ypr_loss = tf.summary.scalar("val_fc_ypr_loss", ts_losses_dict["fc_ypr_loss"])
        val_se3_xyz_loss = tf.summary.scalar("val_se3_xyz_loss", ts_losses_dict["se3_xyz_loss"])
        val_se3_quat_loss = tf.summary.scalar("val_se3_quat_loss", ts_losses_dict["se3_quat_loss"])
        val_x_sum = tf.summary.scalar("val_x_loss", ts_losses_dict["x_loss"])
        val_y_sum = tf.summary.scalar("val_y_loss", ts_losses_dict["y_loss"])
        val_z_sum = tf.summary.scalar("val_z_loss", ts_losses_dict["z_loss"])
        self.op_val_merged_summary = tf.summary.merge(
                [val_loss_sum, val_fc_sum, val_se3_sum, val_fc_xyz_loss, val_fc_ypr_loss, val_se3_xyz_loss,
                 val_se3_quat_loss, val_x_sum, val_y_sum, val_z_sum])

    # validation loss
    def __run_val_loss(self, i_epoch, alpha_set):
        curr_lstm_states = np.zeros([2, self.cfg.lstm_layers, self.cfg.batch_size, self.cfg.lstm_size])
        curr_ekf_state = np.zeros([self.cfg.batch_size, 17])
        curr_ekf_cov_state = 0.01 * np.repeat(np.expand_dims(np.identity(17, dtype=np.float32), axis=0),
                                              repeats=self.cfg.batch_size, axis=0)

        self.val_data_gen.next_epoch(randomize=False)

        val_se3_losses_log = np.zeros([self.val_data_gen.total_batches()])

        use_init_val = self.cfg.use_init

        while self.val_data_gen.has_next_batch():
            j_batch = self.val_data_gen.curr_batch()

            # never resets state, there will only be one sequence in validation
            batch_id, curr_seq, batch_data, fc_ground_truth, se3_ground_truth, imu_measurements = self.val_data_gen.next_batch()

            init_poses = se3_ground_truth[0, :, :]

            _curr_lstm_states, _curr_ekf_states, _curr_ekf_covar, _summary, _total_losses, _se3_losses = \
                self.tf_session.run(
                        [self.t_lstm_states, self.t_ekf_states, self.t_ekf_covar_states,
                         self.op_val_merged_summary, self.t_total_loss, self.t_se3_loss],
                        feed_dict={
                            self.t_inputs: batch_data,
                            self.t_se3_labels: se3_ground_truth[1:, :, :],
                            self.t_fc_labels: fc_ground_truth,
                            self.t_lstm_initial_state: curr_lstm_states,
                            self.t_initial_poses: init_poses,
                            self.t_alpha: alpha_set,
                            self.t_is_training: False,
                            self.t_use_initializer: use_init_val,
                            self.t_ekf_initial_state: curr_ekf_state,
                            self.t_ekf_initial_covariance: curr_ekf_cov_state,
                            self.t_imu_data: imu_measurements
                        },
                        options=self.tf_run_options,
                        run_metadata=self.tf_run_metadata)

            use_init_val = False

            curr_lstm_states = np.stack(_curr_lstm_states, 0)
            curr_ekf_state = _curr_ekf_states
            curr_ekf_cov_state = _curr_ekf_covar
            self.tf_tb_writer.add_summary(_summary, i_epoch * self.val_data_gen.total_batches() + j_batch)
            val_se3_losses_log[j_batch] = _se3_losses

        return np.average(val_se3_losses_log)

    @staticmethod
    def __set_from_schedule(schedule, i_epoch):
        switch_points = np.array(list(schedule.keys()))
        set_point = switch_points[switch_points <= i_epoch].max()
        return schedule[set_point]

    def __run_train(self):
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        if self.cfg.debug:
            self.tf_session = tf_debug.LocalCLIDebugWrapperSession(tf.Session(config=sess_config))
        else:
            self.tf_session = tf.Session(config=sess_config)

        self.tf_session.run(tf.global_variables_initializer())
        if self.restore_file:
            tools.printf("Restoring model weights from %s..." % self.restore_file)
            self.tf_saver_restore.restore(self.tf_session, self.restore_file)

        else:
            if self.cfg.use_init and self.cfg.only_train_init:
                raise ValueError("Set to only train initializer, but restore file was not provided!?!?!")
            tools.printf("Initializing variables...")

        # initialize tensorboard writer
        self.tf_tb_writer = tf.summary.FileWriter(os.path.join(self.results_dir_path, 'graph_viz'))
        self.tf_tb_writer.add_graph(tf.get_default_graph())
        self.tf_tb_writer.flush()

        # initialize lstm and ekf states
        curr_lstm_states = np.zeros([2, self.cfg.lstm_layers, self.cfg.batch_size, self.cfg.lstm_size],
                                    dtype=np.float32)

        curr_ekf_state = np.zeros([self.cfg.batch_size, 17], dtype=np.float32)
        curr_ekf_cov_state = self.cfg.ekf_initial_state_covariance * \
                             np.repeat(np.expand_dims(np.identity(17, dtype=np.float32), axis=0),
                                       repeats=self.cfg.batch_size, axis=0)

        lstm_states_dic = {}
        ekf_states_dic = {}
        ekf_cov_states_dic = {}

        for seq in self.train_sequences:
            lstm_states_dic[seq] = np.zeros(
                    [self.train_data_gen.batch_counts[seq], 2, self.cfg.lstm_layers, self.cfg.batch_size,
                     self.cfg.lstm_size], dtype=np.float32)

        if self.restore_ekf_state_file:
            ekf_states_dic = pickle.load(open(self.restore_ekf_state_file + ".pickle", "rb"))
            ekf_cov_states_dic = pickle.load(open(self.restore_ekf_state_file + ".cov.pickle", "rb"))
            tools.printf("Restore ekf states from %s" % self.restore_ekf_state_file)
        else:
            for seq in self.train_sequences:
                ekf_states_dic[seq] = np.zeros([self.train_data_gen.batch_counts[seq], self.cfg.batch_size, 17],
                                               dtype=np.float32)
                ekf_cov_states_dic[seq] = np.repeat(np.expand_dims(curr_ekf_cov_state, axis=0),
                                                    self.train_data_gen.batch_counts[seq], axis=0)
            tools.printf("Using default ekf states")

        _train_image_summary = None
        total_batches = self.train_data_gen.total_batches()
        best_val_loss = 9999999999
        i_epoch = 0

        for i_epoch in range(self.start_epoch, self.cfg.num_epochs):
            tools.printf("Training Epoch: %d ..." % i_epoch)
            start_time = time.time()

            alpha_set = Train.__set_from_schedule(self.cfg.alpha_schedule, i_epoch)
            lr_set = Train.__set_from_schedule(self.cfg.lr_schedule, i_epoch)
            tools.printf("alpha set to %f" % alpha_set)
            tools.printf("learning rate set to %f" % lr_set)

            while self.train_data_gen.has_next_batch():
                j_batch = self.train_data_gen.curr_batch()

                # get inputs
                batch_id, curr_seq, batch_data, fc_ground_truth, se3_ground_truth, imu_measurements = self.train_data_gen.next_batch()
                data_roller.get_init_lstm_state(lstm_states_dic, curr_lstm_states, curr_seq, batch_id,
                                                self.cfg.bidir_aug)

                data_roller.get_init_ekf_states(ekf_states_dic, ekf_cov_states_dic, curr_ekf_state, curr_ekf_cov_state,
                                                curr_seq, batch_id, self.cfg.bidir_aug)
                if (self.cfg.gt_init_vel_state and self.cfg.gt_init_vel_state_only_first and j_batch == 0 and
                    i_epoch == 0) or \
                        (self.cfg.gt_init_vel_state and not self.cfg.gt_init_vel_state_only_first):
                    curr_ekf_state[:, 3] = fc_ground_truth[0, :, 0] * 10

                # shift se3 ground truth to be relative to the first pose
                init_poses = se3_ground_truth[0, :, :]

                nrnd = np.random.rand(1)
                use_init_train = False
                if self.cfg.use_init and (j_batch == 0 or nrnd < self.cfg.init_prob):
                    use_init_train = True

                # Run training session
                _, _curr_lstm_states, _curr_ekf_states, _curr_ekf_covar, _train_summary, _train_image_summary, _total_losses = \
                    self.tf_session.run(
                            [self.op_trainer, self.t_lstm_states, self.t_ekf_states, self.t_ekf_covar_states,
                             self.op_train_merged_summary, self.op_train_image_summary,
                             self.t_total_loss],
                            feed_dict={
                                self.t_inputs: batch_data,
                                self.t_se3_labels: se3_ground_truth[1:, :, :],
                                self.t_fc_labels: fc_ground_truth,
                                self.t_lstm_initial_state: curr_lstm_states,
                                self.t_initial_poses: init_poses,
                                self.t_lr: lr_set,
                                self.t_alpha: alpha_set,
                                self.t_is_training: True,
                                self.t_use_initializer: use_init_train,
                                self.t_sequence_id: int(curr_seq),
                                self.t_epoch: i_epoch,
                                self.t_ekf_initial_state: curr_ekf_state,
                                self.t_ekf_initial_covariance: curr_ekf_cov_state,
                                self.t_imu_data: imu_measurements
                            },
                            options=self.tf_run_options,
                            run_metadata=self.tf_run_metadata)

                data_roller.update_lstm_state(lstm_states_dic, _curr_lstm_states, curr_seq, batch_id)
                data_roller.update_ekf_state(ekf_states_dic, ekf_cov_states_dic, _curr_ekf_states, _curr_ekf_covar,
                                             curr_seq, batch_id)

                if self.tensorboard_meta:
                    self.tf_tb_writer.add_run_metadata(self.tf_run_metadata,
                                                       'epochid=%d_batchid=%d' % (i_epoch, j_batch))
                self.tf_tb_writer.add_summary(_train_summary, i_epoch * total_batches + j_batch)

                # print stats
                tools.printf("batch %d/%d: Loss:%.7f" % (j_batch + 1, total_batches, _total_losses))

            self.tf_tb_writer.add_summary(_train_image_summary, (i_epoch + 1) * total_batches)

            tools.printf("Evaluating validation loss...")
            curr_val_loss = self.__run_val_loss(i_epoch, alpha_set)

            # check for best results
            if curr_val_loss < best_val_loss:
                tools.printf("Saving best result...")
                best_val_loss = curr_val_loss
                self.tf_saver_best.save(self.tf_session, os.path.join(self.best_val_path, "model_best_val_checkpoint"),
                                        global_step=i_epoch)
                tools.printf("Best val loss, model saved.")
                pickle.dump(ekf_states_dic, open(os.path.join(self.best_val_path,
                                                              "best_val_ekf_states-%d.pickle" % i_epoch), "wb"))
                pickle.dump(ekf_cov_states_dic, open(os.path.join(self.best_val_path,
                                                                  "best_val_ekf_states-%d.cov.pickle" % i_epoch), "wb"))
            if i_epoch % 5 == 0:
                tools.printf("Saving checkpoint...")
                self.tf_saver_checkpoint.save(self.tf_session,
                                              os.path.join(self.model_epoch_path, "model_epoch_checkpoint"),
                                              global_step=i_epoch)
                tools.printf("Checkpoint saved")
                pickle.dump(ekf_states_dic, open(os.path.join(self.model_epoch_path,
                                                              "model_epoch_ekf_states-%d.pickle" % i_epoch), "wb"))
                pickle.dump(ekf_cov_states_dic, open(os.path.join(self.best_val_path,
                                                                  "best_val_ekf_states-%d.cov.pickle" % i_epoch), "wb"))

            self.tf_tb_writer.flush()
            tools.printf("ave_val_loss(se3): %f, time: %f\n" % (curr_val_loss, time.time() - start_time))

            self.train_data_gen.next_epoch()

        tools.printf("Final save...")
        self.tf_saver_checkpoint.save(self.tf_session, os.path.join(self.model_epoch_path, "model_epoch_checkpoint"),
                                      global_step=i_epoch)
        tools.printf("Saved results to %s" % self.results_dir_path)
        pickle.dump(ekf_states_dic,
                    open(os.path.join(self.model_epoch_path, "model_epoch_ekf_states-%d.pickle" % i_epoch), "wb"))
        pickle.dump(ekf_cov_states_dic,
                    open(os.path.join(self.model_epoch_path, "model_epoch_ekf_states-%d.cov.pickle" % i_epoch), "wb"))

        self.tf_session.close()
