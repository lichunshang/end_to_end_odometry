import model

inputs, lstm_initial_state, initial_poses, is_training, fc_outputs, se3_outputs, lstm_states = model.build_seq_training_model()
se3_labels, fc_labels = model.model_labels(cfg)
