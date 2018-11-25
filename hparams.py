"""
Hyperparameters class
"""

class Hparams(object):
    def __init__(self):
        """Initializes the Hparams instance.
        """
        self.model_hparams = ModelHparams()
        self.training_hparams = TrainingHparams()
        self.inference_hparams = InferenceHparams()


class ModelHparams(object):
    def __init__(self):
        """Initializes the ModelHparams instance.
        """
        self.rnn_cell_type = "lstm"
        self.rnn_size = 256
        self.use_bidirectional_encoder = True
        self.encoder_num_layers = 2
        self.decoder_num_layers = 2
        self.encoder_embedding_size = 256
        self.decoder_embedding_size = 256
        self.share_embedding = True
        self.attention_type = "scaled_luong"
        self.beam_width = 10
        self.enable_sampling = False
        self.max_gradient_norm = 5.
        self.gpu_dynamic_memory_growth = True

class TrainingHparams(object):
    def __init__(self):
        """Initializes the TrainingHparams instance.
        """        
        self.min_question_words = 1
        self.max_question_answer_words = 30
        self.max_conversations = -1
        self.conv_history_length = 6
        self.input_vocab_threshold = 2
        self.output_vocab_threshold = 2
        self.validation_set_percent = 0.01
        self.random_train_val_split = True
        self.validation_metric = "loss"
        self.epochs = 500
        self.early_stopping_epochs = 500
        self.batch_size = 128
        self.learning_rate = 1.0
        self.learning_rate_decay = 0.99
        self.min_learning_rate = 0.001
        self.dropout = 0.2
        self.checkpoint_on_training = True
        self.checkpoint_on_validation = True
        self.log_summary = False
        self.stats_after_n_batches = 5
        self.verify_after_n_prints = 30

        
class InferenceHparams(object):
    def __init__(self):
        """Initializes the InferenceHparams instance.
        """        
        self.beam_length_penalty_weight = 1.25
        self.sampling_temperature = 0.5
        self.max_answer_words = 100
        self.conv_history_length = 6
        self.log_summary = False
        self.log_chat = True

