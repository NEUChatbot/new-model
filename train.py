"""
Script for training the chatbot model
"""
import os
import time
import math
from os import path

from chatbot_model import ChatbotModel
from database_reader import DataBaseReader
from hparams import Hparams
from training_stats import TrainingStats

basedir = './'
def train(waiting_queue=None, chat_setting=None, result_queue=None):
    # Read the hyperparameters and paths
    checkpoint_filepath = os.path.join(basedir, 'models/best_weights_training.ckpt')

    resume_checkpoint = None  # 重新训练
    #  resume_checkpoint = os.path.basename(checkpoint_filepath)  # 继续训练
    model_dir = os.path.dirname(checkpoint_filepath)
    dataset_dir = os.path.join(basedir, "datasets")
    training_stats_filepath = path.join(model_dir, "training_stats.json")
    hparams = Hparams()

    # Read the chatbot dataset
    dataset_reader = DataBaseReader()

    print()
    print("Reading dataset '{0}'...".format(dataset_reader.dataset_name))
    dataset, _ = dataset_reader.read_dataset(dataset_dir=dataset_dir,
                                             model_dir=model_dir,
                                             training_hparams=hparams.training_hparams,
                                             share_vocab=hparams.model_hparams.share_embedding)

    # Split the chatbot dataset into training & validation datasets
    print("Splitting {0} samples into training & validation sets ({1}% used for validation)..."
          .format(dataset.size(), hparams.training_hparams.validation_set_percent))

    training_dataset, validation_dataset = dataset.train_val_split(
        val_percent=hparams.training_hparams.validation_set_percent,
        random_split=hparams.training_hparams.random_train_val_split)
    training_dataset_size = training_dataset.size()
    validation_dataset_size = validation_dataset.size()
    print("Training set: {0} samples. Validation set: {1} samples."
          .format(training_dataset_size, validation_dataset_size))

    print("Sorting training & validation sets to increase training efficiency...")
    training_dataset.sort()
    validation_dataset.sort()

    # Create the model
    print("Initializing model...")
    print()
    with ChatbotModel(mode="train",
                      model_hparams=hparams.model_hparams,
                      input_vocabulary=dataset.input_vocabulary,
                      output_vocabulary=dataset.output_vocabulary,
                      model_dir=model_dir) as model:
        print()

        # Restore from checkpoint if specified
        training_stats = TrainingStats(hparams.training_hparams)
        if resume_checkpoint is not None:
            print("Resuming training from checkpoint {0}...".format(resume_checkpoint))
            model.load(resume_checkpoint)
            training_stats.load(training_stats_filepath)
        else:
            print("Initializing training...")

        if hparams.model_hparams.share_embedding:
            print("Shared Vocab size: {0}".format(dataset.input_vocabulary.size()))
        else:
            print("Input Vocab size: {0}".format(dataset.input_vocabulary.size()))
            print("Output Vocab size: {0}".format(dataset.output_vocabulary.size()))
        print("Epochs: {0}".format(hparams.training_hparams.epochs))
        print("Batch Size: {0}".format(hparams.training_hparams.batch_size))

        best_train_checkpoint = "best_weights_training.ckpt"
        best_val_checkpoint = "best_weights_validation.ckpt"

        # Train on all batches in epoch
        print_counter = 0
        for epoch in range(1, hparams.training_hparams.epochs + 1):
            batch_counter = 0
            batches_starting_time = time.time()
            batches_total_train_loss = 0
            epoch_starting_time = time.time()
            epoch_total_train_loss = 0
            train_batches = training_dataset.batches(hparams.training_hparams.batch_size)
            batch_index = 0
            for questions, answers, seqlen_questions, seqlen_answers, emotion in train_batches:

                batch_train_loss = model.train_batch(inputs=questions,
                                                     targets=answers,
                                                     input_sequence_length=seqlen_questions,
                                                     target_sequence_length=seqlen_answers,
                                                     emotion_values=emotion,
                                                     learning_rate=training_stats.learning_rate,
                                                     dropout=hparams.training_hparams.dropout,
                                                     global_step=training_stats.global_step,
                                                     log_summary=hparams.training_hparams.log_summary)
                batches_total_train_loss += batch_train_loss
                epoch_total_train_loss += batch_train_loss
                batch_counter += 1
                training_stats.global_step += 1
                if waiting_queue is not None and hparams.training_hparams.response_in_training:
                    if not waiting_queue.empty():
                        task = waiting_queue.get()
                        with model.as_infer():
                            result = model.chat(task.data, chat_setting)
                            print('result: {}'.format(result))
                            result_queue[task.id] = result

                if batch_counter == hparams.training_hparams.stats_after_n_batches or batch_index == (
                        training_dataset_size // hparams.training_hparams.batch_size):
                    batches_average_train_loss = batches_total_train_loss / batch_counter
                    epoch_average_train_loss = epoch_total_train_loss / (batch_index + 1)
                    print(
                        'Epoch: {:>3}/{}, Batch: {:>4}/{}, Stats for last {} batches: (Training Loss: {:>6.3f}, Training Time: {:d} seconds), Stats for epoch: (Training Loss: {:>6.3f}, Training Time: {:d} seconds)'.format(
                            epoch,
                            hparams.training_hparams.epochs,
                            batch_index + 1,
                            math.ceil(training_dataset_size / hparams.training_hparams.batch_size),
                            batch_counter,
                            batches_average_train_loss,
                            int(time.time() - batches_starting_time),
                            epoch_average_train_loss,
                            int(time.time() - epoch_starting_time)))
                    batches_total_train_loss = 0
                    batch_counter = 0
                    print_counter += 1
                    batches_starting_time = time.time()
                batch_index += 1

                # End of epoch activities
                # Run validation
                if print_counter == hparams.training_hparams.verify_after_n_prints:
                    print_counter = 0
                    if validation_dataset_size > 0:
                        total_val_metric_value = 0
                        batches_starting_time = time.time()
                        val_batches = validation_dataset.batches(hparams.training_hparams.batch_size)
                        for batch_index_validation, (questions, answers, seqlen_questions, seqlen_answers, _) in enumerate(
                                val_batches):
                            batch_val_metric_value = model.validate_batch(inputs=questions,
                                                                          targets=answers,
                                                                          input_sequence_length=seqlen_questions,
                                                                          target_sequence_length=seqlen_answers,
                                                                          metric=hparams.training_hparams.validation_metric)
                            total_val_metric_value += batch_val_metric_value
                        average_val_metric_value = total_val_metric_value / math.ceil(
                            validation_dataset_size / hparams.training_hparams.batch_size)
                        print('Epoch: {:>3}/{}, Validation {}: {:>6.3f}, Batch Validation Time: {:d} seconds'.format(
                            epoch,
                            hparams.training_hparams.epochs,
                            hparams.training_hparams.validation_metric,
                            average_val_metric_value,
                            int(time.time() - batches_starting_time)))

                    # Apply learning rate decay
                    if hparams.training_hparams.learning_rate_decay > 0:
                        prev_learning_rate, learning_rate = training_stats.decay_learning_rate()
                        print('Learning rate decay: adjusting from {:>6.3f} to {:>6.3f}'.format(prev_learning_rate, learning_rate))
                    if hparams.training_hparams.checkpoint_on_training:
                        model.save(best_train_checkpoint)
                        training_stats.save(training_stats_filepath)
                        print('model saved')
                    # Checkpoint - training
                    if training_stats.compare_training_loss(epoch_average_train_loss):

                        print('Training loss improved!')

                    # Checkpoint - validation
                    if validation_dataset_size > 0:
                        if training_stats.compare_validation_metric(average_val_metric_value):
                            if hparams.training_hparams.checkpoint_on_validation:
                                model.save(best_val_checkpoint)
                                training_stats.save(training_stats_filepath)
                            print('Validation {0} improved!'.format(hparams.training_hparams.validation_metric))
                        else:
                            if training_stats.early_stopping_check == hparams.training_hparams.early_stopping_epochs:
                                print(
                                    "Early stopping checkpoint reached - validation loss has not improved in {0} epochs. Terminating training...".format(
                                        hparams.training_hparams.early_stopping_epochs))
                                break

        # Training is complete... if no checkpointing was turned on, save the final model state
        if not hparams.training_hparams.checkpoint_on_training and not hparams.training_hparams.checkpoint_on_validation:
            model.save(best_train_checkpoint)
            model.save(best_val_checkpoint)
            training_stats.save(training_stats_filepath)
            print('Model saved.')
        print("Training Complete!")


if __name__ == '__main__':
    train()
    os.system('../shutdown.sh')
