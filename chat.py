"""
Script for chatting with a trained chatbot model
"""
import datetime
from os import path
import os

import jieba

import chat_command_handler
from chat_settings import ChatSettings
from chatbot_model import ChatbotModel
from vocabulary import Vocabulary
from hparams import Hparams


def initialize_session(mode):
    """Helper method for initializing a chatbot training session
    by loading the model dir from command line args and reading the hparams in

    Args:
        mode: "train" or "chat"
    """

    # Make sure script was run in the correct working directory
    models_dir = "models"
    datasets_dir = "datasets"
    if not os.path.isdir(models_dir) or not os.path.isdir(datasets_dir):
        raise NotADirectoryError(
            "Cannot find models directory 'models' and datasets directory 'datasets' within working directory '{0}'. Make sure to set the working directory to the chatbot root folder."
                .format(os.getcwd()))

    checkpointfile = r'models/training_data_in_database/20180520_144933/best_weights_training.ckpt'
    # Make sure checkpoint file & hparams file exists
    checkpoint_filepath = os.path.relpath(checkpointfile)
    if not os.path.isfile(checkpoint_filepath + ".meta"):
        raise FileNotFoundError(
            "The checkpoint file '{0}' was not found.".format(os.path.realpath(checkpoint_filepath)))
    # Get the checkpoint model directory
    checkpoint = os.path.basename(checkpoint_filepath)
    model_dir = os.path.dirname(checkpoint_filepath)
    dataset_name = os.path.basename(os.path.dirname(model_dir))
    dataset_dir = os.path.join(datasets_dir, dataset_name)

    # Load the hparams from file
    hparams_filepath = os.path.join(model_dir, "hparams.json")
    hparams = Hparams.load(hparams_filepath)

    return dataset_dir, model_dir, hparams, checkpoint


class ChatSession:

    def __init__(self):
        # Read the hyperparameters and configure paths
        _, model_dir, hparams, checkpoint = initialize_session("chat")

        # Load the vocabulary
        print()
        print("Loading vocabulary...")
        if hparams.model_hparams.share_embedding:
            shared_vocab_filepath = path.join(model_dir, Vocabulary.SHARED_VOCAB_FILENAME)
            input_vocabulary = Vocabulary.load(shared_vocab_filepath)
            output_vocabulary = input_vocabulary
        else:
            input_vocab_filepath = path.join(model_dir, Vocabulary.INPUT_VOCAB_FILENAME)
            input_vocabulary = Vocabulary.load(input_vocab_filepath)
            output_vocab_filepath = path.join(model_dir, Vocabulary.OUTPUT_VOCAB_FILENAME)
            output_vocabulary = Vocabulary.load(output_vocab_filepath)

        # Create the model
        print("Initializing model...")
        print()
        model = ChatbotModel(mode="infer",
                             model_hparams=hparams.model_hparams,
                             input_vocabulary=input_vocabulary,
                             output_vocabulary=output_vocabulary,
                             model_dir=model_dir)

        # Load the weights
        print()
        print("Loading model weights...")
        model.load(checkpoint)
        self.model = model

        # Setting up the chat
        self.chatlog_filepath = path.join(model_dir, "chat_logs",
                                          "chatlog_{0}.txt".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
        self.chat_settings = ChatSettings(hparams.inference_hparams)
        chat_command_handler.print_commands()

    def chat(self, question):
        is_command, terminate_chat = chat_command_handler.handle_command(question, self.model, self.chat_settings)
        if terminate_chat:
            return "Terminate is not supported in wechat model."
        elif not is_command:
            # If it is not a command (it is a question), pass it on to the chatbot model to get the answer
            question_with_history, answer = self.model.chat(question, self.chat_settings)

            # Print the answer or answer beams and log to chat log
            if self.chat_settings.show_question_context:
                return "Question with history (context): {0}".format(question_with_history)

            if self.chat_settings.show_all_beams:
                for i in range(len(answer)):
                    return "ChatBot (Beam {0}): {1}".format(i, answer[i])
            else:
                return format(answer)

            if self.chat_settings.inference_hparams.log_chat:
                chat_command_handler.append_to_chatlog(self.chatlog_filepath, question, answer)


if __name__ == '__main__':
    session = ChatSession()
    while True:
        q = input("You: ")
        q = ' '.join(jieba.cut(q))
        print("Bot: {}".format(session.chat(q)))
