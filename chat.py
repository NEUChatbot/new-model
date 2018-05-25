"""
Script for chatting with a trained chatbot model
"""
import datetime
from os import path

import general_utils
import chat_command_handler
from chat_settings import ChatSettings
from chatbot_model import ChatbotModel
from vocabulary import Vocabulary

class ChatSession:

    def __init__(self):
        #Read the hyperparameters and configure paths
        _, model_dir, hparams, checkpoint = general_utils.initialize_session("chat")

        #Load the vocabulary
        print()
        print ("Loading vocabulary...")
        if hparams.model_hparams.share_embedding:
            shared_vocab_filepath = path.join(model_dir, Vocabulary.SHARED_VOCAB_FILENAME)
            input_vocabulary = Vocabulary.load(shared_vocab_filepath)
            output_vocabulary = input_vocabulary
        else:
            input_vocab_filepath = path.join(model_dir, Vocabulary.INPUT_VOCAB_FILENAME)
            input_vocabulary = Vocabulary.load(input_vocab_filepath)
            output_vocab_filepath = path.join(model_dir, Vocabulary.OUTPUT_VOCAB_FILENAME)
            output_vocabulary = Vocabulary.load(output_vocab_filepath)

        #Create the model
        print ("Initializing model...")
        print()
        model = ChatbotModel(mode = "infer",
                          model_hparams = hparams.model_hparams,
                          input_vocabulary = input_vocabulary,
                          output_vocabulary = output_vocabulary,
                          model_dir = model_dir)

        #Load the weights
        print()
        print ("Loading model weights...")
        model.load(checkpoint)
        self.model = model

        # Setting up the chat
        self.chatlog_filepath = path.join(model_dir, "chat_logs", "chatlog_{0}.txt".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
        self.chat_settings = ChatSettings(hparams.inference_hparams)
        chat_command_handler.print_commands()

    def chat(self, question):
        is_command, terminate_chat = chat_command_handler.handle_command(question, self.model, self.chat_settings)
        if terminate_chat:
            return "Terminate is not support in wechat model."
        elif not is_command:
            #If it is not a command (it is a question), pass it on to the chatbot model to get the answer
            question_with_history, answer = self.model.chat(question, self.chat_settings)
            
            #Print the answer or answer beams and log to chat log
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
        print("BOt: {}".format(session.chat(q)))