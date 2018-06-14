import http.server
import os
import queue
import datetime
import threading
import time
from cgi import parse_header
from urllib.parse import parse_qs

import train
from os import path
from chat_settings import ChatSettings
from hparams import Hparams

waiting_queue = queue.Queue()
chat_setting = None
result_queue = queue.Queue()


class Struct():
    def __init__(self, id, data):
        self.id = id
        self.data = data


class ServerClass(http.server.CGIHTTPRequestHandler):
    def __init__(self, request, client_address, server):
        super().__init__(request, client_address, server)

    def do_POST(self):
        ctype, pdict = parse_header(self.headers['content-type'])
        if ctype == 'application/x-www-form-urlencoded':
            length = int(self.headers['content-length'])
            postvars = parse_qs(self.rfile.read(length).decode(), keep_blank_values=1)
            print('chat :{} {}'.format(postvars['id'], postvars['question']))

            waiting_queue.put(Struct(postvars['id'][0], postvars['question'][0]))
        else:
            return

class ChatServer(object):
    def __init__(self):
        checkpointfile = r'models\training_data_in_database\20180520_144933\best_weights_training.ckpt'
        # Make sure checkpoint file & hparams file exists
        checkpoint_filepath = os.path.relpath(checkpointfile)
        model_dir = os.path.dirname(checkpoint_filepath)
        hparams_filepath = os.path.join(model_dir, "hparams.json")
        hparams = Hparams.load(hparams_filepath)
        global chat_setting
        # Setting up the chat
        self.chatlog_filepath = path.join(model_dir, "chat_logs", "chatlog_{0}.txt".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
        chat_setting = self.chat_settings = ChatSettings(hparams.inference_hparams)
        # chat_command_handler.print_commands()

        self.train_thred = threading.Thread(target=train.train, args=(waiting_queue, chat_setting, result_queue))
        self.train_thred.start()
        self.handle_thread = threading.Thread(target=self.handleResult)
        self.handle_thread.start()
        self.server = http.server.HTTPServer(('127.0.0.1', 8000), ServerClass)
        print('server init finish')

    def handleResult(self):
        while True:
            if not result_queue.empty():
                r = result_queue.get()
                print('{}: {}'.format(r.id, r.data[1]))
            time.sleep(0.01)


if __name__ == '__main__':
    s = ChatServer()
    s.server.serve_forever()
