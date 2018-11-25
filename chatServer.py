import os
import sys
import time
import queue
import train
import requests
import datetime
import threading
import jieba
import http.server
from os import path
from hparams import Hparams
from cgi import parse_header
from urllib.parse import parse_qs
from chat_settings import ChatSettings
from chat import ChatSession


waiting_queue = queue.Queue()
chat_setting = None
result_queue = dict()


class Struct():
    def __init__(self, id, data):
        self.id = id
        self.data = data


ids = set()


class ServerClass(http.server.CGIHTTPRequestHandler):
    def __init__(self, request, client_address, server):
        super().__init__(request, client_address, server)

    def do_POST(self):
        ctype, pdict = parse_header(self.headers['content-type'])
        if ctype == 'application/x-www-form-urlencoded':
            length = int(self.headers['content-length'])
            postvars = parse_qs(self.rfile.read(length).decode(), keep_blank_values=1)
            question = ' '.join(jieba.cut(postvars['question'][0]))
            print('chat :{} {}'.format(postvars['id'], question))
            if postvars['id'][0] in ids:
                print('ignore')
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'success')
                return
            ids.add(postvars['id'][0])
            waiting_queue.put(Struct(postvars['id'][0], question))
            start_time = time.time()
            while time.time() - start_time < 10 and result_queue.get(postvars['id'][0], None) is None:
                pass
            response = result_queue.pop(postvars['id'][0], 'server timeout :(')
            self.send_response(200)
            self.end_headers()
            self.wfile.write(response.encode())
            ids.remove(postvars['id'][0])
        else:
            self.send_response(200)
            self.end_headers()
            self.wfile.write('wrong')

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write('ok'.encode())


class ChatServer(object):
    def __init__(self, training):
        if training:
            checkpointfile = r'models\best_weights_training.ckpt'
            # Make sure checkpoint file & hparams file exists
            checkpoint_filepath = os.path.relpath(checkpointfile)
            model_dir = os.path.dirname(checkpoint_filepath)
            hparams = Hparams()
            global chat_setting
            # Setting up the chat
            self.chatlog_filepath = path.join(model_dir, "chat_logs", "chatlog_{0}.txt".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
            chat_setting = self.chat_settings = ChatSettings(hparams.inference_hparams)
            # chat_command_handler.print_commands()
            self.train_thred = threading.Thread(target=train.train, args=(waiting_queue, chat_setting, result_queue))
            self.train_thred.start()
        else:
            def server_thread_function():
                sess = ChatSession()
                while True:
                    if not waiting_queue.empty():
                        q = waiting_queue.get()
                        if q.data == 'version':
                            t = os.path.getmtime('models/best_weights_training.ckpt.data-00000-of-00001')
                            result_queue[q.id] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t))

                        else:
                            result_queue[q.id] = sess.chat(q.data)
                        print(result_queue[q.id])
            threading.Thread(target=server_thread_function).start()

        try:
            myip = requests.get('http://fun.alphamj.cn/wx/registered').content.decode()
        except:
            myip = '127.0.0.1'
        print('listen {}:4321'.format(myip))
        self.server = http.server.HTTPServer((myip, 4321), ServerClass)
        print('server init finish')


if __name__ == '__main__':
    jieba.initialize()
    s = ChatServer(len(sys.argv) > 1 and sys.argv[1] == 'train')
    s.server.serve_forever()
