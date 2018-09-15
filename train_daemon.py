import os
import time

if __name__ == '__main__':
    while True:
        os.system(r'python train.py')
        with open('log.txt', mode='a') as f:
            f.write('train stop at {}\n'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
