import os
import time

if __name__ == '__main__':
    while True:
        os.system(r'python train.py --checkpoint=models\training_data_in_database\20180520_144933\best_weights_training.ckpt')
        with open('log.txt', mode='a') as f:
            f.write('train stop at {}\n'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
