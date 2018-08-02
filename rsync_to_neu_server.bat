scp models/training_data_in_database/20180520_144933/best_weights_training.ckpt.* wmj@219.216.64.117:~/new-model/models/training_data_in_database/20180520_144933
scp models/training_data_in_database/20180520_144933/checkpoint wmj@219.216.64.117:~/new-model/models/training_data_in_database/20180520_144933
scp models/training_data_in_database/20180520_144933/hparams.json wmj@219.216.64.117:~/new-model/models/training_data_in_database/20180520_144933
ssh wmj@219.216.64.117 "./restart"
