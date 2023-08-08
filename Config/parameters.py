#training
num_epoch = 300
batch_size = 32
gpu = 2
smoothing_value = 0.85
isReduced = False
device_cpu = False
output_channel = 1
device_ids = [2, 3]
allow_bbox = True
mixed_precision = True

train_conf_th = 0.7
val_conf_th = 0.7
test_conf_th = 0.7

#optimizer
learning_rate = 0.00005

#augmentation
height = 512
width = 512
scale = (0.75, 0.75)
ratio=(1, 1)
scale_factor = 0.5


#model related
checkpoint = 1      #no of epoch to save model

#visualization
visualize_epoch_freq = 3
print_stats = 40
visualize_mask = 20

visualize_mask_val = 5
print_stats_val = 10

visualize_mask_test = 1

# Training Resume
resume = True