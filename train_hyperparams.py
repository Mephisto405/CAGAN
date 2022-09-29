# Constant
KNOWLEDGE_DISTILLATION_MODE = ['Output_Only', 'Intermediate']
LPIPS_IMAGE_SIZE = 256

# Params
data_folder = '/workspace/iycho/nalbi-test/ffhq_256/'
generated_img_size = 256
channel_multiplier = 2
latent = 512
n_mlp = 8
ckpt = 'Model/pruned_model/content_aware_pruned_0.7_256px_model_2022-09-29_05:11:05.pth'
load_train_state = False 

gpu_device_ids = [0,1]
primary_device = 'cuda:0'

training_iters = 10001 # 450001 
batch_size = 8 # 16
init_lr = 0.002
discriminator_r1 = 10
generator_path_reg_weight = 2
path_reg_batch_shrink = 2
g_reg_freq = 10000 #4
d_reg_freq = 16
noise_mixing = 0.9

val_sample_num = 25
val_sample_freq = 100
model_save_freq = 10000
fid_n_sample = 50000
fid_batch = 32

teacher = 'Model/full_size_model/256px_full_size.pt'
kd_l1_lambda = 3
kd_lpips_lambda = 3
kd_mode = KNOWLEDGE_DISTILLATION_MODE[0]
content_aware_KD = True
