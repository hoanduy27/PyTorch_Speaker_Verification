# import shutil
# import glob
# from tqdm import tqdm

# SOURCE_PATH = '/home/duckhoan/Documents/Code/icassp-2020-se-rvae/data/vivos/train/waves'
# DEST_PATH = '/home/duckhoan/Documents/Code/PyTorch_Speaker_Verification/vivos_noise'

# folders = glob.glob(SOURCE_PATH + '/*')

# for folder in tqdm(folders):
#     folder_name = folder.split('/')[-1]
#     des = DEST_PATH + '/' + folder_name
    
#     files = glob.glob(folder + '/*.wav')
#     for file in files:
#         shutil.copy2(file, des)
epochs = []
total_loss = []
with open('./speech_id_checkpoint_contrastive_vivos/Stats', 'r') as f:
    lines = f.readlines()
    stats = [line.split('\t') for line in lines]
    for stat in stats:
        if '[4/4]' in stat[1]:
            epoch = int(stat[1].split(',')[0].split(':')[-1].split('[')[0])
            epochs.append(epoch)
            loss = float(stat[-2].split(':')[-1])
            total_loss.append(loss)
        else:
            continue

import numpy as np
import matplotlib.pyplot as plt

x = np.array(epochs)
y = np.array(total_loss)

plt.plot(x, y)

plt.xlabel("Epochs")
plt.ylabel("Total loss")

plt.show()
    


