from tqdm import tqdm
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils.datasets import getTestDataset
from models.model_tfc import jsccEnd2End_tfc

dataset_root = '/workspace/dataset/cifar-10-batches-py'
test_ds, shape = getTestDataset(dataset_root)

BW = 6
train_SNRs = [1, 4, 7, 13, 19]
plot_color = ['#0072bd', '#d95319', '#edb120', '#7e2f8e', '#77ac30']
plot_marker = ['-*', '-|', '-o', '-s', '-d']
test_SNRs = [1, 4, 7, 10, 13, 16, 19 ,22, 25]

for i, train_SNR in enumerate(train_SNRs):
    model_dir = f'./logs/BW{BW}_{train_SNR}dB/weights'
    avg_PSNR = []

    for test_SNR in tqdm(test_SNRs):
        test_model = jsccEnd2End_tfc(shape, 1/BW, test_SNR)
        test_model.load_weights(model_dir)
        PSNRs = []
        for image in test_ds:
            output = test_model(image)
            PSNRs.append(tf.image.psnr(image, output, max_val=255))
        avg_PSNR.append(np.mean(PSNRs))    
    plt.plot(test_SNRs, avg_PSNR, plot_marker[i], color=plot_color[i], label=f'deepJSCC (SNR_train = {train_SNR}dB)')

plt.title(f'AWGN Channel (k/n = 1/{BW}) [self made]')
plt.xlabel('SNR_test (dB)')
plt.ylabel('PSNR (dB)')
plt.axis((0, 25, 20, 36))
plt.legend()
plt.show()
