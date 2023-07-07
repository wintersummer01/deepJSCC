from tqdm import tqdm
import numpy as np
import tensorflow as tf
from utils.datasets import getTrainDataset
from models.model_tfc import jsccEnd2End_tfc

# Datasets
dataset_root = '/workspace/dataset/cifar-10-batches-py'
train_ds, shape = getTrainDataset(dataset_root)

# Hyperparameters
SNR_train_dB = [1, 4, 7, 13, 19]
bandwidth_compression_ratio = [6, 12]
lr = 1e-3
num_epoch = 1000

for BW in bandwidth_compression_ratio:
    for SNR in SNR_train_dB:
        log_dir = f'./logs/BW{BW}_{SNR}dB'
        summary_writer = tf.summary.create_file_writer(log_dir)
        model = jsccEnd2End_tfc(shape, 1/BW, SNR)
        mse_loss = tf.keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        best_psnr = 0

        for i in tqdm(range(num_epoch)):
            losses = []
            PSNRs = []
            for image in train_ds:
                # Forward
                with tf.GradientTape() as tape:
                    output = model(image)
                    loss = mse_loss(image, output)
                # Backward
                grads = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                # Accuracy
                losses.append(loss)
                PSNRs.append(tf.image.psnr(image, output, max_val=255))

            # Save Model
            avg_psnr = np.mean(PSNRs)
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                model.save_weights(f'{log_dir}/weights')
            # Tensorboard Record
            with summary_writer.as_default():
                tf.summary.scalar('average loss', np.mean(losses), i+1)
                tf.summary.scalar('average PSNR', avg_psnr, i+1)
