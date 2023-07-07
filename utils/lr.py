import tensorflow as tf

class lrStepScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr):
        self.lr = initial_lr

    def __call__(self, step):
        if step == 700:
            self.lr *= 0.1
        return self.lr
