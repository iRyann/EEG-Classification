import numpy as np
    
def add_gaussian_noise(x):
    x_noised =  x.copy()
    x_std = np.std(x) * 0.05
    noise = np.ramdom.normal(0, x_std, x.shape)
    x_noised += noise
    return x_noised

def temporal_shift(x):
    time_shift = np.random.randint(-2, 3)
    if time_shift != 0:
        return np.roll(x, shift=time_shift, axis=2)
    
def rescale(x):
        scale_factor = np.random.uniform(0.9, 1.1)
        return x * scale_factor

def augment_eeg_data(x, y, augmentation_factor=2):
    print("=== AUGMENTATION DE DONNÉES ===")
    x_res = [x]
    y_res = [y]

    for _ in range(augmentation_factor):
        x_temp = add_gaussian_noise(x)
        x_temp = temporal_shift(x_temp)
        x_temp = rescale(x_temp)
        x_res.append(x_temp)
        y_res.append(y.copy())

    x_res = np.concatenate(x_res, axis=0)
    y_res = np.concatenate(y_res, axis=0)

    return x_res, y_res
    