from IPython.display import display, FileLink

python_code = """import numpy as np
import tensorflow as tf
import os
from skimage.color import rgb2lab, lab2rgb
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, Concatenate, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# --- CONFIGURATION ---
IMG_SIZE = 128
DATASET_PATH = "/kaggle/input/unsplash-lite-5k-colorization/train/color/"
MAX_IMAGES_TO_LOAD = 4000
EPOCHS = 50
BATCH_SIZE = 32
CHECKPOINT_PATH = "/kaggle/working/RECOVERED.weights.h5"

# --- 1. DATA LOADING & PREPROCESSING ---
def load_and_preprocess_data(max_images):
    image_files = sorted(os.listdir(DATASET_PATH))[:max_images]
    X_l, Y_ab = [], []
    print(f"Loading and processing {len(image_files)} images...")
    
    for filename in tqdm(image_files):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_rgb = imread(os.path.join(DATASET_PATH, filename))
            if img_rgb.ndim == 2:
                img_rgb = np.stack((img_rgb,)*3, axis=-1)
            
            img_resized = resize(img_rgb, (IMG_SIZE, IMG_SIZE), anti_aliasing=True)
            img_lab = rgb2lab(img_resized)
            
            X_l.append(img_lab[:, :, 0] / 100.0)
            Y_ab.append(img_lab[:, :, 1:] / 128.0)
            
    X_l = np.array(X_l, dtype='float32').reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    Y_ab = np.array(Y_ab, dtype='float32')
    print(f"Data shapes ready. Input X_l: {X_l.shape}, Target Y_ab: {Y_ab.shape}")
    return X_l, Y_ab

X_l, Y_ab = load_and_preprocess_data(MAX_IMAGES_TO_LOAD)

# --- 2. U-NET MODEL ARCHITECTURE ---
def build_colorization_unet(input_shape):
    inputs = Input(shape=input_shape)
    
    def conv_block(input_tensor, filters, use_bn=True):
        x = Conv2D(filters, (3, 3), padding='same')(input_tensor)
        if use_bn: x = BatchNormalization()(x)
        return Activation('relu')(x)

    conv1 = conv_block(inputs, 64, use_bn=False); conv1 = conv_block(conv1, 64); pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv_block(pool1, 128); conv2 = conv_block(conv2, 128); pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = conv_block(pool2, 256); conv3 = conv_block(conv3, 256)
    
    up4 = UpSampling2D(size=(2, 2))(conv3); up4 = Concatenate()([up4, conv2]); conv4 = conv_block(up4, 128); conv4 = conv_block(conv4, 128)
    up5 = UpSampling2D(size=(2, 2))(conv4); up5 = Concatenate()([up5, conv1]); conv5 = conv_block(up5, 64); conv5 = conv_block(conv5, 64)
    
    outputs = Conv2D(2, (1, 1), activation='tanh', padding='same')(conv5)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse')
    return model

model = build_colorization_unet((IMG_SIZE, IMG_SIZE, 1))

# --- 3. TRAINING EXECUTION ---
checkpoint = ModelCheckpoint(CHECKPOINT_PATH, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)

print("\\nStarting model training...")
history = model.fit(
    X_l, Y_ab, 
    epochs=EPOCHS, 
    batch_size=BATCH_SIZE, 
    verbose=1, 
    validation_split=0.1, 
    callbacks=[checkpoint, early_stopping]
)

# --- 4. FORCED SAVING ---
model.save_weights(CHECKPOINT_PATH)
print(f"\\nTraining complete! Weights successfully saved to: {CHECKPOINT_PATH}")
"""

# Write the python string to a .py file
file_name = "colorization_training.py"
with open(file_name, "w") as f:
    f.write(python_code)

# Create the direct download link
print("Python file successfully created! Click the link below to download it:")
display(FileLink(file_name))