# This script requires the following packages to be installed:
# pip install tensorflow numpy scikit-image pillow opencv-python

import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import os
import cv2
from PIL import Image, ImageTk, ImageEnhance, ImageOps, ImageFilter 

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from skimage.color import rgb2lab, lab2rgb

# --- CONFIGURATION ---
IMG_SIZE = 128
# FIX: The model file is named 'RECOVERED.weights.h5'
WEIGHTS_PATH = "RECOVERED.weights.h5" 

class ColorizationApp:
    def __init__(self, master):
        self.master = master
        master.title("Image AI & Filter Studio (Python/Tkinter)")
        master.config(bg="#F3F4F6") # Tailwind gray-100

        self.model = None
        self.original_image_path = None
        self.current_pil_image = None
        self.original_pil_image = None
        
        # --- UI SETUP ---
        
        self.main_frame = tk.Frame(master, bg="#F3F4F6", padx=10, pady=10)
        self.main_frame.pack(expand=True, fill='both')

        self.controls_frame = tk.Frame(self.main_frame, bg="white", padx=20, pady=20, relief=tk.RAISED, bd=1)
        self.controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        self.image_frame = tk.Frame(self.main_frame, bg="white", relief=tk.RAISED, bd=1)
        self.image_frame.pack(side=tk.RIGHT, expand=True, fill='both')

        # Load Status
        self.status_label = tk.Label(self.controls_frame, text="Loading AI Model...", bg="white", fg="#4F46E5", font=('Arial', 10, 'bold'))
        self.status_label.pack(pady=5)

        # File Button
        self.load_button = tk.Button(self.controls_frame, text="1. Load Image", command=self.load_image, bg="#4F46E5", fg="white", font=('Arial', 12, 'bold'))
        self.load_button.pack(fill='x', pady=(10, 20))
        self.load_button.config(state=tk.DISABLED) # Disabled until model loads

        # Filter Buttons
        self.filter_label = tk.Label(self.controls_frame, text="2. Filters & AI", bg="white", font=('Arial', 12, 'bold'))
        self.filter_label.pack(pady=5)

        self.colorize_button = self.create_button("Colorize (AI Model)", self.apply_colorization, "#4F46E5")
        self.reset_button = self.create_button("Reset to Original", self.reset_image, "#6B7280")
        self.ace_button = self.create_button("Apply ACE Filter", self.apply_ace, "#10B981")
        self.emboss_button = self.create_button("Apply Emboss Filter", self.apply_emboss, "#F59E0B")
        self.hot_cold_button = self.create_button("Show Hot/Cold Spots", self.apply_hot_cold_map, "#EF4444")

        # Canvas for Image Display
        self.image_label = tk.Label(self.image_frame, bg="#E5E7EB") # Tailwind gray-200
        self.image_label.pack(expand=True, fill='both')

        self.load_model()
        
    def create_button(self, text, command, color):
        btn = tk.Button(self.controls_frame, text=text, command=command, bg=color, fg="white", font=('Arial', 10, 'bold'), relief=tk.FLAT, pady=5)
        btn.pack(fill='x', pady=5)
        btn.config(state=tk.DISABLED)
        return btn

    def toggle_filter_buttons(self, state):
        self.colorize_button.config(state=state)
        self.reset_button.config(state=state)
        self.ace_button.config(state=state)
        self.emboss_button.config(state=state)
        self.hot_cold_button.config(state=state)

    # --- MODEL LOADING (Synchronous since it's a desktop app) ---

    def load_model(self):
        try:
            # Recreate the model structure first
            input_shape = (IMG_SIZE, IMG_SIZE, 1)
            self.model = self._build_colorization_unet(input_shape)
            
            # Load weights from the h5 file
            self.model.load_weights(WEIGHTS_PATH)
            
            self.status_label.config(text="AI Model Ready!", fg="#10B981") # Tailwind green-500
            self.load_button.config(state=tk.NORMAL)

        except FileNotFoundError:
            self.status_label.config(text=f"ERROR: Weights file not found at {WEIGHTS_PATH}", fg="#EF4444")
            messagebox.showerror("Error", f"Model weights file not found. Please ensure '{WEIGHTS_PATH}' is in the correct directory.")
        except Exception as e:
            self.status_label.config(text="ERROR during model load", fg="#EF4444")
            messagebox.showerror("Error", f"Failed to load the model: {e}")

    # --- U-NET ARCHITECTURE (Must match training code) ---

    def _build_colorization_unet(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        
        def conv_block(input_tensor, filters, use_bn=True):
            x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(input_tensor)
            if use_bn:
                x = tf.keras.layers.BatchNormalization()(x) # FIXED: Removed non-commented image tag
            x = tf.keras.layers.Activation('relu')(x)
            return x

        # Encoder
        conv1 = conv_block(inputs, 64, use_bn=False); conv1 = conv_block(conv1, 64)
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1) 

        conv2 = conv_block(pool1, 128); conv2 = conv_block(conv2, 128)
        pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = conv_block(pool2, 256); conv3 = conv_block(conv3, 256)

        # Decoder
        up4 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv3)
        up4 = tf.keras.layers.Concatenate()([up4, conv2]) 
        conv4 = conv_block(up4, 128); conv4 = conv_block(conv4, 128)

        up5 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv4)
        up5 = tf.keras.layers.Concatenate()([up5, conv1]) 
        conv5 = conv_block(up5, 64); conv5 = conv_block(conv5, 64)
        
        outputs = tf.keras.layers.Conv2D(2, (1, 1), activation='tanh', padding='same')(conv5)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Colorization_UNet_BN")
        
        return model

    # --- IMAGE DISPLAY & UTILITY ---

    def load_image(self):
        self.original_image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if self.original_image_path:
            try:
                # Load with PIL (keeps the original size for display)
                self.original_pil_image = Image.open(self.original_image_path).convert('RGB')
                self.current_pil_image = self.original_pil_image.copy()
                self.display_image(self.current_pil_image)
                self.toggle_filter_buttons(tk.NORMAL)

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}")
                self.toggle_filter_buttons(tk.DISABLED)

    def display_image(self, pil_image):
        # Resize image to fit the label, maintaining aspect ratio
        w_frame = self.image_frame.winfo_width() - 20
        h_frame = self.image_frame.winfo_height() - 20
        
        if w_frame > 1 and h_frame > 1:
            pil_image.thumbnail((w_frame, h_frame), Image.Resampling.LANCZOS)
        
        self.tk_image = ImageTk.PhotoImage(pil_image)
        self.image_label.config(image=self.tk_image, width=pil_image.width, height=pil_image.height)
        self.image_label.image = self.tk_image # Keep a reference

    def reset_image(self):
        if self.original_pil_image:
            self.current_pil_image = self.original_pil_image.copy()
            self.display_image(self.current_pil_image)

    # --- AI COLORIZATION FILTER ---

    def apply_colorization(self):
        if not self.model or not self.current_pil_image:
            return

        self.status_label.config(text="Processing (Colorization)...", fg="#4F46E5")
        self.master.update() # Update GUI to show loading status

        try:
            # 1. Prepare image array (resize to 128x128 for model)
            img_array_128 = np.array(self.current_pil_image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS))
            
            # 2. Convert RGB to L*a*b* (Normalizing to 0-1 for skimage input)
            img_lab = rgb2lab(img_array_128 / 255.0) 

            # 3. Extract L channel and normalize
            L_channel = img_lab[:, :, 0]
            X_l_input = L_channel / 100.0
            X_l_input = X_l_input.reshape(1, IMG_SIZE, IMG_SIZE, 1).astype(np.float32)

            # 4. Predict a/b channels
            predicted_ab_normalized = self.model.predict(X_l_input, verbose=0)[0]
            
            # 5. Denormalize prediction
            ab_predicted_denorm = predicted_ab_normalized * 128.0

            # 6. Reassemble L*a*b* array (using original L channel data)
            full_lab_predicted = np.concatenate(
                (L_channel.reshape(IMG_SIZE, IMG_SIZE, 1), ab_predicted_denorm), axis=-1
            )
            
            # 7. Convert L*a*b* back to RGB (0-1 range)
            img_rgb_final = lab2rgb(full_lab_predicted)
            
            # 8. Convert to 0-255 PIL Image (Resizing back to original image size)
            img_rgb_255 = (img_rgb_final * 255).astype(np.uint8)
            
            # Use PIL to resize back to the original image dimensions for display
            final_pil = Image.fromarray(img_rgb_255)
            final_pil = final_pil.resize(self.original_pil_image.size, Image.Resampling.LANCZOS)
            
            self.current_pil_image = final_pil
            self.display_image(self.current_pil_image)
            self.status_label.config(text="Colorization Complete!", fg="#10B981")

        except Exception as e:
            self.status_label.config(text="Colorization Failed", fg="#EF4444")
            messagebox.showerror("Error", f"Colorization failed: {e}")

    # --- IMAGE FILTER FUNCTIONS ---

    def apply_ace(self):
        if not self.current_pil_image: return
        self.status_label.config(text="Applying ACE Filter...", fg="#4F46E5"); self.master.update()
        
        # PIL's Contrast enhancement is fast and achieves a similar effect to ACE (Adaptive Contrast Enhancement)
        enhancer = ImageEnhance.Contrast(self.current_pil_image)
        self.current_pil_image = enhancer.enhance(1.5) # Increase contrast by 50%
        self.display_image(self.current_pil_image)
        self.status_label.config(text="ACE Filter Applied!", fg="#10B981")

    def apply_emboss(self):
        if not self.current_pil_image: return
        self.status_label.config(text="Applying Emboss Filter...", fg="#4F46E5"); self.master.update()

        # FIX: Changed Image.EMBOSS to ImageFilter.EMBOSS
        self.current_pil_image = self.current_pil_image.filter(ImageFilter.EMBOSS)
        self.display_image(self.current_pil_image)
        self.status_label.config(text="Emboss Filter Applied!", fg="#10B981")

    def apply_hot_cold_map(self):
        if not self.current_pil_image: return
        self.status_label.config(text="Mapping Hot/Cold Spots...", fg="#4F46E5"); self.master.update()

        # Use OpenCV for fast color mapping (Requires cv2)
        try:
            # 1. Convert PIL image to OpenCV BGR format
            img_cv = cv2.cvtColor(np.array(self.current_pil_image), cv2.COLOR_RGB2BGR)

            # 2. Convert to Grayscale
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

            # 3. Apply the COLORMAP_JET (Hot/Cold) heatmap
            heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

            # 4. Convert back to PIL for display
            img_rgb_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            self.current_pil_image = Image.fromarray(img_rgb_heatmap)
            
            self.display_image(self.current_pil_image)
            self.status_label.config(text="Hot/Cold Map Applied!", fg="#10B981")
            
        except Exception as e:
            messagebox.showerror("Error", f"Hot/Cold Map failed (Requires OpenCV): {e}")
            self.status_label.config(text="Hot/Cold Map Failed", fg="#EF4444")


if __name__ == '__main__':
    root = tk.Tk()
    # The display function relies on frame size being known, so bind display to resize event
    app = ColorizationApp(root)
    root.bind('<Configure>', lambda event: app.display_image(app.current_pil_image) if app.current_pil_image else None)
    root.mainloop()