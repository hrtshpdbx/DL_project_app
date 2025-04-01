import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow.keras.models import Model
import cv2

def grad_cam_heatmap(model, img_array, layer_name, class_idx):
    """
    Generates a Grad-CAM heatmap.
    
    Args:
        model: Trained Keras model.
        img_array: Preprocessed input image (single image).
        layer_name: Name of the convolutional layer for Grad-CAM.
        class_idx: Class index for which to generate Grad-CAM.
    """
    grad_model = Model(inputs=model.input, outputs=[model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(np.expand_dims(img_array, axis=0))
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()

    # Multiply each channel by the importance weights
    for i in range(pooled_grads.shape[-1]):
        conv_output[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_output, axis=-1)
    heatmap = np.maximum(heatmap, 0)  # ReLU
    heatmap /= np.max(heatmap)

    return heatmap

def overlay_heatmap(img, heatmap, alpha=0.5):
    # Normalize heatmap to range 0-255 if it's not already
    heatmap = np.uint8(255 * (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8))

    # Apply color map
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Ensure image is uint8
    if img.dtype in [np.float32, np.float64]:
        img = (img * 255).astype(np.uint8)

    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Overlay heatmap on original image
    superimposed_img = cv2.addWeighted(img, alpha, heatmap, 1 - alpha, 0)

    return superimposed_img

def visualize_batch_norm_effect(model, layer_name, sample_input):
    """Extracts feature maps before and after a BatchNorm layer and returns histograms as arrays."""

    # Find BatchNorm layer and previous layer
    bn_layer = model.get_layer(layer_name)
    prev_layer = model.get_layer(index=model.layers.index(bn_layer) - 1)

    # Create sub-models
    model_before_bn = Model(inputs=model.input, outputs=prev_layer.output)
    model_after_bn = Model(inputs=model.input, outputs=bn_layer.output)

    # Compute activations
    activations_before = model_before_bn.predict(np.expand_dims(sample_input, axis=0))
    activations_after = model_after_bn.predict(np.expand_dims(sample_input, axis=0))

    return activations_before.flatten(), activations_after.flatten()
