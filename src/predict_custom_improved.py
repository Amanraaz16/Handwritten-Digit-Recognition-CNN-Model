# src/predict_custom_improved.py
import sys
import cv2
import numpy as np
import tensorflow as tf

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return img

def center_of_mass_shift(img28):
    M = cv2.moments(img28)
    if M["m00"] == 0:
        return 0, 0  # no shift needed if no mass detected
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    shift_x = int(round(14 - cx))
    shift_y = int(round(14 - cy))
    return shift_x, shift_y

def preprocess_for_mnist(img):
    # Resize large input
    h, w = img.shape
    max_dim = max(h, w)
    if max_dim > 800:
        scale = 800 / max_dim
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    # Gaussian blur reduces noise
    blur = cv2.GaussianBlur(img, (5,5), 0)

    # Adaptive threshold (invert -> digit = white, background = black)
    th = cv2.adaptiveThreshold(blur, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV,
                               25, 10)

    # Find digit contour
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        digit = th
    else:
        c = max(contours, key=cv2.contourArea)
        x, y, ww, hh = cv2.boundingRect(c)
        pad = int(0.2 * max(ww, hh))
        digit = th[max(0, y-pad):min(h, y+hh+pad),
                   max(0, x-pad):min(w, x+ww+pad)]

    # Resize to fit into 20x20 while keeping aspect ratio
    h2, w2 = digit.shape
    if h2 > w2:
        new_h = 20
        new_w = int((w2 * 20) / h2)
    else:
        new_w = 20
        new_h = int((h2 * 20) / w2)

    digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Paste into 28x28 canvas
    canvas = np.zeros((28,28), dtype=np.uint8)
    start_x = (28 - new_w) // 2
    start_y = (28 - new_h) // 2
    canvas[start_y:start_y+new_h, start_x:start_x+new_w] = digit

    # Center-of-mass shift
    shift_x, shift_y = center_of_mass_shift(canvas)
    M = np.float32([[1,0,shift_x],[0,1,shift_y]])
    shifted = cv2.warpAffine(canvas, M, (28,28))

    # Normalize to [0,1]
    final = shifted.astype(np.float32) / 255.0

    return final

def predict(image_path, model_path="models/mnist_cnn.h5"):
    model = tf.keras.models.load_model(model_path)
    img = load_image(image_path)
    processed = preprocess_for_mnist(img)
    processed = processed.reshape(1, 28, 28, 1)
    pred = model.predict(processed)
    digit = int(np.argmax(pred))
    return digit, pred.flatten()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_custom_improved.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    digit, probs = predict(image_path)
    print("\nPredicted Digit:", digit)
    print("Probabilities:", np.round(probs, 4))
