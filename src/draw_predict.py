import cv2
import numpy as np
import tensorflow as tf
import os
from datetime import datetime

MODEL_PATH = os.path.join("models", "mnist_cnn.h5")  # change if you use another file

# ---------- Parameters ----------
CANVAS_SIZE = 400            # window canvas for drawing
BRUSH_RADIUS = 14           # thicker brush helps recognition
LINE_COLOR = 255            # white foreground on black background
BG_COLOR = 0                # black background
THICKEN_KERNEL = (2,2)      # morphological kernel to thicken strokes if needed

# ---------- Load model ----------
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print("ERROR: Could not load model:", MODEL_PATH)
    print("Exception:", e)
    raise SystemExit(1)

# ---------- Drawing state ----------
canvas = np.ones((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8) * BG_COLOR
drawing = False
last_pt = None

def draw(event, x, y, flags, param):
    global drawing, last_pt
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        last_pt = (x, y)
        cv2.circle(canvas, (x,y), BRUSH_RADIUS, LINE_COLOR, -1)
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        if last_pt is None:
            last_pt = (x, y)
        # draw smooth line between last point and current
        cv2.line(canvas, last_pt, (x, y), LINE_COLOR, thickness=BRUSH_RADIUS*2, lineType=cv2.LINE_AA)
        last_pt = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        last_pt = None

cv2.namedWindow("Draw Digit")
cv2.setMouseCallback("Draw Digit", draw)

# ---------- Preprocessing pipeline (same ideas as predict_custom_improved) ----------
def preprocess_canvas_for_model(img_gray):
    """Take canvas (uint8) -> return 28x28 float32 normalized array"""
    # 1. Crop to content bounding box (with padding)
    coords = cv2.findNonZero(img_gray)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        pad = int(0.15 * max(w, h))
        x1 = max(0, x - pad); y1 = max(0, y - pad)
        x2 = min(img_gray.shape[1], x + w + pad); y2 = min(img_gray.shape[0], y + h + pad)
        roi = img_gray[y1:y2, x1:x2]
    else:
        # nothing drawn: return blank 28x28
        roi = img_gray

    # 2. Resize into 20x20 box preserving aspect ratio (like MNIST)
    h, w = roi.shape
    if h == 0 or w == 0:
        roi = cv2.resize(img_gray, (20,20), interpolation=cv2.INTER_AREA)
        h, w = roi.shape

    if h > w:
        new_h = 20
        new_w = max(1, int(round((w * 20) / h)))
    else:
        new_w = 20
        new_h = max(1, int(round((h * 20) / w)))

    roi_resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 3. Place into 28x28 canvas centered
    canvas28 = np.zeros((28,28), dtype=np.uint8)
    start_x = (28 - new_w) // 2
    start_y = (28 - new_h) // 2
    canvas28[start_y:start_y+new_h, start_x:start_x+new_w] = roi_resized

    # 4. Center of mass shift (match MNIST centering)
    M = cv2.moments(canvas28)
    if M["m00"] != 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        shift_x = int(round(14 - cx))
        shift_y = int(round(14 - cy))
        T = np.float32([[1,0,shift_x],[0,1,shift_y]])
        canvas28 = cv2.warpAffine(canvas28, T, (28,28))

    # 5. Optional: thicken faint strokes (small dilation)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, THICKEN_KERNEL)
    canvas28 = cv2.dilate(canvas28, kernel, iterations=1)

    # 6. Normalize to float32 [0,1] and invert if needed.
    # Our drawing uses white on black (255 foreground). MNIST training used values where strokes are dark on light background originally,
    # but our CNN was trained with x_train/255.0 where pixel values represent ink intensity.
    # We will keep white foreground as "1.0" to match previous processing (we used THRESH_BINARY_INV in file-based pipeline).
    final = canvas28.astype(np.float32) / 255.0

    return final

# ---------- Prediction helper ----------
def predict_from_canvas():
    # preprocess
    processed = preprocess_canvas_for_model(canvas)
    x = processed.reshape(1,28,28,1)
    p = model.predict(x, verbose=0).flatten()
    pred = int(np.argmax(p))
    conf = float(np.max(p))
    return pred, conf, processed

# ---------- UI loop ----------
while True:
    display = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    # Show instructions on screen
    cv2.putText(display, "Draw: LMB | p:predict | s:save28 | c:clear | q:quit", (8,20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1, cv2.LINE_AA)
    cv2.imshow("Draw Digit", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('p'):
        pred, conf, proc28 = predict_from_canvas()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Predicted: {pred}  Confidence: {conf:.3f}")
        # overlay prediction on window
        overlay = display.copy()
        cv2.putText(overlay, f"Pred: {pred}  ({conf:.2f})", (8, CANVAS_SIZE - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
        cv2.imshow("Draw Digit", overlay)
        cv2.waitKey(600)  # show overlay for short time

    if key == ord('s'):
        # save the 28x28 processed image for debugging
        _, _, proc28 = predict_from_canvas()
        save_img = (proc28 * 255).astype(np.uint8)
        cv2.imwrite("last_draw_28.png", save_img)
        print("Saved last_draw_28.png (28x28 input)")

    if key == ord('c'):
        canvas[:] = BG_COLOR
        print("Canvas cleared")

    if key == ord('q'):
        break

cv2.destroyAllWindows()
