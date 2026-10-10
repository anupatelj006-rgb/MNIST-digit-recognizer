import os, sys, glob, json, time, base64, threading, subprocess, random
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(BASE, "history.json")


# ------------------------------------------------------------------ model
def find_model():
    robust = os.path.join(BASE, "model", "mnist_cnn_robust.keras")
    if os.path.exists(robust):
        return robust
    files = glob.glob(os.path.join(BASE, "model", "*.keras")) + \
            glob.glob(os.path.join(BASE, "model", "*.h5"))
    if not files:
        raise FileNotFoundError("No model in /model. Run train_model.py first.")
    return max(files, key=os.path.getmtime)


MODEL_PATH = find_model()
model = tf.keras.models.load_model(MODEL_PATH)
print("Loaded model:", MODEL_PATH)

_mnist_test = None


def mnist_test():
    global _mnist_test
    if _mnist_test is None:
        (_, _), (x, y) = tf.keras.datasets.mnist.load_data()
        _mnist_test = (x, y)
    return _mnist_test


# ------------------------------------------------------------ preprocessing
def decode_image(raw):
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 4:          # transparency -> white
        a = img[:, :, 3:4] / 255.0
        img = (img[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    if max(h, w) > 1000:                              # shrink huge phone photos
        s = 1000 / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def _build_digit(g):
    """g = grayscale picture where the digit is BRIGHT. Returns 28x28 or None."""
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    soft = cv2.GaussianBlur(g, (0, 0), 3)             # sharpen blurry edges
    g = cv2.addWeighted(g, 1.8, soft, -0.8, 0)
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    t, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask = cv2.threshold(g, max(20, t * 0.6), 255, cv2.THRESH_BINARY)

    H, W = mask.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        touches_edge = x == 0 or y == 0 or x + w >= W or y + h >= H
        if touches_edge and area > 0.25 * H * W:      # page margin / frame
            continue
        if area < max(8, 0.0015 * H * W):             # specks
            continue
        comps.append((i, area))
    if not comps:
        return None

    biggest = max(a for _, a in comps)
    keep = np.zeros_like(mask)
    for i, a in comps:
        if a >= 0.08 * biggest:                       # drops axis numbers etc.
            keep[labels == i] = 255

    ys, xs = np.where(keep > 0)
    y1, y2, x1, x2 = ys.min(), ys.max(), xs.min(), xs.max()
    box = keep[y1:y2 + 1, x1:x2 + 1]
    big_box = (y2 - y1 + 1) * (x2 - x1 + 1) > 0.4 * H * W
    if big_box and (box.mean() / 255) > 0.7:          # solid block, not a digit
        return None

    g = np.where(keep > 0, g, 0).astype(np.uint8)[y1:y2 + 1, x1:x2 + 1]
    h, w = g.shape
    s = 20.0 / max(h, w)
    g = cv2.resize(g, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), np.uint8)
    y0, x0 = (28 - g.shape[0]) // 2, (28 - g.shape[1]) // 2
    canvas[y0:y0 + g.shape[0], x0:x0 + g.shape[1]] = g
    m = cv2.moments(canvas)
    if m["m00"] > 0:                                  # centre by mass like MNIST
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        canvas = cv2.warpAffine(canvas, np.float32([[1, 0, 14 - cx], [0, 1, 14 - cy]]), (28, 28))
    return canvas.astype("float32") / 255.0


def preprocess_candidates(gray):
    """Returns [(digit28, bonus), ...] for both colour polarities."""
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]]).mean()
    preferred_invert = border > 127
    out = []
    for invert in (False, True):
        g = 255 - gray if invert else gray
        d = _build_digit(g)
        if d is not None:
            out.append((d, 1.0 if invert == preferred_invert else 0.75))
    return out


def predict_probs(digit28):
    """Average over small tilts/shifts so rotated and blurry digits are stable."""
    variants = []
    for ang in (-15, -8, 0, 8, 15):
        M = cv2.getRotationMatrix2D((14, 14), ang, 1.0)
        variants.append(cv2.warpAffine(digit28, M, (28, 28)))
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        variants.append(cv2.warpAffine(digit28,
                                       np.float32([[1, 0, dx], [0, 1, dy]]), (28, 28)))
    batch = np.stack(variants)[..., None]
    return model.predict(batch, verbose=0).mean(axis=0)


def result_json(probs):
    pred = int(np.argmax(probs))
    conf = float(probs[pred]) * 100
    return {"prediction": pred, "confidence": round(conf, 1),
            "probabilities": [round(float(p) * 100, 1) for p in probs],
            "low_confidence": conf < 50}


# ------------------------------------------------------------------ history
def load_history():
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def write_history(items):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f)


# ------------------------------------------------------------------- routes
@app.route("/")
def index():
    return render_template("index.html",
                           robust=os.path.basename(MODEL_PATH) == "mnist_cnn_robust.keras")


@app.route("/predict", methods=["POST"])
def predict():
    raw = None
    if request.files:
        raw = next(iter(request.files.values())).read()
    else:
        data = request.get_json(silent=True) or request.form
        b64 = data.get("image") if data else None
        if b64:
            raw = base64.b64decode(b64.split(",")[-1])
    if not raw:
        return jsonify(error="No image received."), 400
    gray = decode_image(raw)
    if gray is None:
        return jsonify(error="That file is not a readable image."), 400

    cands = preprocess_candidates(gray)
    if not cands:
        return jsonify(error="No digit found. Try a clearer picture."), 400

    best = None
    for digit, bonus in cands:
        probs = predict_probs(digit)
        score = float(probs.max()) * bonus
        if best is None or score > best[0]:
            best = (score, probs)
    return jsonify(result_json(best[1]))


@app.route("/sample", methods=["POST"])
def sample():
    """Type a number: for each typed digit, pick a random MNIST test image of
    that digit and let the model read it."""
    text = (request.get_json(silent=True) or {}).get("digits", "")
    digits = [c for c in text if c.isdigit()][:12]
    if not digits:
        return jsonify(error="Type at least one digit (0-9)."), 400
    x, y = mnist_test()
    out = []
    for d in digits:
        idx = random.choice(np.where(y == int(d))[0])
        img = x[idx].astype("float32") / 255.0
        r = result_json(predict_probs(img))
        ok, buf = cv2.imencode(".png", x[idx])
        r["typed"] = int(d)
        r["image"] = "data:image/png;base64," + base64.b64encode(buf).decode()
        out.append(r)
    return jsonify(items=out)


@app.route("/history")
def history():
    return jsonify(load_history())


@app.route("/save", methods=["POST"])
def save():
    d = request.get_json(silent=True) or {}
    items = load_history()
    items.insert(0, {"digit": int(d.get("digit", 0)),
                     "confidence": float(d.get("confidence", 0)),
                     "source": str(d.get("source", "draw"))[:20],
                     "time": time.strftime("%d-%m-%Y %H:%M:%S")})
    write_history(items[:200])
    return jsonify(ok=True, count=len(items[:200]))


@app.route("/history/clear", methods=["POST"])
def clear_history():
    write_history([])
    return jsonify(ok=True)


# ----------------------------------------------- open in a REAL browser window
def open_real_browser(url):
    os.environ.pop("BROWSER", None)      # VS Code sets this to its own preview
    paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    for p in paths:
        if os.path.exists(p):
            subprocess.Popen([p, "--new-window", url])
            return
    if sys.platform.startswith("win"):
        os.startfile(url)


if __name__ == "__main__":
    threading.Timer(1.5, lambda: open_real_browser("http://127.0.0.1:5000")).start()
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)