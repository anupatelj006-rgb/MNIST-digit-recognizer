import glob, os
import numpy as np, cv2, tensorflow as tf
import matplotlib.pyplot as plt

# load your saved model
path = "model/mnist_cnn_robust.keras"
if not os.path.exists(path):
    path = max(glob.glob("model/*.keras") + glob.glob("model/*.h5"), key=os.path.getmtime)
model = tf.keras.models.load_model(path)

# 3000 MNIST test images
(_, _), (x, y) = tf.keras.datasets.mnist.load_data()
x = x[:3000].astype("float32") / 255.0
y = y[:3000]

blur = [0, 0.5, 1, 1.5, 2, 2.5, 3]
acc = []
for s in blur:
    imgs = np.array([cv2.GaussianBlur(i, (0, 0), s) if s else i for i in x])
    pred = np.argmax(model.predict(imgs[..., None], verbose=0), axis=1)
    acc.append((pred == y).mean() * 100)

plt.plot(blur, acc, marker="o", color="green")
plt.title("Accuracy vs blur")
plt.xlabel("Blur level (0 = clear image)")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.grid(alpha=0.3)
plt.savefig("model/simple_curve.png", dpi=150)
plt.show()
