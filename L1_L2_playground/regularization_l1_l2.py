import numpy as np

# # data
# x1 = np.array([2, 4, 6], dtype=float)
# x2 = np.array([5, -3, 2], dtype=float)
# y  = x1*2
# y  = np.array([-9, -4, 1], dtype=float)


# from "https://www.youtube.com/watch?v=iqXEnO2a-no"
x1 = np.array([3, 4, 1], dtype=float)
x2 = np.array([2.8, 4.1, 1.3], dtype=float)
y  = np.array([6, 8, 2], dtype=float) # y  = x1*2

# params
w1 = 10.0
w2 = 10.0

lr = 0.001
lam = 7.5  # strong L1 to see effect

for epoch in range(10000):
    y_pred = w1 * x1 + w2 * x2
    error = y_pred - y

    # MSE gradients
    number_of_params = len(x1)
    grad_w1 = (2/number_of_params) * np.sum(error * x1)
    grad_w2 = (2/number_of_params) * np.sum(error * x2)

    # L1 gradients
    grad_w1 += lam * np.sign(w1)
    grad_w2 += lam * np.sign(w2)

    # update
    w1 -= lr * grad_w1
    w2 -= lr * grad_w2

    print(f"Epoch {epoch+1}: w1={w1:.3f}, w2={w2:.3f}")

    if epoch > 250:
        continue






# import numpy as np
# import matplotlib.pyplot as plt
#
# # -----------------------
# # DATA
# # -----------------------
# x1 = np.array([2, 4, 6], dtype=float)
# x2 = np.array([-5, -3, -2], dtype=float)
# y  = x1 * 2
#
# # -----------------------
# # LOSS FUNCTION
# # -----------------------
# def loss_fn(w1, w2, lam=0.0):
#     y_pred = w1 * x1 + w2 * x2
#     mse = np.mean((y_pred - y) ** 2)
#     l1 = lam * (abs(w1) + abs(w2))
#     return mse + l1
#
# # -----------------------
# # FIX one param, vary the other
# # -----------------------
# fixed_w2 = 0.0
# fixed_w1 = 2.0  # near true solution
#
# w1_range = np.linspace(-2, 10, 200)
# w2_range = np.linspace(-10, 5, 200)
#
# # compute losses
# loss_w1_no_reg = [loss_fn(w1, fixed_w2, lam=0.0) for w1 in w1_range]
# loss_w1_l1     = [loss_fn(w1, fixed_w2, lam=0.5) for w1 in w1_range]
#
# loss_w2_no_reg = [loss_fn(fixed_w1, w2, lam=0.0) for w2 in w2_range]
# loss_w2_l1     = [loss_fn(fixed_w1, w2, lam=0.5) for w2 in w2_range]
#
# # -----------------------
# # PLOTS
# # -----------------------
# plt.figure(figsize=(12, 5))
#
# # ---- Plot 1: Loss vs w1 ----
# plt.subplot(1, 2, 1)
# plt.plot(w1_range, loss_w1_no_reg, label="No Reg")
# plt.plot(w1_range, loss_w1_l1, label="L1 Reg")
# plt.axvline(2, linestyle="--", label="True w1")
# plt.xlabel("w1")
# plt.ylabel("Loss")
# plt.title("Loss vs w1")
# plt.legend()
#
# # ---- Plot 2: Loss vs w2 ----
# plt.subplot(1, 2, 2)
# plt.plot(w2_range, loss_w2_no_reg, label="No Reg")
# plt.plot(w2_range, loss_w2_l1, label="L1 Reg")
# plt.axvline(0, linestyle="--", label="True w2")
# plt.xlabel("w2")
# plt.ylabel("Loss")
# plt.title("Loss vs w2")
# plt.legend()
#
# plt.tight_layout()
# plt.show()
