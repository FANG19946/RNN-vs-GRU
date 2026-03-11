import numpy as np
import matplotlib.pyplot as plt

z = np.load("A2_mem_rnn_tanh_clip005_test_final_state.npz")

grad_time = z["grad_time"]

g = grad_time[-1]
g = g[np.isfinite(g)]

vals = np.log10(g + 1e-12)

print("Mean:", np.mean(vals))
print("Variance:", np.var(vals))
print("Std Dev:", np.std(vals))

print("\nHigh precision stats")
print("Mean:", format(np.mean(vals), ".12f"))
print("Variance:", format(np.var(vals), ".12f"))
print("Std Dev:", format(np.std(vals), ".12f"))

print("Unique gradient values:", np.unique(g)[:10])
print("Number of unique values:", len(np.unique(g)))