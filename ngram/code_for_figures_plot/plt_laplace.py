import numpy as np
import matplotlib.pyplot as plt

alpha = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]

log_alpha = np.zeros(7)

for i in range(len(alpha)):
	log_alpha[i] = np.log10(alpha[i]) 

train_perplexity  = [71.02, 71.57, 76.12, 101.48, 207.94, 632.73, 2068.18]

val_perplexity = [502.13, 349.94, 256.34, 232.10, 319.87, 694.12, 1946.43]

plt.plot(log_alpha, train_perplexity, color = "r", linestyle = "-", marker = "^", linewidth = 1, label = "train")

plt.plot(log_alpha, val_perplexity, color = "b", linestyle = "-", marker = "s", linewidth = 1, label = "validation")

plt.legend(bbox_to_anchor=(0.4, 0.2))

plt.xlabel("log_alpha")

plt.ylabel("perplexity")

plt.title("perplexity versus alpha")

plt.savefig("alpha.pdf", dpi = 200)