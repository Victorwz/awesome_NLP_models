import numpy as np
import matplotlib.pyplot as plt

frac = np.zeros(10)
frac[0] = 0.1
for i in range(1, len(frac)):
	frac[i] = frac[i-1] + 0.1 

train_perplexity  = [101.60, 103.35, 102.75, 103.03, 103.63, 102.47, 101.87, 101.57, 101.38, 101.43]

val_perplexity = [601.93, 436.71, 362.35, 320.87, 295.11, 275.76, 262.86, 250.52, 240.85, 232.10]

plt.plot(frac, train_perplexity, color = "r", linestyle = "-", marker = "^", linewidth = 1, label = "train")

plt.plot(frac, val_perplexity, color = "b", linestyle = "-", marker = "s", linewidth = 1, label = "validation")

plt.legend(loc='upper center', bbox_to_anchor=(0.6,0.95))

plt.xlabel("train_fraction")

plt.ylabel("perplexity")

plt.title("perplexity versus train_fraction")

plt.savefig("frac.pdf", dpi = 200)