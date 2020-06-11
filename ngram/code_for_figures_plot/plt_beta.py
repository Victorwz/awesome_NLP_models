import numpy as np
import matplotlib.pyplot as plt

beta = np.zeros(9)
beta[0] = 0.1
for i in range(1, len(beta)):
	beta[i] = beta[i-1] + 0.1 

opt_perplexity  = [152.70, 41,32, 15.68, 13.14, 2.44]

best_lr = [0.0003, 0.0003, 0.0003, 0.0003, 0.00003]

dim = [1, 5, 10, 100, 200]

plt.plot(beta, train_perplexity, color = "r", linestyle = "-", marker = "^", linewidth = 1, label = "train")

plt.plot(beta, val_perplexity, color = "b", linestyle = "-", marker = "s", linewidth = 1, label = "validation")

plt.legend(loc='upper center', bbox_to_anchor=(0.6,0.95))

plt.xlabel("beta")

plt.ylabel("perplexity")

plt.title("perplexity versus beta")

plt.savefig("beta.pdf", dpi = 200)