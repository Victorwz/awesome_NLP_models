import numpy as np
import matplotlib.pyplot as plt

opt_perplexity  = [152.70, 41.32, 15.68, 2.31, 2.44]

best_lr = [0.0003, 0.0003, 0.0003, 0.0001, 0.00003]

dim = [1, 5, 10, 100, 200]

#plt.plot(dim, opt_perplexity, color = "r", linestyle = "-", marker = "^", linewidth = 1, label = "train")

plt.plot(dim, opt_perplexity, color = "r", linestyle = "-", marker = "s", linewidth = 1)

#plt.legend(loc='upper center', bbox_to_anchor=(0.6,0.95))

plt.xlabel("hidden_dim")

plt.ylabel("optimized_val_perplexity")

plt.title("optimized val ppl versus hidden dim, nlayer = 0")

plt.savefig("dim_linear.pdf", dpi = 200)

