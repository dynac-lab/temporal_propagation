import config as cfg
import matplotlib.pyplot as plt
import numpy as np

plt.figure()
plt.plot([0,1,2,3], [0,1,2,3], linestyle='-', c='k', label='Novel')
plt.plot([0,1,2,3], [3,2,1,3], linestyle=':', c='k', label='Familiar')
plt.legend()
plt.savefig(cfg.dir_fig + 'Paper/supplementary/' + "anticipation_plot_with_legend.png")
plt.savefig(cfg.dir_fig + 'Paper/supplementary/' + "anticipation_plot_with_legend.svg")
plt.savefig(cfg.dir_fig + 'Paper/supplementary/' + "anticipation_plot_with_legend.pdf")

