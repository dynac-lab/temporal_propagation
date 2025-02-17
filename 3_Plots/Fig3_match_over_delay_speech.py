import matplotlib.pyplot as plt
import numpy as np
import config as cfg
from funcs import p_to_str
import matplotlib.gridspec as gridspec


dir_save = cfg.dir_fig + 'Paper/'
dir_results = cfg.dir_analysis_results + "2_alignment_speech/"
box_width = 0.3

## Load results
matches_relative = np.load(dir_results + 'matches_relative_over_delays.npy', allow_pickle=True).item()
optimal_delays = np.load(dir_results + 'optimal_delays.npy', allow_pickle=True).item()
delays = cfg.speech_delays

## Create figure
f = plt.figure(figsize=(cfg.fig_width['two_column'] * 1.5, 3.5 * 1.5))
gs = gridspec.GridSpec(2,3,width_ratios = [3,3,1], height_ratios=[5,1])
ax_clause_low_lines = f.add_subplot(gs[0])
ax_clause_high_lines = f.add_subplot(gs[1], sharey=ax_clause_low_lines)
ax_delays_comparison = f.add_subplot(gs[2])
ax_clause_low_delays = f.add_subplot(gs[3], sharex=ax_clause_low_lines)
ax_clause_high_delays = f.add_subplot(gs[4], sharex=ax_clause_high_lines)

# Gridlines
ax_delays_comparison.set_axisbelow(True)
ax_delays_comparison.grid(axis='y', color=cfg.grid_color)
ax_clause_low_lines.set_axisbelow(True)
ax_clause_low_lines.grid(axis='x', color=cfg.grid_color)
ax_clause_low_delays.set_axisbelow(True)
ax_clause_low_delays.grid(axis='x', color=cfg.grid_color)
ax_clause_high_lines.set_axisbelow(True)
ax_clause_high_lines.grid(axis='x', color=cfg.grid_color)
ax_clause_high_delays.set_axisbelow(True)
ax_clause_high_delays.grid(axis='x', color=cfg.grid_color)


def plot_individual_curves(curve_per_subject):
    grayscale = plt.cm.gray(np.linspace(0.5, 0.95, len(curve_per_subject.keys())))
    curves_all = []
    for n_idx, n in enumerate(curve_per_subject.keys()):
        curve = curve_per_subject[n]
        if len(curve) == 0:
            continue
        curves_all.append(curve)
        plt.plot(delays,curve,c=grayscale[n_idx], zorder=1,linewidth=2)
        plt.scatter(optimal_delays[n][ROI]['clauses'], curve[np.where(delays == optimal_delays[n][ROI]['clauses'])[0][0]], color=cfg.colors_ROI[ROI], zorder=3,  edgecolors='k')
    curve_mean = np.mean(np.asarray(curves_all), axis=0)
    plt.plot(delays,curve_mean, c='blue',zorder=2, linewidth=2)
    plt.axhline(0, color='k', linestyle='--', zorder=0)

def boxplot_scatter(numbers):
    plt.boxplot(numbers,vert=False, positions=[0], zorder=1, patch_artist=True, boxprops={'facecolor': 'white'},
                medianprops={'color': cfg.colors_ROI[ROI], 'linewidth': 3}, widths=0.5)
    plt.scatter(numbers, np.zeros(len(numbers)), color=cfg.colors_ROI[ROI],  edgecolors='k', zorder=2)


ROI = 'low'
plt.sca(ax_clause_low_lines)
plot_individual_curves(matches_relative['clauses'][ROI])
plt.ylabel('Relative Gaussian match')
plt.title("A) Match over delay in ROI low", loc='left')
ax_clause_low_lines.xaxis.set_tick_params(which='major',labelbottom=False)


plt.sca(ax_clause_low_delays)
boxplot_scatter([optimal_delays[n][ROI]['clauses'] for n in optimal_delays.keys() if not(n==12 and ROI == 'low')])
plt.xlabel('Delay (s)')
plt.yticks([],[])


ROI = 'high'
plt.sca(ax_clause_high_lines)
plot_individual_curves(matches_relative['clauses'][ROI])
plt.title("B) Match over delay in ROI high", loc='left')
ax_clause_high_lines.xaxis.set_tick_params(which='major',labelbottom=False)
ax_clause_high_lines.yaxis.set_tick_params(which='major',labelleft=False)




plt.sca(ax_clause_high_delays)
boxplot_scatter([optimal_delays[n][ROI]['clauses'] for n in optimal_delays.keys() if not(n==12 and ROI == 'low')])
plt.xlabel('Delay (s)')
plt.yticks([],[])



# Optimal delay difference
plt.sca(ax_delays_comparison)
for n in optimal_delays.keys():
    x = [0, 1]
    colors_ROIs = [cfg.colors_ROI['low'], cfg.colors_ROI['high']]
    if n == 12:
        x = [1]
        y = optimal_delays[n]['high']['clauses']
        colors_ROIs = [cfg.colors_ROI['high']]
        if np.max(matches_relative['clauses']['high'][n]) > 0:
            color = 'k'
            zorder = 1
        else:
            color = 'lightgray'
            zorder = 0
    else:
        y = [optimal_delays[n]['low']['clauses'], optimal_delays[n]['high']['clauses']]
        if np.max(matches_relative['clauses']['low'][n]) > 0 and np.max(
                matches_relative['clauses']['high'][n]) > 0:
            color = 'k'
            zorder = 1
        else:
            color = 'lightgray'
            zorder = 0

    plt.plot(x, y, '-', c=color, zorder=zorder)
    plt.scatter(x=x,y=y,facecolors=colors_ROIs, edgecolors='k')

# Add significance
p_value = np.load(cfg.dir_analysis_results +  '2_alignment_speech/wilcoxon_delaydiff_pvalues.npy').item()
max_value = np.max([optimal_delays[n][ROI]['clauses'] for n in optimal_delays.keys() for ROI in ['low', 'high'] if not(n == 12 and ROI == 'low')])
y1 = max_value + 0.025
y2 = y1 + 0.03
y3 = y2
y4 = y1
x1 = 0
x2 = 0
x3=1
x4 = 1
plt.plot([x1,x2,x3,x4], [y1,y2,y3,y4], linewidth=1, color='k')
x_text = (x1 + x3)/2
y_text = y2 + 0.01
stars = p_to_str(p_value)
plt.text(x=x_text,y=y_text,s=stars, ha='center')


plt.xticks(x, ['ROI low', 'ROI high'])
plt.ylim([-0.05, 0.68])
plt.xlim([-0.2,1.2])
plt.ylabel('Optimal delay with clauses (s)')
plt.title("C) Optimal delay\ncomparison", loc='left')

plt.tight_layout()
plt.savefig(dir_save + 'fig3_match_over_delay_speech.png')
plt.savefig(dir_save + 'fig3_match_over_delay_speech.pdf')
plt.savefig(dir_save + 'fig3_match_over_delay_speech.svg')