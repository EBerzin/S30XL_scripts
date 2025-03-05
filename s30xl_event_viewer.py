import sqlite3
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import moyal
from collections import Counter
from utils import *
from plotting_tools import *
import os
import argparse
import pickle

colors = ['#0088EE', '#66CC00', '#6600CC', '#CC6600', '#00CCCC', '#CC0000', '#CC00CC', '#FFEE00', '#00CC00', '#0000CC', '#00CC66', '#CC0066', '#A3FF47', '#850AFF', '#85FF0A', '#A347FF']

parser = argparse.ArgumentParser(
                    prog='s30xl_event_viewer.py',
                    description='Event viewer for s30xl data. Press right/left key to move between events.')

parser.add_argument('-f', '--event_file', type=str, default="",
                    help=".pkl file with awkward array holding charge split by event")
parser.add_argument('-n', '--fig_name', type=str, default="",
                    help="Name of figure")
parser.add_argument('-s', '--nsamples', type=int, default=12,
                    help = "Number of samples per event")
parser.add_argument('-ch8', '--eight_channel', action='store_true', default=False,
                    help = "Parse to 8-channel format")
#parser.add_argument('-old_fmt', '--new_format', action='store_false', default=True,
#                    help = "Interpret database in old format (without ror_timestamp)")
#parser.add_argument('-use_bc', '--by_bunch_count', action='store_true', default=False,
#                    help = "Use consecutive bunch counts rather than timestamps to filter events")
parser.add_argument('-min_s', '--min_samp', type=int, default=0,
                    help = "Minimum sample to plot")
parser.add_argument('-max_s', '--max_samp', type=int, default=None,
                    help = "Maximum sample to plot")
parser.add_argument('-pf', '--pedestal_file', type=str, default=None,
                    help = ".txt file with pedestals")
parser.add_argument('-gf', '--gain_file', type=str, default=None,
                    help = ".txt file with gains")
parser.add_argument('-th', '--cluster_threshold', type=int, default=30,
                    help = "Treshold used to add hit to clusters.")


def slice_when(predicate, iterable):
  i, x, size = 0, 0, len(iterable)
  while i < size-1:
    if predicate(iterable[i], iterable[i+1]):
      yield iterable[x:i+1]
      x = i + 1
    i += 1
  yield iterable[x:size]

def get_clusters(PE_array, seed_threshold, cluster_threshold, eight_channel = False):
    if (eight_channel): PE_array[1] = cluster_threshold + 1
    active_channels = np.where(PE_array > cluster_threshold)[0]
    # Remapping to cluster coordinates
    active_channels_map = np.where(active_channels <= 5, 2 * (active_channels) + 1, active_channels)
    active_channels_map = np.where(active_channels > 5, (active_channels - 6)*2, active_channels_map)
    
    # Getting clusters
    clusters = list(slice_when(lambda x,y: y - x > 2, np.sort(active_channels_map)))
    clusters = ak.from_iter(clusters)

    gt_4_hits = ak.count(clusters, axis = -1) > 4
    no_hits_in_top = ak.sum(clusters % 2 == 0, axis = -1) == 0
    no_hits_in_bottom = ak.sum(clusters % 2 == 1, axis = -1) == 0

    # Converting back to indices
    cluster_indices = ak.where(clusters % 2 == 0, clusters/2 + 6, clusters)
    cluster_indices = ak.where(clusters % 2 == 1, (clusters - 1) / 2, cluster_indices)
    if (eight_channel): PE_array[1] = 0
    all_channels = ak.from_numpy(np.tile(PE_array,(len(clusters),1)))

    cluster_indices = ak.values_astype(cluster_indices, "int64")
    #for_slicing = ak.broadcast_arrays(all_channels, cluster_indices)
    pe_values = all_channels[cluster_indices]

    no_seed = ak.sum(pe_values > seed_threshold, axis = -1) == 0

    pe_values_top = all_channels[cluster_indices[cluster_indices <= 5]]
    pe_values_bottom = all_channels[cluster_indices[cluster_indices > 5]]

    cluster_pe_sum = ak.sum(pe_values, axis = -1)
    cluster_pe_sum_top = ak.sum(pe_values_top, axis = -1)
    cluster_pe_sum_bottom = ak.sum(pe_values_bottom, axis = -1)

    return cluster_indices, gt_4_hits, no_hits_in_top, no_hits_in_bottom, no_seed

def event_viewer(split_array, idx, fig, axes, nsamples, eight_channel, gains, peds, min_samp, max_samp):
    channel_charge = []
    cluster_colors = ['green', 'blue', 'gray', 'orange']
        
    for i in range(12):
        lane = SIX_CHANNEL_LIST[i] // 6
        ch = SIX_CHANNEL_LIST[i] % 6
        if (eight_channel) :
            lane = EIGHT_CHANNEL_LIST[i] // 8
            ch = EIGHT_CHANNEL_LIST[i] % 8
        plot_id = i
        #charge_array = adc_to_Q(np.array(df['adc%d'%(ch)][(df['lane'] == lane)]))
        #split_array = group_bunches(df[df['lane'] == lane], np.array(charge_array), by_bunch_count=by_bunch_count, new_format = new_format, ror_length = ror_length)
        charge_array_sum = ak.Array(split_array[i])[ak.num(split_array[i]) >= nsamples,min_samp:max_samp][idx]
        PE_array_sum = (ak.sum(charge_array_sum) - (peds[i] * (max_samp - min_samp))) / gains[i]
        channel_charge.append(PE_array_sum)
        if (lane == 0) : c = 'k'
        if (lane == 1) : c = 'r'
        axes[plot_id].plot(np.arange(min_samp, max_samp), (charge_array_sum - peds[i]) / gains[i], color = c)
        textstr = 'PE: %.2f'%(PE_array_sum)
        axes[plot_id].text(0.6, 0.95, textstr, transform=axes[plot_id].transAxes,verticalalignment='top')
        axes[plot_id].set_ylim(-1, 80)
        axes[plot_id].grid()
        axes[plot_id].set_xticks(np.arange(min_samp, max_samp, 2))
        #axes[plot_id].legend()
        if i < 6: 
            axes[plot_id].set_xticklabels([])
        if (i != 0 and i != 6):
            axes[plot_id].set_yticklabels([])
        if (i >= 6): axes[plot_id].set_xlabel('Sample')
    axes[0].set_ylabel('PEs')
    axes[6].set_ylabel('PEs')

    # Here we do the clustering
    cluster_indices, gt_4_hits, no_hits_in_top, no_hits_in_bottom, no_seed = get_clusters(np.array(channel_charge), 25, 10)
    print(cluster_indices)
    

    for c_idx,c in enumerate(cluster_indices):
        cluster_charge = np.sum(np.array(channel_charge)[c])
        cluster_charge_top = np.sum(np.array(channel_charge)[c[c <= 5]])
        cluster_charge_bottom = np.sum(np.array(channel_charge)[c[c > 5]])
        for h_idx,h in enumerate(c):
            if h <= 5: 
                row = 'top'
                row_charge = cluster_charge_top
            else: 
                row = 'bot.'
                row_charge = cluster_charge_bottom
            textstr = '\n'.join((
                r'$f(%s) = %.2f$' % (row, channel_charge[h] / row_charge),
                r'$f(tot.)$ = %.2f' % (channel_charge[h] / cluster_charge )))
            #axes[h].tick_params(color=cluster_colors[c_idx], labelcolor=cluster_colors[c_idx])
            axes[h].text(0.05, 0.95, textstr, transform=axes[h].transAxes,verticalalignment='top')
            for spine in axes[h].spines.values():
                spine.set_edgecolor(cluster_colors[c_idx])
                spine.set_linewidth(4)
    if (eight_channel):
        for spine in axes[1].spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(4)


    plt.tight_layout()
    fig.suptitle("Event %d"%(idx))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

def onclick(event, split_array, fig, axes, nsamples, eight_channel, gains, peds, min_samp, max_samp):
    global idx
    if event.key == 'right':
        idx += 1
    elif event.key == 'left': idx -= 1
    for ax in axes: 
        for spine in ax.spines.values():
            spine.set_edgecolor('k')
            spine.set_linewidth(1)
        ax.clear()
    event_viewer(split_array, idx, fig, axes, nsamples, eight_channel, gains, peds, min_samp, max_samp); #inform matplotlib of the new data
    plt.draw() #redraw

idx = 0
def main():
    global idx
    args = parser.parse_args()
    fig = plt.figure(args.fig_name)
    gs = mpl.gridspec.GridSpec(2, 13)
    axes = []

    with open(args.event_file, 'rb') as file:
        split_array = pickle.load(file)

    gain =  np.loadtxt(args.gain_file) #np.loadtxt('gain_list_52V.txt')
    peds =  np.loadtxt(args.pedestal_file) #np.loadtxt('pedestal_list_52V.txt')

    for i in range(0, 12):
        if i < 6:
            ax = plt.subplot(gs[0, 2 * i + 1:2 * i + 2 + 1])
            if (i != 0):
                ax = plt.subplot(gs[0, 2 * i + 1:2 * i + 2 + 1])
                ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_xticks(np.arange(12))
            axes.append(ax)
        else:
            ax = plt.subplot(gs[1, 2 * i - 12:2 * i + 2 - 12])
            if (i != 6): ax.set_yticklabels([])
            ax.set_xticks(np.arange(12))
            axes.append(ax)
    event_viewer(split_array, idx, fig, axes, args.nsamples, args.eight_channel, gain, peds, args.min_samp, args.max_samp)
    fig.suptitle("Event %d"%(idx))
    fig.canvas.mpl_connect('key_press_event',lambda event: onclick(event, split_array, fig, axes, args.nsamples, args.eight_channel, gain, peds, args.min_samp, args.max_samp))
    plt.show()
    plt.draw()

if __name__ == '__main__':
    main()