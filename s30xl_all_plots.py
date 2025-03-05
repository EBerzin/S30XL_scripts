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
from matplotlib.backends.backend_pdf import PdfPages

mpl.rc("figure", dpi=300)
plt.rcParams["font.size"] =   20.0
colors = ['#0088EE', '#66CC00', '#6600CC', '#CC6600', '#00CCCC', '#CC0000', '#CC00CC', '#FFEE00', '#00CC00', '#0000CC', '#00CC66', '#CC0066', '#A3FF47', '#850AFF', '#85FF0A', '#A347FF']

parser = argparse.ArgumentParser(
                    prog='s30xl_all_plots.py',
                    description='Generates all standard plots for S30XL analysis')

parser.add_argument('-f', '--db_file_list', nargs='+', type=str, default="",
                    help="List of database files")
parser.add_argument('-s', '--nsamples', type=int, default=12,
                    help = "Number of samples per event")
parser.add_argument('-ch8', '--eight_channel', action='store_true', default=False,
                    help = "Parse to 8-channel format")
parser.add_argument('-old_fmt', '--new_format', action='store_false', default=True,
                    help = "Interpret database in old format (without ror_timestamp)")
parser.add_argument('-use_bc', '--by_bunch_count', action='store_true', default=False,
                    help = "Use consecutive bunch counts rather than timestamps to filter events")
parser.add_argument('-min_s', '--min_samp', type=int, default=0,
                    help = "Minimum sample considered when summing event charge")
parser.add_argument('-max_s', '--max_samp', type=int, default=None,
                    help = "Maximum sample considered when summing event charge")
parser.add_argument('-pf', '--pedestal_file', type=str, default=None,
                    help = "File used to determine pedestal/gain. Default will be first file given in the db_file_list.")
parser.add_argument('-ps', '--pedestal_nsamp', type=int, default=None,
                    help = "Number of samples in pedestal determination.")
parser.add_argument('-th', '--threshold', type=int, default=30,
                    help = "Treshold used to take data, or threshold for hit selection.")
parser.add_argument('-cl_s_th', '--cluster_seed_threshold', type=int, default=20,
                    help = "Treshold used to seed clusters")
parser.add_argument('-cl_th', '--cluster_threshold', type=int, default=10,
                    help = "Treshold used to add hits to clusters")

def main():
    args = parser.parse_args()
    base_filename = os.path.basename(args.db_file_list[0])
    save_pdf = PdfPages(os.path.splitext(base_filename)[0] + '_plots.pdf')

    if args.pedestal_file is None:
        pedestal_file = args.db_file_list[0]
    else: pedestal_file = args.pedestal_file

    if args.pedestal_nsamp is None:
        pedestal_nsamps = args.nsamples
    else: pedestal_nsamps = args.pedestal_nsamp

    conn = connect_to_db(pedestal_file)
    df_pedestal = create_df(conn)
    if (args.eight_channel) : df_pedestal = six_to_eight_optimized(df_pedestal, args.new_format)
    adc_split_array_pedestal = get_adc_split_array(df_pedestal, eight_channel=args.eight_channel, by_bunch_count=args.by_bunch_count, new_format=args.new_format, ror_length=args.nsamples)
    gain_list, pedestal_list = get_gain(adc_split_array_pedestal, get_gain=True, eight_channel=args.eight_channel, nsamples=pedestal_nsamps, new_format=args.new_format, by_bunch_count=args.by_bunch_count, gain_list = None, pedestal_list = None, n_peaks=3, pdf = save_pdf)

    conns = []
    for file in args.db_file_list:
        conns.append(connect_to_db(file))
    df_full = create_df_combo(conns)
    df_trigger_full = create_trigger_df_combo(conns)
    if (args.eight_channel) : df_full = six_to_eight_optimized(df_full, args.new_format)

    print("Misaligned points: ", check_alignment(df_full))
    adc_split_array = get_adc_split_array(df_full,eight_channel=args.eight_channel, by_bunch_count=args.by_bunch_count, new_format=args.new_format, ror_length=args.nsamples)
    plot_adcs_lanes(df_full, eight_channel = False, pdf = save_pdf)
    plot_charge_v_samples(adc_split_array, threshold = 0, eight_channel=args.eight_channel, new_format=args.new_format, nsamples=args.nsamples, by_bunch_count=args.by_bunch_count, trig_samp = 0, ror_length=args.nsamples, pdf = save_pdf)
    plot_charge_lanes(adc_split_array, threshold = args.threshold, eight_channel=args.eight_channel, nsamples=args.nsamples, new_format = args.new_format, by_bunch_count=args.by_bunch_count, landau_fit= False, gain_list=gain_list, pedestal_list=pedestal_list, ror_length=args.nsamples, calc_pedestal=False, min_samp=args.min_samp, max_samp=args.max_samp, pdf = save_pdf)
    channel_correlations(adc_split_array, df_full, eight_channel = args.eight_channel, use_sum = True, nsamples = args.nsamples, by_bunch_count = args.by_bunch_count, new_format = args.new_format, ror_length = args.nsamples, gain_list = gain_list, pedestal_list = pedestal_list, min_samp = args.min_samp, max_samp = args.max_samp, calc_pedestal=False, pdf = save_pdf)
    channel_covariances(adc_split_array, df_full, eight_channel = args.eight_channel, conv_bar_id=True, use_sum = True, nsamples = args.nsamples, by_bunch_count = args.by_bunch_count, new_format = args.new_format, ror_length = args.nsamples, gain_list = gain_list, pedestal_list = pedestal_list, min_samp = args.min_samp, max_samp = args.max_samp, pdf = save_pdf)
    plot_hits_over_threshold_corr(adc_split_array, args.threshold, eight_channel = args.eight_channel, nsamples=args.nsamples, pdf = save_pdf)
    cluster_energies, cluster_energies_top, cluster_energies_bottom, n_hits, clusters, pe_values = get_clusters_optimize(adc_split_array, gain_list, pedestal_list, seed_threshold = args.cluster_seed_threshold, cluster_threshold = args.cluster_threshold, fiducial = False, eight_channel = args.eight_channel, nsamples = args.nsamples, new_format = args.new_format, by_bunch_count = args.by_bunch_count, ror_length = args.nsamples, min_samp = args.min_samp, max_samp = args.max_samp)

    # Cluster Plotting
    fig = plt.figure()
    n, bins = np.histogram(ak.flatten(cluster_energies), bins = np.linspace(10,250,50))
    plt.errorbar(bins[:-1], n, yerr = np.sqrt(n) ,fmt = 'o-', color = 'k', label = "Both Rows")
    n, bins = np.histogram(ak.flatten(cluster_energies_top), bins = np.linspace(10,250,50))
    plt.errorbar(bins[:-1], n, yerr = np.sqrt(n) ,fmt = 'o-', color = colors[0], label = "Top Row")
    n, bins = np.histogram(ak.flatten(cluster_energies_bottom), bins = np.linspace(10,250,50))
    plt.errorbar(bins[:-1], n, yerr = np.sqrt(n) ,fmt = 'o-', color = colors[1], label = "Bottom Row")
    plt.xlabel('PE')
    plt.title('Cluster Charge')
    plt.ylabel("Counts")
    plt.legend()
    save_pdf.savefig(fig, bbox_inches='tight')



    save_pdf.close()
if __name__ == '__main__':
    main()