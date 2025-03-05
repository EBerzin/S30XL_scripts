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
import os
from scipy.stats import gaussian_kde
#%matplotlib qt5
colors = ['#0088EE', '#66CC00', '#6600CC', '#CC6600', '#00CCCC', '#CC0000', '#CC00CC', '#FFEE00', '#00CC00', '#0000CC', '#00CC66', '#CC0066', '#A3FF47', '#850AFF', '#85FF0A', '#A347FF']

# Basic Plotting Scripts

def get_gain(split_array, eight_channel = False, get_gain = False, nsamples = 5, new_format = False, by_bunch_count = False, n_peaks = 4, pedestal_list = None, gain_list = None, pdf = None):

    # First let's just plot everything with the proper binning:

    fig, axes = plt.subplots(4,3, figsize = (20,20))
    if (get_gain) : fig2, axes2 = plt.subplots(4,3, figsize = (20,20))
    if (gain_list is None): gain_list = []
    if (pedestal_list is None): pedestal_list = []
    combined_pe_array = []
    for i in range(12):
        lane = SIX_CHANNEL_LIST[i] // 6
        ch = SIX_CHANNEL_LIST[i] % 6
        if (eight_channel) :
            lane = EIGHT_CHANNEL_LIST[i] // 8
            ch = EIGHT_CHANNEL_LIST[i] % 8
        plot_id = i
        #charge_array = adc_to_Q(np.array(df['adc%d'%(ch)][(df['lane'] == lane)]))
        #split_array = group_bunches(df[df['lane'] == lane], np.array(charge_array), by_bunch_count=by_bunch_count, new_format = new_format, ror_length= nsamples)
        charge_array_sum = ak.sum(ak.Array(split_array[i])[ak.num(split_array[i]) >= nsamples,0:nsamples], axis = -1)
        print(charge_array_sum)
        n, bins = np.histogram(charge_array_sum, bins = np.linspace(0,6000,300))
        axes.flatten()[plot_id].errorbar(bins[:-1], n, yerr = np.sqrt(n) ,fmt = 'o-', color = 'k', markersize = 1, linewidth = 0.5)
        axes.flatten()[plot_id].set_xlabel("Q [fC]")
        axes.flatten()[plot_id].set_ylabel("Events")
        axes.flatten()[plot_id].set_title('adc%d, Lane %d'%(ch, lane))
        axes.flatten()[plot_id].set_yscale('log')
        axes.flatten()[plot_id].set_ylim(1, 1e5)

        # If fitting, we can now extract the gain
        if (get_gain) :
            fit_width = 100 #predicted width in fC
            fitting_hist = np.array(n)
            means = []
            sigmas = []
            for j in range(n_peaks):
                mean = bins[:-1][np.argmax(fitting_hist)]
                #mean = bins[:-1][np.where(fitting_hist > 10)] + 30

                fit = (bins[:-1] >= mean - 50) & (bins[:-1] <= mean + 50)
                p0 = [np.max(fitting_hist), mean, 5]
                popt, pcov = curve_fit(gaussian, np.array(bins[:-1])[fit], np.array(fitting_hist)[fit], sigma =  np.sqrt(fitting_hist)[fit], p0 = p0, maxfev = 2000)
                axes.flatten()[plot_id].plot(np.linspace(0,6000,200), gaussian(np.linspace(0,6000,200), *popt) , color = 'r', label = "$\mu = %.3f$ \n$\sigma = %.3f$"%(popt[1], np.abs(popt[2])))
                means.append(popt[1])
                sigmas.append(popt[2])
                axes.flatten()[plot_id].axvline(np.array(bins[:-1])[fit][0])
                axes.flatten()[plot_id].axvline(np.array(bins[:-1])[fit][-1])

                # Zeroing out histogram for the next iteration
                fitting_hist[bins[:-1] <= mean + 7*popt[2]] = 0

            axes2.flatten()[plot_id].errorbar(np.arange(n_peaks), means, yerr = np.abs(sigmas), fmt = 'o', color = 'k')
            p = np.polyfit(np.arange(n_peaks), np.array(means), 1)
            print(i, p[0])
            gain_list.append(p[0])
            pedestal_list.append(p[1]/nsamples)
            axes2.flatten()[plot_id].plot(np.arange(n_peaks), np.arange(n_peaks)* p[0] + p[1], color = 'r', label = "Gain : %.3f \n Pedestal = %.3f"%(p[0], p[1]/nsamples))
            axes2.flatten()[plot_id].set_xlabel("PE")
            axes2.flatten()[plot_id].set_ylabel("Q [fC]")
            axes2.flatten()[plot_id].set_title('adc%d, Lane %d'%(ch, lane))
            axes2.flatten()[plot_id].legend(fontsize = 12)
        if (get_gain or ((len(pedestal_list) > 0) and (len(gain_list) > 0))):
            print(i, len(gain_list), len(pedestal_list))
            combined_pe_array.append((charge_array_sum - (pedestal_list[i]) * nsamples)/gain_list[i])

        axes.flatten()[plot_id].legend(fontsize = 12)
    fig.tight_layout()
    if (get_gain) : fig2.tight_layout()
    
    if (pdf is not None) :
        pdf.savefig(fig, bbox_inches='tight')
        pdf.savefig(fig2, bbox_inches='tight')

    if (get_gain or ((len(pedestal_list) > 0) and (len(gain_list) > 0))):
        fig = plt.figure()
        n, bins = np.histogram(ak.flatten(combined_pe_array), bins = np.linspace(-5,40,200))
        plt.errorbar(bins[:-1], n, yerr = np.sqrt(n) ,fmt = 'o-', color = 'k', markersize = 1, linewidth = 0.5)
        plt.xlabel("PE")
        plt.ylabel("Events")
        plt.ylim(1, 1e5)
        plt.yscale('log')
        if (pdf is not None): pdf.savefig(bbox_inches='tight')



    return gain_list, pedestal_list

def plot_adcs_lanes(df, eight_channel = False, pdf = None):
    fig, axes = plt.subplots(4,3, figsize = (20,20))

    for i in range(12):
        lane = SIX_CHANNEL_LIST[i] // 6
        ch = SIX_CHANNEL_LIST[i] % 6
        if (eight_channel) :
            lane = EIGHT_CHANNEL_LIST[i] // 8
            ch = EIGHT_CHANNEL_LIST[i] % 8
        plot_id = i
        max_ADC = 255
        n, bins = np.histogram(df['adc%d'%(ch)][(df['lane'] == lane)], bins = np.arange(max_ADC))
        x = np.arange(max_ADC-1)
        axes.flatten()[plot_id].errorbar(x, n, yerr = np.sqrt(n) ,fmt = 'o-', color = 'k')
        axes.flatten()[plot_id].set_xlabel("ADC Value")
        axes.flatten()[plot_id].set_ylabel("Events")
        axes.flatten()[plot_id].set_title('adc%d, Lane %d'%(ch, lane))
        axes.flatten()[plot_id].set_yscale('log')
        fit_range = (x >= 3) & (x <=  9)
        axes.flatten()[plot_id].axvline(3)
        axes.flatten()[plot_id].axvline(9)
        try: 
            popt, pcov = curve_fit(gaussian, x[fit_range], n[fit_range], p0 = [500, 5, 0.3], sigma =  np.sqrt(n)[fit_range])
            axes.flatten()[plot_id].plot(np.linspace(0,10,100), gaussian(np.linspace(0,10,100), *popt) , color = 'r', label = "$\mu = %.3f$ \n$\sigma = %.3f$"%(popt[1], np.abs(popt[2])))
        except: print("ADC Fit Failed")
        axes.flatten()[plot_id].legend()
        print(ch, lane, popt[1], np.abs(popt[2]))
        axes.flatten()[plot_id].set_ylim(1, 5e5)

    fig.tight_layout()
    if pdf is not None: pdf.savefig(fig, bbox_inches='tight')

def plot_charge_lanes(split_array, threshold, eight_channel = False, trigger_df = None, nsamples = 5, by_bunch_count = True, new_format = False, ror_length = 12, landau_fit = False, gain_list = None, pedestal_list = None, min_samp = 0, max_samp = None, calc_pedestal = False, pdf = None):
    fig, axes = plt.subplots(4,3, figsize = (20,20))
    fig2, axes2 = plt.subplots(4,3, figsize = (20,20))

    if (max_samp is None): max_samp = nsamples

    total_charge = []
    total_charge_top = []
    total_charge_bottom = []

    PE_array_sum_top = []
    PE_array_sum_bottom = []

    for i in range(12):
        lane = SIX_CHANNEL_LIST[i] // 6
        ch = SIX_CHANNEL_LIST[i] % 6
        if (eight_channel) :
            lane = EIGHT_CHANNEL_LIST[i] // 8
            ch = EIGHT_CHANNEL_LIST[i] % 8
        plot_id = i
        chunk_size = 1
        #charge_array = adc_to_Q(np.array(df['adc%d'%(ch)][(df['lane'] == lane)]))

        # If providing a list of gains, subtract the pedestal
        #if (gain_list and pedestal_list and not calc_pedestal):
        #    charge_array = charge_array - pedestal_list[i]

        #tdc_array = np.array(df['tdc%d'%(ch)][(df['lane'] == lane)])
        #print("Total Entries: ", len(charge_array))
        
        #split_array = group_bunches(df[df['lane'] == lane], np.array(charge_array), by_bunch_count=by_bunch_count, new_format = new_format, ror_length = ror_length)
        #print("Total Events: ", len(split_array))
        
        #select_tdc_array = tdc_array < 63
        charge_array_sum = ak.sum(ak.Array(split_array[i])[ak.num(split_array[i]) >= nsamples,min_samp:max_samp], axis = -1)
        if (calc_pedestal):
            pedestal_calc = ak.mean(ak.Array(split_array[i])[ak.num(split_array[i]) >= nsamples,0:10], axis = -1)
            charge_array_sum = charge_array_sum - (pedestal_calc * (max_samp - min_samp))
        else: charge_array_sum = charge_array_sum - (pedestal_list[i] * (max_samp - min_samp))

        #print(ak.count(ak.Array(split_array)[ak.num(split_array) >= nsamples,35:45], axis = -1))

        #print(charge_array_sum)
        #max_size = len(charge_array) // chunk_size * chunk_size
        #charge_array_sum = charge_array[0:max_size].reshape(max_size//chunk_size, chunk_size) @ np.ones(chunk_size)

        if (gain_list) : 
            PE_array_sum = charge_array_sum / gain_list[i]
            print(PE_array_sum)
            if (i < 6): PE_array_sum_top.append(PE_array_sum)
            else: PE_array_sum_bottom.append(PE_array_sum)

        n, bins = np.histogram(charge_array_sum, bins = np.linspace(0,40000,100))
        total_charge.append(charge_array_sum)
        if (i < 6): total_charge_top.append(charge_array_sum)
        else: total_charge_bottom.append(charge_array_sum)
 
        axes.flatten()[plot_id].errorbar(bins[:-1], n, yerr = np.sqrt(n) ,fmt = 'o-', color = 'k')
        axes.flatten()[plot_id].set_xlabel("Q")
        axes.flatten()[plot_id].set_ylabel("Events")
        axes.flatten()[plot_id].set_title('adc%d, Lane %d'%(ch, lane))
        axes.flatten()[plot_id].set_yscale('log')
        axes.flatten()[plot_id].axvline(threshold*160, color = 'k', linestyle = '--', label = '1x thresh.')
        axes.flatten()[plot_id].axvline(2*threshold*160, color = 'k', linestyle = ':', label = '2x thresh.')
        

        if (gain_list):
            n_PE, bins_PE = np.histogram(PE_array_sum, bins = np.linspace(0,200,100))
            axes2.flatten()[plot_id].errorbar(bins_PE[:-1], n_PE, yerr = np.sqrt(n_PE) ,fmt = 'o-', color = 'k')
            axes2.flatten()[plot_id].set_xlabel("PE")
            axes2.flatten()[plot_id].set_ylabel("Events")
            axes2.flatten()[plot_id].set_title('adc%d, Lane %d'%(ch, lane))
            axes2.flatten()[plot_id].set_yscale('log')
            axes2.flatten()[plot_id].axvline(threshold*160/gain_list[i], color = 'k', linestyle = '--', label = '1x thresh.')
            axes2.flatten()[plot_id].axvline(2*threshold*160/gain_list[i], color = 'k', linestyle = ':', label = '2x thresh.')
        if (landau_fit and gain_list):
            try:
                fit_range = (bins[:-1] > 9000) & (bins[:-1] < 20000)
                popt, pcov = curve_fit(moyal_pdf, bins[:-1][fit_range], n[fit_range], sigma = np.sqrt(n)[fit_range], p0 = [0.8, 10000, 2000])
                axes.flatten()[plot_id].plot(np.linspace(0,80000,100), moyal_pdf(np.linspace(0,80000,100), *popt) , color = 'r', label = "$\mu = %.3f$ PE \n$\sigma = %.3f$ PE"%(popt[1]/gain_list[i], np.abs(popt[2])/gain_list[i]))
            except: 
                print(lane, ch, "Moyal Fit Failed")
        axes.flatten()[plot_id].legend()
        axes.flatten()[plot_id].set_ylim(1, 1e3)

        axes2.flatten()[plot_id].legend()
        axes2.flatten()[plot_id].set_ylim(1, 1e3)
    
    #if (trigger_df is not None):
    #    for i in range(12):
    #        charge = np.array(trigger_df['amplitude%d'%(i)]) / (0.00625)
    #        split_array = group_bunches(df[df['lane'] == i // 6], np.array(charge), by_bunch_count=True)
    #        charge_array_sum = ak.sum(ak.Array(split_array)[ak.num(split_array) > 4,0:6], axis = -1)
    #        n, bins = np.histogram(charge_array_sum, bins = np.linspace(0,80000,100))
    #        print(np.sum(ak.count(split_array, axis = -1) - old_diff[i]))
    #        axes.flatten()[i].errorbar(bins[:-1], n, yerr = np.sqrt(n) ,fmt = 'o-', color = 'r')

    fig.tight_layout()
    fig2.tight_layout()
    if (pdf is not None):
        pdf.savefig(fig, bbox_inches='tight')
        pdf.savefig(fig2, bbox_inches='tight')


    fig = plt.figure()
    total_charge = ak.Array(total_charge)
    n, bins = np.histogram(ak.sum(total_charge, axis = 0), bins = np.linspace(0,80000,100))
    plt.errorbar(bins[:-1], n, yerr = np.sqrt(n) ,fmt = 'o-', color = 'k')
    plt.title("Summed Charge")
    plt.yscale('log')
    if (pdf is not None): pdf.savefig(bbox_inches='tight')

    fig = plt.figure()
    total_charge_top = ak.Array(total_charge_top)
    #n, bins = np.histogram(ak.sum(total_charge_top, axis = 0), bins = np.linspace(0,80000,100))
    n, bins = np.histogram(ak.sum(PE_array_sum_top, axis = 0), bins = np.linspace(20,400,100))
    plt.errorbar(bins[:-1], n, yerr = np.sqrt(n) ,fmt = 'o-', color = 'k')
    plt.title("Summed Charge (Top)")
    plt.xlabel('PE')
    if (pdf is not None): pdf.savefig(bbox_inches='tight')
    #plt.yscale('log')

    fig = plt.figure()
    total_charge_bottom = ak.Array(total_charge_bottom)
    #n, bins = np.histogram(ak.sum(total_charge_bottom, axis = 0 ), bins = np.linspace(0,80000,100))
    n, bins = np.histogram(ak.sum(PE_array_sum_bottom, axis = 0), bins = np.linspace(20,400,100))
    plt.errorbar(bins[:-1], n, yerr = np.sqrt(n) ,fmt = 'o-', color = 'k')
    plt.title("Summed Charge (Bottom)")
    plt.xlabel('PE')
    if (pdf is not None): pdf.savefig(bbox_inches='tight')
    #plt.yscale('log')


# Channel Correlations Plotting

# EDIT THESE TO ALLOW PULSE SELECTION AND SUM 
def channel_correlations(split_array, df, eight_channel = False, use_sum = False, nsamples = 5, by_bunch_count = True, new_format = False, ror_length = 12, gain_list = None, pedestal_list = None, min_samp = 0, max_samp = None, threshold = 0, calc_pedestal = False, pdf = None):
    fig, axes = plt.subplots(12,12, figsize = (25,25), sharex = True, sharey = True, gridspec_kw={'wspace': 0, 'hspace': 0})
    if (max_samp is None): max_samp = nsamples
    for i1 in range(12):
        lane1 = SIX_CHANNEL_LIST[i1] // 6
        ch1 = SIX_CHANNEL_LIST[i1] % 6
        if (eight_channel) :
            lane1 = EIGHT_CHANNEL_LIST[i1] // 8
            ch1 = EIGHT_CHANNEL_LIST[i1] % 8
        for i2 in range(12):
            lane2 = SIX_CHANNEL_LIST[i2] // 6
            ch2 = SIX_CHANNEL_LIST[i2] % 6
            if (eight_channel) :
                lane2 = EIGHT_CHANNEL_LIST[i2] // 8
                ch2 = EIGHT_CHANNEL_LIST[i2] % 8     
            chunk_size = 1
            #charge_array1 = adc_to_Q(np.array(df['adc%d'%(ch1)][(df['lane'] == lane1)]))
            #charge_array2 = adc_to_Q(np.array(df['adc%d'%(ch2)][(df['lane'] == lane2)]))

            if (use_sum):
                #split_array1 = group_bunches(df[df['lane'] == lane1], np.array(charge_array1), by_bunch_count=by_bunch_count, new_format = new_format, ror_length = ror_length)
                charge_array_sum1 = ak.sum(ak.Array(split_array[i1])[ak.num(split_array[i1]) >= nsamples,min_samp:max_samp], axis = -1)
                if (calc_pedestal):
                    pedestal_calc = ak.mean(ak.Array(split_array[i1])[ak.num(split_array[i1]) >= nsamples,0:10], axis = -1)
                    charge_array_sum1 = charge_array_sum1 - pedestal_calc * (max_samp - min_samp)
                else: charge_array_sum1 = charge_array_sum1 - pedestal_list[i1] * (max_samp - min_samp)

                #split_array2 = group_bunches(df[df['lane'] == lane2], np.array(charge_array2), by_bunch_count=by_bunch_count, new_format = new_format, ror_length = ror_length)
                charge_array_sum2 = ak.sum(ak.Array(split_array[i2])[ak.num(split_array[i2]) >= nsamples,min_samp:max_samp], axis = -1)
                if (calc_pedestal):
                    pedestal_calc = ak.mean(ak.Array(split_array[i2])[ak.num(split_array[i2]) >= nsamples,0:10], axis = -1)
                    charge_array_sum2 = charge_array_sum2 - pedestal_calc * (max_samp - min_samp)
                else: charge_array_sum2 = charge_array_sum2 - pedestal_list[i2] * (max_samp - min_samp)
            else:  
                charge_array1 = adc_to_Q(np.array(df['adc%d'%(ch1)][(df['lane'] == lane1)]))
                charge_array2 = adc_to_Q(np.array(df['adc%d'%(ch2)][(df['lane'] == lane2)]))
                max_size1 = len(charge_array1) // chunk_size * chunk_size
                charge_array_sum1 = charge_array1[0:max_size1].reshape(max_size1//chunk_size, chunk_size) @ np.ones(chunk_size)   
                max_size2 = len(charge_array2) // chunk_size * chunk_size
                charge_array_sum2 = charge_array2[0:max_size2].reshape(max_size2//chunk_size, chunk_size) @ np.ones(chunk_size)
            # Calculate the point density     
            axes[i1,i2].hist2d(np.array(charge_array_sum1), np.array(charge_array_sum2), bins = np.linspace(-500, 30000, 100), norm = mpl.colors.LogNorm(), rasterized = True)
            #axes[i1,i2].scatter(charge_array_sum1, charge_array_sum2, s = 1,label = 'adc%d, Lane %d \n adc%d, Lane %d'%(ch1, lane1, ch2, lane2), rasterized=True)
            axes[i1,i2].set_xlim(-500,30000)
            axes[i1,i2].set_ylim(-500,30000)
            #axes[i1,i2].legend(fontsize = 8)
            axes[i1,i2].text(0.95, 0.95, 'adc%d, Lane %d \n adc%d, Lane %d'%(ch1, lane1, ch2, lane2), transform=axes[i1,i2].transAxes,
                                horizontalalignment='right',
                                verticalalignment='top', fontsize = 8)
            if (i1 == 0) : axes[i1,i2].set_title('adc%d, Lane %d'%(ch2, lane2), fontsize = 10)

    fig.tight_layout()
    if (pdf is not None): pdf.savefig(fig, bbox_inches='tight', dpi = 100)

def channel_covariances(split_array, df, eight_channel = False, conv_bar_id = False, use_sum = False, nsamples = 5, by_bunch_count = True, new_format = False, ror_length = 12, gain_list = None, pedestal_list = None, min_samp = 0, max_samp = None, pdf = None):
    #fig, axes = plt.subplots(12,12, figsize = (25,25), sharex = True, sharey = True, gridspec_kw={'wspace': 0, 'hspace': 0})
    corr_matrix = np.zeros((12,12))
    if (max_samp is None): max_samp = nsamples
    for i1 in range(12):
        lane1 = SIX_CHANNEL_LIST[i1] // 6
        ch1 = SIX_CHANNEL_LIST[i1] % 6
        if (eight_channel) :
            lane1 = EIGHT_CHANNEL_LIST[i1] // 8
            ch1 = EIGHT_CHANNEL_LIST[i1] % 8
        for i2 in range(12):
            lane2 = SIX_CHANNEL_LIST[i2] // 6
            ch2 = SIX_CHANNEL_LIST[i2] % 6
            if (eight_channel) :
                lane2 = EIGHT_CHANNEL_LIST[i2] // 8
                ch2 = EIGHT_CHANNEL_LIST[i2] % 8     
            chunk_size = 1
            

            if (use_sum):
                #split_array1 = group_bunches(df[df['lane'] == lane1], np.array(charge_array1), by_bunch_count=by_bunch_count, new_format = new_format, ror_length = ror_length)
                charge_array_sum1 = ak.sum(ak.Array(split_array[i1])[ak.num(split_array[i1]) >= nsamples,min_samp:max_samp], axis = -1)
                charge_array_sum1 = charge_array_sum1 - pedestal_list[i1] * (max_samp - min_samp)

                #split_array2 = group_bunches(df[df['lane'] == lane2], np.array(charge_array2), by_bunch_count=by_bunch_count, new_format = new_format, ror_length = ror_length)
                charge_array_sum2 = ak.sum(ak.Array(split_array[i2])[ak.num(split_array[i2]) >= nsamples,min_samp:max_samp], axis = -1)
                charge_array_sum2 = charge_array_sum2 - pedestal_list[i2] * (max_samp - min_samp)
            else:  
                charge_array1 = adc_to_Q(np.array(df['adc%d'%(ch1)][(df['lane'] == lane1)]))
                charge_array2 = adc_to_Q(np.array(df['adc%d'%(ch2)][(df['lane'] == lane2)]))
                max_size1 = len(charge_array1) // chunk_size * chunk_size
                charge_array_sum1 = charge_array1[0:max_size1].reshape(max_size1//chunk_size, chunk_size) @ np.ones(chunk_size)   
                max_size2 = len(charge_array2) // chunk_size * chunk_size
                charge_array_sum2 = charge_array2[0:max_size2].reshape(max_size2//chunk_size, chunk_size) @ np.ones(chunk_size)

            cov_matrix = np.cov(charge_array_sum1, charge_array_sum2) 
            barID1 = i1
            barID2 = i2
            if (conv_bar_id):
                if (i1 <= 5):
                    barID1 = 10 - 2 * i1
                if (i2 <= 5):
                    barID2 = 10 - 2 * i2
                if (i1 >= 6):
                    barID1 = 12 - (2 * (i1 - 6) + 1)
                if (i2 >= 6):
                    barID2 = 12 - (2 * (i2 - 6) + 1)
            
            corr_matrix[barID1,barID2] = cov_matrix[0][1] / (np.sqrt(cov_matrix[0][0]) * np.sqrt(cov_matrix[1][1]))
            print(i1, barID1)
            print(i2, barID2)
            print(corr_matrix[barID1,barID2])
            #axes[i1,i2].scatter(charge_array_sum1, charge_array_sum2, s = 1,label = 'adc%d, Lane %d \n adc%d, Lane %d'%(ch1, lane1, ch2, lane2))
            #axes[i1,i2].set_xlim(-500,30000)
            #axes[i1,i2].set_ylim(-500,30000)
            #axes[i1,i2].legend(fontsize = 8)
            #if (i1 == 0) : axes[i1,i2].set_title('adc%d, Lane %d'%(ch2, lane2), fontsize = 10)
    plt.matshow(corr_matrix)
    cbar = plt.colorbar()
    cbar.set_label('Correlation', rotation=270,labelpad=20)
    #for i1 in range(12):
    #    for i2 in range(12):


    #fig.tight_layout()
    if pdf is not None: pdf.savefig(bbox_inches='tight')


def hits_per_chan(df, threshold, eight_channel = False):

    hits = np.zeros(12)
    for i in range(12):
        lane = SIX_CHANNEL_LIST[i] // 6
        ch = SIX_CHANNEL_LIST[i] % 6
        if (eight_channel) :
            lane = EIGHT_CHANNEL_LIST[i] // 8
            ch = EIGHT_CHANNEL_LIST[i] % 8
        plot_id = i
        max_ADC = 255
        chunk_size = 1
        charge_array = adc_to_Q(np.array(df['adc%d'%(ch)][(df['lane'] == lane)]))
        PE_array = charge_array * 0.00625
        hits[i] = np.sum(PE_array >= threshold)
    #plt.scatter(np.arange(0,12), hits)
    plt.bar(np.arange(0,12), hits, width = 1, align= 'center', fill = None)
    plt.xticks(np.arange(0,12))
    plt.xlabel("Channel")
    plt.ylabel("Hits > %d PE"%(threshold))


x1 = np.array([17, 15.5, 14, 12.5, 11, 9.5, 8, 6.5, 5, 3.5, 2, 0.5]) - 11
x2 = x1 + 3
z1 = np.array([0, -2, 0, -2, 0, -2, 0, -2, 0, -2, 0,-2])
z2 = z1 + 2
y1 = np.ones(12) * -15
y2 = np.ones(12) * 15

ch_bar_map_6 = {'lane0' : {'adc0' : 2, 'adc1' : 0, 'adc2' : 4, 'adc3' : 6, 'adc4' : 8, 'adc5' : 10},
              'lane1' : {'adc0' : 11, 'adc1' : 9, 'adc2' : 7, 'adc3' : 5, 'adc4' : 3, 'adc5' : 1}}
#top_row = [5, 4, 3, 2, 0, 1]
top_row = [10, 8, 6, 4, 2, 0]
ch_bar_map_8 = {'lane0' : {'adc0' : 0, 'adc1' : 2, 'adc2' : 10, 'adc3' : 4, 'adc4' : 3, 'adc5' : 1, 'adc6' : 5, 'adc7' : 11},
             'lane1' : {'adc0' : -1, 'adc1' : 6, 'adc2' : -1, 'adc3' : 8, 'adc4' : 7, 'adc5' : -1, 'adc6' : 9, 'adc7' : -1}}
#top_row = [2, 11, 9, 3, 1, 0]

#ch_bar_map_6 = [5, 4, 3, 2, 0, 1, 6, 7, 8, 9, 10, 11]
EIGHT_CHANNEL_LIST = [2, 11, 9, 3, 1, 0, 7, 14, 12, 6, 4, 5]

ch_bar_map_6 = [1, 11, 0, 10, 2, 9, 3, 8, 4, 7, 5, 6]

ch_bar_map_8 = [0, 5, 1, 4, 3, 6, 9, 12, 11, 14, 2, 7]


def getpath(nested_dict, value, prepath=()):
    for k, v in nested_dict.items():
        path = prepath + (k,)
        if v == value: # found value
            return path
        elif hasattr(v, 'items'): # v is a dict
            p = getpath(v, value, path) # recursive call
            if p is not None:
                return p

def plot_hits_over_threshold_corr(split_array, threshold, eight_channel = False, nsamples = 5, min_samp = 0, max_samp = None, pdf = None):
    cmap = mpl.cm.get_cmap('viridis')
    if (max_samp is None): max_samp = nsamples
    if (eight_channel) : ch_bar_map = ch_bar_map_8
    else : ch_bar_map = ch_bar_map_6
    #for idx, i in enumerate(top_row):
    for i in range(6):
        #ch_dict = getpath(ch_bar_map, i)
        #lane_tag = int(ch_dict[0][-1])
        #ch_tag = int(ch_dict[1][-1])
        #if ch_bar_map['lane%d'%(lane_tag)]['adc%d'%(ch_tag)] == -1 : continue
        fig, ax = plt.subplots(figsize = (10,6))
        num_hits = np.zeros(12)
        #charge_array_tag = adc_to_Q(np.array(df['adc%d'%(ch_tag)][(df['lane'] == lane_tag)]))
        charge_array_sum_tag = ak.sum(ak.Array(split_array[i])[ak.num(split_array[i]) >= nsamples,min_samp:max_samp], axis = -1)
        PE_array_sum = charge_array_sum_tag * (0.00625)
        tag = PE_array_sum >= threshold
        #tag = np.repeat(tag, 2)
        tag_id = i
        num_hits[tag_id] = np.sum(tag)
        lane = SIX_CHANNEL_LIST[i] // 6
        ch = SIX_CHANNEL_LIST[i] % 6
        if (eight_channel):
            lane = EIGHT_CHANNEL_LIST[i] // 8
            ch = EIGHT_CHANNEL_LIST[i] % 8 
        for j in range(12):
        #for lane in range(2):
        #    for ch in range(len(ch_bar_map['lane%d'%(lane)])):
            #if ch_bar_map['lane%d'%(lane)]['adc%d'%(ch)] == -1 : continue
            #plot_id = ch + len(ch_bar_map['lane%d'%(lane)])*lane
            #charge_array = adc_to_Q(np.array(df[tag]['adc%d'%(ch)][(df[tag]['lane'] == lane)]))
            charge_array_sum = ak.sum(ak.Array(split_array[j])[ak.num(split_array[j]) >= nsamples,min_samp:max_samp], axis = -1)[tag]
            PE_array_sum = charge_array_sum * (0.00625)
            if (j <= 5):
                rect_id = 10 - 2 * j
            if (j >= 6):
                rect_id = 12 - (2 * (j - 6) + 1)
            #rect_id = ch_bar_map['lane%d'%(lane)]['adc%d'%(ch)]
            num_hits[j] = np.sum(PE_array_sum >= threshold)
            rect=plt.Rectangle(xy=(x1[rect_id],z1[rect_id]), width=3, height=2, color=cmap(num_hits[j]/ num_hits[tag_id]))
            ax.add_patch(rect)

        plt.xlim(-15,15)
        plt.ylim(-4,4)
        cbar = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap), ax=ax)
        cbar.set_label('Fraction of hits in channel', rotation=270,labelpad=20)
        ax.set_title("Selecting on adc %d, lane %d"%(ch, lane))
        if (pdf is not None): pdf.savefig(bbox_inches='tight')        

# Pulse shape plotting
def plot_charge_v_samples(split_array, threshold, eight_channel = False, trigger_df = None, nsamples = 5, by_bunch_count = True, new_format = False, ror_length = 12, trig_samp = 0, pdf = None):
    fig, axes = plt.subplots(4,3, figsize = (20,20))
    select_pulses = []
    for i in range(12):
        print(i)

        lane = SIX_CHANNEL_LIST[i] // 6
        ch = SIX_CHANNEL_LIST[i] % 6
        if (eight_channel) :
            lane = EIGHT_CHANNEL_LIST[i] // 8
            ch = EIGHT_CHANNEL_LIST[i] % 8
        plot_id = i
        max_ADC = 255
        chunk_size = 1
        #charge_array = adc_to_Q(np.array(df['adc%d'%(ch)][(df['lane'] == lane)]))
        #tdc_array = np.array(df['tdc%d'%(ch)][(df['lane'] == lane)])
        
        #split_array = group_bunches(df[df['lane'] == lane], np.array(charge_array), by_bunch_count=by_bunch_count, new_format = new_format, ror_length = ror_length)
        #split_array_tdcs = group_bunches(df[df['lane'] == lane], np.array(tdc_array), by_bunch_count=by_bunch_count, new_format = new_format, ror_length = ror_length)
        charge_array_sum = ak.Array(split_array[i])[ak.num(split_array[i]) >= nsamples,0:nsamples]
        #tdc_array_sum = ak.Array(split_array_tdcs)[ak.num(split_array_tdcs) >= nsamples,0:nsamples]
        PE_array =  ak.Array(split_array[i]) * 0.00625
        samples = np.tile(np.arange(nsamples), (len(charge_array_sum),1))
        axes.flatten()[plot_id].hist2d(np.array(ak.flatten(samples)), np.array(ak.flatten(charge_array_sum)), bins = [np.arange(nsamples), np.linspace(0, 10000, 50)],norm=mpl.colors.LogNorm())
        #select_tdc_array = tdc_array < 63
        #above_threshold = PE_array[ak.num(split_array[i]) >= nsamples,0:nsamples][:, trig_samp] >= threshold
        select_pulses.append(charge_array_sum)
        axes.flatten()[plot_id].set_xlabel("Sample")
        axes.flatten()[plot_id].set_ylabel("Charge [fC]")
        axes.flatten()[plot_id].set_title('adc%d, Lane %d'%(ch, lane))
    fig.tight_layout()
    if (pdf is not None) : pdf.savefig(bbox_inches='tight')
                        

    fig, axes = plt.subplots(4,3, figsize = (20,20))
    for i in range(12):
        lane = SIX_CHANNEL_LIST[i] // 6
        ch = SIX_CHANNEL_LIST[i] % 6
        if (eight_channel):
            lane = EIGHT_CHANNEL_LIST[i] // 8
            ch = EIGHT_CHANNEL_LIST[i] % 8
        plot_id = i
        print(len(select_pulses[plot_id]))
        if (len(select_pulses[plot_id]) == 0) : continue
        #axes.flatten()[plot_id].plot(np.arange(nsamples), np.ones(len(np.arange(nsamples))) * 30*200, color = 'grey', alpha = 0.5)
        for i in range(300):
            try:
                axes.flatten()[plot_id].plot(np.arange(nsamples), select_pulses[plot_id][i], color = 'grey', alpha = 0.5)
                axes.flatten()[plot_id].set_xlabel("Sample")
                axes.flatten()[plot_id].set_title('adc%d, Lane %d'%(ch, lane))
                axes.flatten()[plot_id].set_ylabel("Charge [fC]")
            except:
                continue

    fig.tight_layout()
    if (pdf is not None) : pdf.savefig(bbox_inches='tight')


# Clustering

def slice_when(predicate, iterable):
  i, x, size = 0, 0, len(iterable)
  while i < size-1:
    if predicate(iterable[i], iterable[i+1]):
      yield iterable[x:i+1]
      x = i + 1
    i += 1
  yield iterable[x:size]

def get_clusters_optimize(split_array, gain_list, pedestal_list, seed_threshold = 25, cluster_threshold = 6, fiducial = False, eight_channel = True, nsamples = 5, new_format = False, by_bunch_count = False, ror_length = 12, min_samp = 0, max_samp = None):
    channel_charge = []
    if (max_samp is None): max_samp = nsamples
    for i in range(12):
        lane = SIX_CHANNEL_LIST[i] // 6
        ch = SIX_CHANNEL_LIST[i] % 6
        if (eight_channel) :
            lane = EIGHT_CHANNEL_LIST[i] // 8
            ch = EIGHT_CHANNEL_LIST[i] % 8
        plot_id = i
        #charge_array = adc_to_Q(np.array(df['adc%d'%(ch)][(df['lane'] == lane)]))        
        #split_array = group_bunches(df[df['lane'] == lane], np.array(charge_array), by_bunch_count=by_bunch_count, new_format = new_format, ror_length = ror_length)
        charge_array_sum = ak.sum(ak.Array(split_array[i])[ak.num(split_array[i]) >= nsamples,min_samp:max_samp], axis = -1)
        PE_array_sum = (charge_array_sum - pedestal_list[i] * (max_samp - min_samp)) / gain_list[i]
        channel_charge.append(PE_array_sum)

    all_clusters = 0
    cluster_energies = []
    cluster_energies_top = []
    cluster_energies_bottom = []
    n_hits = []
    flag_counts = np.zeros(5)


    channel_charge = np.array(channel_charge)
    mask = channel_charge > cluster_threshold
    row_indices, col_indices = np.nonzero(mask)
    active_channels = ak.from_iter([row_indices[col_indices == col] for col in range(channel_charge.shape[1])])
    active_channels_map = ak.where(active_channels <= 5, 2 * (active_channels) + 1, active_channels)
    active_channels_map = ak.where(active_channels > 5, (active_channels - 6)*2, active_channels_map)

    clusters = [list(slice_when(lambda x,y: y - x > 2, np.sort(active_channels_map[col]))) for col in range(len(active_channels_map))]
    clusters = ak.from_iter(clusters)
    clusters.show()

    gt_4_hits = ak.count(clusters, axis = -1) > 4
    flag_counts[0] += np.sum(gt_4_hits)
    no_hits_in_top = ak.sum(clusters % 2 == 0, axis = -1) == 0
    flag_counts[1] += np.sum(no_hits_in_top)
    no_hits_in_bottom = ak.sum(clusters % 2 == 1, axis = -1) == 0
    flag_counts[2] += np.sum(no_hits_in_bottom)

    cluster_indices = ak.where(clusters % 2 == 0, clusters/2 + 6, clusters)
    cluster_indices = ak.where(clusters % 2 == 1, (clusters - 1) / 2, cluster_indices)

    all_channels = ak.from_iter([np.tile(channel_charge[:,col],(len(clusters[col]),1)) for col in range(len(clusters))])

    cluster_indices = ak.values_astype(cluster_indices, "int64")

    pe_values = all_channels[cluster_indices]

    no_seed = ak.sum(pe_values > seed_threshold, axis = -1) == 0

    pe_values_top = all_channels[cluster_indices[cluster_indices <= 5]]
    pe_values_bottom = all_channels[cluster_indices[cluster_indices > 5]]

    cluster_pe_sum = ak.sum(pe_values, axis = -1)
    cluster_pe_sum_top = ak.sum(pe_values_top, axis = -1)
    cluster_pe_sum_bottom = ak.sum(pe_values_bottom, axis = -1)

    cluster_energies=cluster_pe_sum[~gt_4_hits & ~no_hits_in_top & ~no_hits_in_bottom & ~no_seed]
    cluster_energies_top=cluster_pe_sum_top[~gt_4_hits & ~no_hits_in_top & ~no_hits_in_bottom & ~no_seed]
    cluster_energies_bottom=cluster_pe_sum_bottom[~gt_4_hits & ~no_hits_in_top & ~no_hits_in_bottom & ~no_seed]

    return cluster_energies, cluster_energies_top, cluster_energies_bottom, n_hits, clusters, pe_values


