import sqlite3
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import moyal
from collections import Counter
import os
#%matplotlib qt5
colors = ['#0088EE', '#66CC00', '#6600CC', '#CC6600', '#00CCCC', '#CC0000', '#CC00CC', '#FFEE00', '#00CC00', '#0000CC', '#00CC66', '#CC0066', '#A3FF47', '#850AFF', '#85FF0A', '#A347FF']

EIGHT_CHANNEL_LIST = [2, 11, 9, 3, 1, 0, 7, 14, 12, 6, 4, 5]
SIX_CHANNEL_LIST = [5, 4, 3, 2, 0, 1, 6, 7, 8, 9, 10, 11]

# Constructing data frames
# -------------------------------------------------------------------------------------

def connect_to_db(db_file):
    """Connect to the SQLite database"""
    conn = sqlite3.connect(db_file)
    return conn

def create_df(conn):
    df = pd.read_sql_query("SELECT * FROM ts_raw_daq_event", conn)
    return df

def create_trigger_df(conn):
    df = pd.read_sql_query("SELECT * FROM ts_s30xl_threshold_trigger_event", conn)
    return df

def create_df_combo(conns):
    dfs = []
    for conn in conns:
        df = pd.read_sql_query("SELECT * FROM ts_raw_daq_event", conn)
        dfs.append(df)
    full_df = pd.concat(dfs)
    return full_df

def create_trigger_df_combo(conns):
    dfs = []
    for conn in conns:
        df = pd.read_sql_query("SELECT * FROM ts_s30xl_threshold_trigger_event", conn)
        dfs.append(df)
    full_df = pd.concat(dfs)
    return full_df


# Parsing / unpacking
# -------------------------------------------------------------------------------------

def to_hex(byte):
    return '%.*x' % (2, int('0b'+byte, 0))

def unpack_frame(df):
    events = []
    for i in range(len(df)):
        bytes = []
        tdcs = []
        for j in range(6):
            tdcs.append(np.binary_repr(np.asarray(df['tdc%d'%(j)])[i], width = 6))
        capID = np.binary_repr(np.asarray(df['capId'])[i], width = 2)
        ce = np.binary_repr(np.asarray(df['ce'])[i], width = 1)
        bc0 = np.binary_repr(np.asarray(df['bc0'])[i], width = 1)
        print(capID)
        print(ce)
        print(bc0)

        byte1 = tdcs[0][-4:] + capID + ce + bc0
        print(byte1)
        bytes.append(to_hex(byte1))

        for j in range(6):
            bytes.append(to_hex(np.binary_repr(np.asarray(df['adc%d'%(j)])[i], width = 8)))

        #print(bytes)
        byte8 = tdcs[1] + tdcs[0][0:2]
        byte9 = tdcs[3][-2:] + tdcs[2]
        byte10 = tdcs[4][-4:] + tdcs[3][0:4]
        byte11 = tdcs[5] + tdcs[4][0:2]

        bytes.append(to_hex(byte8))
        bytes.append(to_hex(byte9))
        bytes.append(to_hex(byte10))
        bytes.append(to_hex(byte11))
        events.append(bytes)

    return events

def six_to_eight(df, new_format = False):
    df_out = pd.DataFrame()#pd.DataFrame(df.columns)
    print(len(df))
    for i in range(len(df)):
        if (i % 1000 == 0): print(i)
        bytes = []
        tdcs = []
        for j in range(6):
            tdcs.append(np.binary_repr(np.asarray(df['tdc%d'%(j)])[i], width = 6))
        capID = np.binary_repr(np.asarray(df['capId'])[i], width = 2)
        ce = np.binary_repr(np.asarray(df['ce'])[i], width = 1)
        bc0 = np.binary_repr(np.asarray(df['bc0'])[i], width = 1)

        byte8 = tdcs[1] + tdcs[0][0:2]
        byte9 = tdcs[3][-2:] + tdcs[2]
        byte10 = tdcs[4][-4:] + tdcs[3][0:4]
        byte11 = tdcs[5] + tdcs[4][0:2]

        if (not new_format):
            entry = pd.DataFrame.from_dict({
            "timestamp": [df["timestamp"].iloc[i]],
            "pulse_id": [df["pulse_id"].iloc[i]],
            "bunch_count": [df["bunch_count"].iloc[i]],
            "lane": [df["lane"].iloc[i]],
            "flags": [df["flags"].iloc[i]],
            "capId": [int(capID, 2)],
            "ce":  [int(ce, 2)],
            "bc0":  [int(bc0, 2)],
            "adc0":  [df["adc0"].iloc[i]],
            "adc1":  [df["adc1"].iloc[i]],
            "adc2":  [df["adc2"].iloc[i]],
            "adc3":  [df["adc3"].iloc[i]],
            "adc4":  [df["adc4"].iloc[i]],
            "adc5":  [df["adc5"].iloc[i]],
            "adc6":  [int(byte8, 2)],
            "adc7":  [int(byte9, 2)],
            "tdc0": [np.nan],
            "tdc1": [np.nan],
            "tdc2":  [np.nan],
            "tdc3": [np.nan],
            "tdc4":  [np.nan],
            "tdc5":  [np.nan],
            "tdc6": [np.nan],
            "tdc7": [np.nan]
            })

        if (new_format) :
            entry = pd.DataFrame.from_dict({
            "ror_timestamp": [df["ror_timestamp"].iloc[i]],
            "ror_pulse_id": [df["ror_pulse_id"].iloc[i]],
            "ror_bunch_count": [df["ror_bunch_count"].iloc[i]],
            "data_timestamp": [df["data_timestamp"].iloc[i]],
            "data_pulse_id": [df["data_pulse_id"].iloc[i]],
            "data_bunch_count": [df["data_bunch_count"].iloc[i]],
            "lane": [df["lane"].iloc[i]],
            "flags": [df["flags"].iloc[i]],
            "capId": [int(capID, 2)],
            "ce":  [int(ce, 2)],
            "bc0":  [int(bc0, 2)],
            "adc0":  [df["adc0"].iloc[i]],
            "adc1":  [df["adc1"].iloc[i]],
            "adc2":  [df["adc2"].iloc[i]],
            "adc3":  [df["adc3"].iloc[i]],
            "adc4":  [df["adc4"].iloc[i]],
            "adc5":  [df["adc5"].iloc[i]],
            "adc6":  [int(byte8, 2)],
            "adc7":  [int(byte9, 2)],
            "tdc0": [np.nan],
            "tdc1": [np.nan],
            "tdc2":  [np.nan],
            "tdc3": [np.nan],
            "tdc4":  [np.nan],
            "tdc5":  [np.nan],
            "tdc6": [np.nan],
            "tdc7": [np.nan]
        })


        df_out = pd.concat([df_out, entry], ignore_index=True)

    return df_out

def slicer_vectorized(a,start,end):
    b = a.view((str,1)).reshape(len(a),-1)[:,start:end]
    return np.fromstring(b.tostring(),dtype=(str,end-start))

def six_to_eight_optimized(df, new_format = False):
    tdcs = []
    for j in range(6):
        tdcs.append(pd.Series(df['tdc%d'%(j)]).apply(lambda x: format(x, '06b')).to_numpy(dtype = 'str'))

    byte8 = np.char.add(tdcs[1], slicer_vectorized(tdcs[0],0,2)) #tdcs[1] + tdcs[0][0:2]
    byte9 = np.char.add(slicer_vectorized(tdcs[3],4,6), tdcs[2]) #tdcs[3][-2:] + tdcs[2]
    
    nan_array = np.empty(len(df)) * np.nan
    print(len(byte8), len(byte9), len(nan_array))
    print(len(pd.Series(byte8).apply(lambda x: int(x, 2))))
    if (not new_format):
            
        data = {'timestamp': df["timestamp"],
                'pulse_id': df["pulse_id"],
                "bunch_count": df["bunch_count"],
                "lane": df["lane"],
                "flags": df["flags"],
                "capId": df["capId"],
                "ce":  df["ce"],
                "bc0":  df["bc0"],
                "adc0": df["adc0"],
                "adc1": df["adc1"],
                "adc2": df["adc2"],
                "adc3": df["adc3"],
                "adc4": df["adc4"],
                "adc5": df["adc5"],
                "adc6": pd.Series(byte8).apply(lambda x: int(x, 2)).to_numpy(dtype = 'int'),
                "adc7": pd.Series(byte9).apply(lambda x: int(x, 2)).to_numpy(dtype = 'int'),
                "tdc0": nan_array,
                "tdc1": nan_array,
                "tdc2":  nan_array,
                "tdc3": nan_array,
                "tdc4":  nan_array,
                "tdc5":  nan_array,
                "tdc6": nan_array,
                "tdc7": nan_array}
    
    if (new_format) :
            
        data = {'ror_timestamp': df["ror_timestamp"],
                'ror_pulse_id': df["ror_pulse_id"],
                "ror_bunch_count": df["ror_bunch_count"],
                'data_timestamp': df["data_timestamp"],
                'data_pulse_id': df["data_pulse_id"],
                "data_bunch_count": df["data_bunch_count"],
                "lane": df["lane"],
                "flags": df["flags"],
                "capId": df["capId"],
                "ce":  df["ce"],
                "bc0":  df["bc0"],
                "adc0": df["adc0"],
                "adc1": df["adc1"],
                "adc2": df["adc2"],
                "adc3": df["adc3"],
                "adc4": df["adc4"],
                "adc5": df["adc5"],
                "adc6": pd.Series(byte8).apply(lambda x: int(x, 2)).to_numpy(dtype = 'int'),
                "adc7": pd.Series(byte9).apply(lambda x: int(x, 2)).to_numpy(dtype = 'int'),
                "tdc0": nan_array,
                "tdc1": nan_array,
                "tdc2":  nan_array,
                "tdc3": nan_array,
                "tdc4":  nan_array,
                "tdc5":  nan_array,
                "tdc6": nan_array,
                "tdc7": nan_array}    
    
    return pd.DataFrame(data)

def parse_to_ldmx_sw(df):
    # Okay but we need to figure out how to discard everything after the first 5 time samples for each trigger.
    subEventCounter=0
    samplesPerEvent=5
    outValues = []
    prevCapID0=-1 #reset these for every new event, they can be anything..?
    prevCapID1=-1
    prevBunchCount = -1
    eventCounter=0
    outfile=open('ldmx_sw_parsed.dat', "wb")

    for lanes in zip(np.arange(len(df))[::2], np.arange(len(df))[1::2]):

        if prevBunchCount < 0 :
            prevBunchCount = df['bunch_count'].iloc[lanes[0]]
        
        bunchCountDiff = df['bunch_count'].iloc[lanes[0]] - prevBunchCount
        prevBunchCount = df['bunch_count'].iloc[lanes[0]]

        print(bunchCountDiff)

        if ((bunchCountDiff > 1) | ((bunchCountDiff <= 0) & (bunchCountDiff > -39))):
            print(lanes, "resetting")
            subEventCounter = 0
            prevCapID0=-1 #reset these for every new event, they can be anything..?
            prevCapID1=-1
            outValues.clear()
        else:
            if (subEventCounter > samplesPerEvent): 
                continue

        adcsOut=[] 
        tdcsOut=[] 
        subEventCounter+=1
        # Converting to eight-channel format
        tdcs0 = []
        tdcs1 = []
        for i in range(6):
            tdcs0.append(np.binary_repr(df['tdc%d'%(i)].iloc[lanes[0]], width = 6))
            tdcs1.append(np.binary_repr(df['tdc%d'%(i)].iloc[lanes[1]], width = 6))
        byte8_lane0 = tdcs0[1] + tdcs0[0][0:2]
        byte9_lane0 = tdcs0[3][-2:] + tdcs0[2]
        byte8_lane1 = tdcs1[1] + tdcs1[0][0:2]
        byte9_lane1 = tdcs1[3][-2:] + tdcs1[2]

        for i in range(6):
            adcsOut.append(df['adc%d'%(i)].iloc[lanes[0]])
            tdcsOut.append(63)
        adcsOut.append(int(byte8_lane0, 2))
        adcsOut.append(int(byte9_lane0, 2))
        tdcsOut.append(63)
        tdcsOut.append(63)
        for i in range(6):
            adcsOut.append(df['adc%d'%(i)].iloc[lanes[1]])
            tdcsOut.append(63)
        adcsOut.append(int(byte8_lane1, 2))
        adcsOut.append(int(byte9_lane1, 2))
        tdcsOut.append(63)
        tdcsOut.append(63)

        CIDunsync=(df['capId'].iloc[lanes[0]] != df['capId'].iloc[lanes[1]])
        if CIDunsync :  #useful debugging/header cross check
            print("CID between fiber1 and 2 unsynced!")
        CRC0error=(df['ce'].iloc[lanes[0]] != 0 )
        if CRC0error :
            print(lanes, "fiber 1 has CE error flag!")            
        CRC1error=(df['ce'].iloc[lanes[1]] != 0 )
        if CRC1error :
            print(lanes, "fiber 2 has CE error flag!")
        CIDskip=False
        if (prevCapID0 < 0) :
            prevCapID0 = df['capId'].iloc[lanes[0]]
        else :    #this logic still needs some work: what happens if there is a corrupt time sample word in the middle?
            if (prevCapID0+1)%4  != (df['capId'].iloc[lanes[0]])%4 :
                CIDskip=True
                print("Found CIDskip in fiber 1! previous CapID: "+str(prevCapID0)+", current: "+str(df['capId'].iloc[lanes[0]]))
        prevCapID0=df['capId'].iloc[lanes[0]]#update after checking
        if (prevCapID1 < 0) :
            prevCapID1 = df['capId'].iloc[lanes[1]]
        else :
            if (prevCapID1+1)%4 != (df['capId'].iloc[lanes[1]])%4 :
                CIDskip=True
                print("Found CIDskip in fiber 2! previous CapID: "+str(prevCapID1)+", current: "+str(df['capId'].iloc[lanes[1]]))
        prevCapID1=df['capId'].iloc[lanes[1]] #update

        outValues.append(adcsOut) 
        outValues.append(tdcsOut)
        if (subEventCounter == samplesPerEvent):
            print("Writing Event", lanes, subEventCounter)
            eventCounter+=1
            endian="little"
            outfile.write( int(0).to_bytes(4, byteorder=endian, signed=False)) 
            outfile.write( int(0).to_bytes(4, byteorder=endian, signed=False)) #placeholder number for now, clock ticks?
            outfile.write( int(0).to_bytes(4, byteorder=endian, signed=False)) 
            outfile.write( int(eventCounter).to_bytes(3, byteorder=endian, signed=False)) 
            errorWord=0
            errorWord |= (CRC0error << 0)
            errorWord |= (CRC1error << 1)
            errorWord |= (CIDunsync << 2)
            errorWord |= (CIDskip   << 3)
            outfile.write( int(errorWord).to_bytes(1, byteorder=endian, signed=False)) 
            print(len(outValues))
            if len(outValues) != samplesPerEvent*2 : #1 ADC, 1 TDC vector per time sample 
                print("UH-OH! got "+str(len(outValues))+" words in the event data!")
                print(outValues)
            for vals in outValues :
                if len(vals) != 16 : #expect one word per channel 
                    print("UH-OH! got "+str(len(vals))+" words in the event data!")
                    print(vals)
                for word in vals :  #ADC and TDC both 8-bit words in final format
                    outfile.write( int(word).to_bytes(1, byteorder=endian, signed=False)) 
            outValues.clear()
        # Let's think about how to do this sample parsing. What we can do is track the bunchCount in the same way. 
        # So if the bunchcount increment is 1 or -39, we can continue. But if the number of samples is greater than
        # 5, we don't add it
    outfile.close()
    
def parse_to_ldmx_sw_optimized(df, samplesPerEvent, eight_channel, out_name):
    # Okay maybe it's fine to loop over the array instead of doing it vectorized. Let's see.

    df_splits = group_bunches_splits(df[df['lane'] == 0], True, False)
    adcs_lane_0_split = []
    adcs_lane_1_split = []
    tdcs_lane_0_split = []
    tdcs_lane_1_split = []
    for i in range(6):
        adcs_lane_0_split.append(np.split(np.array(df['adc%d'%(i)][df['lane'] == 0]), df_splits))
        adcs_lane_1_split.append(np.split(np.array(df['adc%d'%(i)][df['lane'] == 1]), df_splits))
        tdcs_lane_0_split.append(np.split(np.array(df['tdc%d'%(i)][df['lane'] == 0]), df_splits))
        tdcs_lane_1_split.append(np.split(np.array(df['tdc%d'%(i)][df['lane'] == 1]), df_splits))
    capID_split_lane0 = np.split(np.array(df['capId'][df['lane'] == 0]), df_splits)
    capID_split_lane1 = np.split(np.array(df['capId'][df['lane'] == 1]), df_splits)
    ce_split_lane0 = np.split(np.array(df['ce'][df['lane'] == 0]), df_splits)
    ce_split_lane1 = np.split(np.array(df['ce'][df['lane'] == 1]), df_splits)

    outValues = []
    eventCounter = 0
    outfile=open(out_name, "wb")

    for ev in range(len(capID_split_lane0)): # Looping over all events
        prevCapID0=-1 #reset these for every new event, they can be anything..?
        prevCapID1=-1
        if (len(capID_split_lane0[ev])) < samplesPerEvent : continue
        for s in range(samplesPerEvent):
            adcsOut=[] 
            tdcsOut=[]

            tdcs0 = []
            tdcs1 = []
            for i in range(6):
                #print(tdcs_lane_0_split[i])
                tdcs0.append(np.binary_repr(tdcs_lane_0_split[i][ev][s], width = 6))
                tdcs1.append(np.binary_repr(tdcs_lane_1_split[i][ev][s], width = 6))

            if (eight_channel):
                byte8_lane0 = tdcs0[1] + tdcs0[0][0:2]
                byte9_lane0 = tdcs0[3][-2:] + tdcs0[2]
                byte8_lane1 = tdcs1[1] + tdcs1[0][0:2]
                byte9_lane1 = tdcs1[3][-2:] + tdcs1[2]

            for i in range(6):
                adcsOut.append(adcs_lane_0_split[i][ev][s])
                tdcsOut.append(tdcs_lane_0_split[i][ev][s])
            if (eight_channel):
                adcsOut.append(int(byte8_lane0, 2))
                adcsOut.append(int(byte9_lane0, 2))
                tdcsOut.append(63)
                tdcsOut.append(63)
            for i in range(6):
                adcsOut.append(adcs_lane_1_split[i][ev][s])
                tdcsOut.append(tdcs_lane_1_split[i][ev][s])
            if (eight_channel):
                adcsOut.append(int(byte8_lane1, 2))
                adcsOut.append(int(byte9_lane1, 2))
                tdcsOut.append(63)
                tdcsOut.append(63) 

            CIDunsync=(capID_split_lane0[ev][s] != capID_split_lane1[ev][s])
            if CIDunsync :  #useful debugging/header cross check
                print("CID between fiber1 and 2 unsynced!")
            CRC0error=(ce_split_lane0[ev][s] != 0 )
            if CRC0error :
                print(ev, s, "fiber 1 has CE error flag!")            
            CRC1error=(ce_split_lane1[ev][s] != 0 )
            if CRC1error :
                print(ev, s, "fiber 2 has CE error flag!")
            CIDskip=False
            if (prevCapID0 < 0) :
                prevCapID0 = capID_split_lane0[ev][s] 
            else :    #this logic still needs some work: what happens if there is a corrupt time sample word in the middle?
                if (prevCapID0+1)%4  != (capID_split_lane0[ev][s])%4 :
                    CIDskip=True
                    print("Found CIDskip in fiber 1! previous CapID: "+str(prevCapID0)+", current: "+str(capID_split_lane0[ev][s]))
            prevCapID0=capID_split_lane0[ev][s] #update after checking
            if (prevCapID1 < 0) :
                prevCapID1 = capID_split_lane1[ev][s]
            else :
                if (prevCapID1+1)%4 != (capID_split_lane1[ev][s])%4 :
                    CIDskip=True
                    print("Found CIDskip in fiber 2! previous CapID: "+str(prevCapID1)+", current: "+str(capID_split_lane1[ev][s]))
            prevCapID1=capID_split_lane1[ev][s] #update
            outValues.append(adcsOut) 
            outValues.append(tdcsOut)
        
        print("Writing Event", ev)
        eventCounter+=1
        endian="little"
        outfile.write( int(0).to_bytes(4, byteorder=endian, signed=False)) 
        outfile.write( int(0).to_bytes(4, byteorder=endian, signed=False)) #placeholder number for now, clock ticks?
        outfile.write( int(0).to_bytes(4, byteorder=endian, signed=False)) 
        outfile.write( int(eventCounter).to_bytes(3, byteorder=endian, signed=False)) 
        errorWord=0
        errorWord |= (CRC0error << 0)
        errorWord |= (CRC1error << 1)
        errorWord |= (CIDunsync << 2)
        errorWord |= (CIDskip   << 3)
        outfile.write( int(errorWord).to_bytes(1, byteorder=endian, signed=False)) 
        print(len(outValues))
        print(errorWord)
        print(outValues)
        if len(outValues) != samplesPerEvent*2 : #1 ADC, 1 TDC vector per time sample 
            print("UH-OH! got "+str(len(outValues))+" words in the event data!")
            print(outValues)
        for vals in outValues :
            if len(vals) != 16 : #expect one word per channel 
                print("UH-OH! got "+str(len(vals))+" words in the event data!")
                print(vals)
            for word in vals :  #ADC and TDC both 8-bit words in final format
                outfile.write( int(word).to_bytes(1, byteorder=endian, signed=False)) 
        outValues.clear()
    outfile.close()

def parse_to_ldmx_sw_sort_triggers(df, df_trigger, samplesPerEvent, eight_channel, out_names):
    # Okay maybe it's fine to loop over the array instead of doing it vectorized. Let's see.

    df_splits, threshold, cosmic, both = group_bunches_sort_triggers_splits(df[df['lane'] == 0], df_trigger, True, samplesPerEvent)
    trigger_indices = [threshold, cosmic, both]
    for i_trigger, trigger in enumerate(trigger_indices):
        adcs_lane_0_split = []
        adcs_lane_1_split = []
        tdcs_lane_0_split = []
        tdcs_lane_1_split = []
        print(trigger)
        for i in range(6):
            print(i)
            adcs_lane_0_split.append(np.array(np.split(np.array(df['adc%d'%(i)][df['lane'] == 0]), df_splits))[trigger])
            adcs_lane_1_split.append(np.array(np.split(np.array(df['adc%d'%(i)][df['lane'] == 1]), df_splits))[trigger])
            tdcs_lane_0_split.append(np.array(np.split(np.array(df['tdc%d'%(i)][df['lane'] == 0]), df_splits))[trigger])
            tdcs_lane_1_split.append(np.array(np.split(np.array(df['tdc%d'%(i)][df['lane'] == 1]), df_splits))[trigger])
        capID_split_lane0 = np.array(np.split(np.array(df['capId'][df['lane'] == 0]), df_splits))[trigger]
        capID_split_lane1 = np.array(np.split(np.array(df['capId'][df['lane'] == 1]), df_splits))[trigger]
        ce_split_lane0 = np.array(np.split(np.array(df['ce'][df['lane'] == 0]), df_splits))[trigger]
        ce_split_lane1 = np.array(np.split(np.array(df['ce'][df['lane'] == 1]), df_splits))[trigger]

        outValues = []
        eventCounter = 0
        outfile=open(out_names[i_trigger], "wb")

        for ev in range(len(capID_split_lane0)): # Looping over all events
            prevCapID0=-1 #reset these for every new event, they can be anything..?
            prevCapID1=-1
            if (len(capID_split_lane0[ev])) < samplesPerEvent : continue
            for s in range(samplesPerEvent):
                adcsOut=[] 
                tdcsOut=[]

                tdcs0 = []
                tdcs1 = []
                for i in range(6):
                    #print(tdcs_lane_0_split[i])
                    tdcs0.append(np.binary_repr(tdcs_lane_0_split[i][ev][s], width = 6))
                    tdcs1.append(np.binary_repr(tdcs_lane_1_split[i][ev][s], width = 6))

                if (eight_channel):
                    byte8_lane0 = tdcs0[1] + tdcs0[0][0:2]
                    byte9_lane0 = tdcs0[3][-2:] + tdcs0[2]
                    byte8_lane1 = tdcs1[1] + tdcs1[0][0:2]
                    byte9_lane1 = tdcs1[3][-2:] + tdcs1[2]

                for i in range(6):
                    adcsOut.append(adcs_lane_0_split[i][ev][s])
                    tdcsOut.append(tdcs_lane_0_split[i][ev][s])
                if (eight_channel):
                    adcsOut.append(int(byte8_lane0, 2))
                    adcsOut.append(int(byte9_lane0, 2))
                    tdcsOut.append(63)
                    tdcsOut.append(63)
                for i in range(6):
                    adcsOut.append(adcs_lane_1_split[i][ev][s])
                    tdcsOut.append(tdcs_lane_1_split[i][ev][s])
                if (eight_channel):
                    adcsOut.append(int(byte8_lane1, 2))
                    adcsOut.append(int(byte9_lane1, 2))
                    tdcsOut.append(63)
                    tdcsOut.append(63) 

                CIDunsync=(capID_split_lane0[ev][s] != capID_split_lane1[ev][s])
                if CIDunsync :  #useful debugging/header cross check
                    print("CID between fiber1 and 2 unsynced!")
                CRC0error=(ce_split_lane0[ev][s] != 0 )
                if CRC0error :
                    print(ev, s, "fiber 1 has CE error flag!")            
                CRC1error=(ce_split_lane1[ev][s] != 0 )
                if CRC1error :
                    print(ev, s, "fiber 2 has CE error flag!")
                CIDskip=False
                if (prevCapID0 < 0) :
                    prevCapID0 = capID_split_lane0[ev][s] 
                else :    #this logic still needs some work: what happens if there is a corrupt time sample word in the middle?
                    if (prevCapID0+1)%4  != (capID_split_lane0[ev][s])%4 :
                        CIDskip=True
                        print("Found CIDskip in fiber 1! previous CapID: "+str(prevCapID0)+", current: "+str(capID_split_lane0[ev][s]))
                prevCapID0=capID_split_lane0[ev][s] #update after checking
                if (prevCapID1 < 0) :
                    prevCapID1 = capID_split_lane1[ev][s]
                else :
                    if (prevCapID1+1)%4 != (capID_split_lane1[ev][s])%4 :
                        CIDskip=True
                        print("Found CIDskip in fiber 2! previous CapID: "+str(prevCapID1)+", current: "+str(capID_split_lane1[ev][s]))
                prevCapID1=capID_split_lane1[ev][s] #update
                outValues.append(adcsOut) 
                outValues.append(tdcsOut)
        
            if (ev % 1000 == 0): print("Writing Event", ev)
            eventCounter+=1
            endian="little"
            outfile.write( int(0).to_bytes(4, byteorder=endian, signed=False)) 
            outfile.write( int(0).to_bytes(4, byteorder=endian, signed=False)) #placeholder number for now, clock ticks?
            outfile.write( int(0).to_bytes(4, byteorder=endian, signed=False)) 
            outfile.write( int(eventCounter).to_bytes(3, byteorder=endian, signed=False)) 
            errorWord=0
            errorWord |= (CRC0error << 0)
            errorWord |= (CRC1error << 1)
            errorWord |= (CIDunsync << 2)
            errorWord |= (CIDskip   << 3)
            outfile.write( int(errorWord).to_bytes(1, byteorder=endian, signed=False)) 
            #print(len(outValues))
            #print(errorWord)
            #print(outValues)
            if len(outValues) != samplesPerEvent*2 : #1 ADC, 1 TDC vector per time sample 
                print("UH-OH! got "+str(len(outValues))+" words in the event data!")
                print(outValues)
            for vals in outValues :
                if len(vals) != 12 : #expect one word per channel 
                    print("UH-OH! got "+str(len(vals))+" words in the event data!")
                    print(vals)
                for word in vals :  #ADC and TDC both 8-bit words in final format
                    outfile.write( int(word).to_bytes(1, byteorder=endian, signed=False)) 
            outValues.clear()
        outfile.close()

def sub_fc_timestamp(lhs: int, rhs: int) -> int:
    """
    Returns number of samples between two timestamps, lhs > rhs
    """

    # 1. Extract fields
    bunch_count_lhs = lhs & 0xFF            # lower 8 bits
    pulse_id_lhs    = (lhs >> 8) & 0xFFFFFFFFFFFFFF  # upper 56 bits

    bunch_count_rhs = rhs & 0xFF            # lower 8 bits
    pulse_id_rhs   = (rhs >> 8) & 0xFFFFFFFFFFFFFF  # upper 56 bits

    
    #print((pulse_id_lhs - pulse_id_rhs),(bunch_count_lhs - bunch_count_rhs))
    #print((pulse_id_lhs - pulse_id_rhs)*40 + (bunch_count_lhs - bunch_count_rhs))


    time_diff = (pulse_id_lhs - pulse_id_rhs) * 40 * 27 + (bunch_count_lhs - bunch_count_rhs) * 27
    return (pulse_id_lhs - pulse_id_rhs) * 40 + (bunch_count_lhs - bunch_count_rhs)

def check_alignment(df):
    lane0ID = df['capId'][(df['lane'] == 0)]
    lane1ID = df['capId'][(df['lane'] == 1)]

    #print(lane0ID)

    return np.sum(np.array(lane0ID) != np.array(lane1ID)) / len(lane0ID)

# Fitting functions
# -------------------------------------------------------------------------------------

def gaussian(x, amp, cen, sigma):
    return amp * np.exp(-(x - cen)**2 / (2 * sigma**2))

def get_mpv(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]

def moyal_pdf(x, c, loc, scale):
    return c * 1e6 * np.array(moyal.pdf(x, loc=loc, scale=scale))


# Linearization / sample grouping
# -------------------------------------------------------------------------------------

NBINS = np.array([0, 16, 36, 57, 64])
EDGES = np.array([0,   34,    158,    419,    517,   915,
        1910,  3990,  4780,   7960,   15900, 32600,
        38900, 64300, 128000, 261000, 350000])
SENSE = np.array([3.1,   6.2,   12.4,  24.8, 24.8, 49.6, 99.2, 198.4,
        198.4, 396.8, 793.6, 1587, 1587, 3174, 6349, 12700])

# This lists channels in physical order as
#  2  11  9  3  1  0
# 7 14  12  6  5  4

# So in the rest of all of the code we index in order of 
#  0  1  2  3  4  5
# 6  7  8  9  10  11

def adc_to_Q(adcs):
    #if (ADC <= 0) return -16;
    #if (ADC >= 255) return 350000;

    rr = adcs // 64
    v1 = adcs % 64
    ss = np.zeros(len(adcs), dtype = np.int32)

    for i in range(1,4):
        ss[v1 > NBINS[i]] += 1

    temp = EDGES[(4 * rr + ss).astype(np.int32)] + (v1 - NBINS[ss]) * SENSE[(4 * rr + ss).astype(np.int32)] + SENSE[(4 * rr + ss).astype(np.int32)] / 2
    temp = np.where(adcs <= 0, -16, temp)
    temp = np.where(adcs >= 255, 350000, temp)
    return temp

def group_bunches(df, arr_to_sum, by_bunch_count = True, new_format = False, ror_length = 12, filter_alignment = False):

    bunch_count_string = 'bunch_count'
    if (new_format) :
        bunch_count_string = 'data_bunch_count'
    if (by_bunch_count):
        diffs = np.diff(df[bunch_count_string])
        splits = np.where((diffs > 1) | ((diffs <= 0) & (diffs > -39)))[0] + 1
    elif (new_format) :
        diffs = np.diff(df['ror_timestamp'])
        iszero = np.concatenate(([0], np.equal(diffs, 0).view(np.int8), [0]))
        absdiff = np.abs(np.diff(iszero))
        # Runs start and end where absdiff is 1.
        ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
        count_num_repeat = ranges[:,1] - ranges[:,0]
        splits = ranges[:,0][count_num_repeat == ror_length - 1]
    # Split the array based on these indices
    adcs_split = np.split(arr_to_sum, splits)
    # Return each chunk
    return adcs_split

def group_bunches_splits(df, by_bunch_count = True, new_format = False, ror_length = 12):

    bunch_count_string = 'bunch_count'
    if (new_format) :
        bunch_count_string = 'data_bunch_count'
    if (by_bunch_count):
        diffs = np.diff(df[bunch_count_string])
        splits = np.where((diffs > 1) | ((diffs <= 0) & (diffs > -39)))[0] + 1
    elif (new_format) :
        diffs = np.diff(df['ror_timestamp'])
        iszero = np.concatenate(([0], np.equal(diffs, 0).view(np.int8), [0]))
        absdiff = np.abs(np.diff(iszero))
        # Runs start and end where absdiff is 1.
        ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
        count_num_repeat = ranges[:,1] - ranges[:,0]
        splits = ranges[:,0][count_num_repeat == ror_length - 1]
    # Split the array based on these indices
    # adcs_split = np.split(arr_to_sum, splits)
    # Return each chunk
    return splits

def group_bunches_sort_triggers(df, df_trigger, arr_to_sum, new_format = False, ror_length = 12):
    bunch_count_string = 'bunch_count'

    if (new_format) :
        bunch_count_string = 'data_bunch_count'
    diffs = np.diff(df['ror_timestamp'])
    iszero = np.concatenate(([0], np.equal(diffs, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    count_num_repeat = ranges[:,1] - ranges[:,0]
    splits = ranges[:,0][count_num_repeat == ror_length - 1]

    # Split the array based on these indices
    adcs_split = np.split(arr_to_sum, splits)
    timestamps_split = np.split(df['ror_timestamp'], splits)
    single_timestamp = ak.fill_none(ak.firsts(ak.Array(timestamps_split), axis = -1), 0)

    # Now look at the trigger table to tag each event as having 0 hits or one hit
    cosmic_triggers = df_trigger['hits'] == 0
    threshold_triggers = df_trigger['hits'] > 0
    vfunc = np.vectorize(sub_fc_timestamp)
    custom_diff_arr = vfunc(df_trigger['timestamp'][1:], df_trigger['timestamp'][:-1])
    both_trigger = (np.array(cosmic_triggers[:-1])) & (np.array(threshold_triggers[1:])) & (np.array(custom_diff_arr) <= 10)
    
    # Okay so both_trigger gives us the index of the last trigger in the consecutive chain. We need
    # the index of the first trigger. We could just loop through and do this...Although that's really not
    # ideal. But we have very few cases where we trigger both. 
    # Let's do this for now and then change it later.
    both_trigger = np.append(both_trigger, False)
    actual_both_trigger_idx = []
    for i_bt in np.where(both_trigger)[0]:
        temp_i = i_bt
        while(custom_diff_arr[temp_i] <= 10):
            temp_i -= 1
        actual_both_trigger_idx.append(temp_i + 1)
    actual_both_trigger = np.zeros(len(both_trigger), dtype=bool)
    actual_both_trigger[actual_both_trigger_idx] = True
            
    indices = np.searchsorted(np.array(single_timestamp), np.array(df_trigger['timestamp']), side='left')
    indices[indices >= len(single_timestamp)] = -1  # Replace out-of-bounds indices with -1

    valid_indices = np.where(np.array((indices != -1) & (single_timestamp[indices] == np.array(df_trigger['timestamp']))), indices, -1)

    adcs_split_both_trigger = ak.Array(adcs_split)[np.array(valid_indices[actual_both_trigger], dtype = int)]
    adcs_split_cosmic_trigger = ak.Array(adcs_split)[np.array(valid_indices[cosmic_triggers & ~actual_both_trigger & (valid_indices != -1)], dtype = int)]
    adcs_split_threshold_trigger = ak.Array(adcs_split)[np.array(valid_indices[threshold_triggers & ~actual_both_trigger & (valid_indices != -1)], dtype = int)]

    # Return each chunk
    return adcs_split_threshold_trigger, adcs_split_cosmic_trigger, adcs_split_both_trigger

def group_bunches_sort_triggers_splits(df, df_trigger, new_format = False, ror_length = 12):
    bunch_count_string = 'bunch_count'

    if (new_format) :
        bunch_count_string = 'data_bunch_count'
    diffs = np.diff(df['ror_timestamp'])
    iszero = np.concatenate(([0], np.equal(diffs, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    count_num_repeat = ranges[:,1] - ranges[:,0]
    splits = ranges[:,0][count_num_repeat == ror_length - 1]

    # Split the array based on these indices
    #adcs_split = np.split(arr_to_sum, splits)
    timestamps_split = np.split(df['ror_timestamp'], splits)
    single_timestamp = ak.fill_none(ak.firsts(ak.Array(timestamps_split), axis = -1), 0)

    # Now look at the trigger table to tag each event as having 0 hits or one hit
    cosmic_triggers = df_trigger['hits'] == 0
    threshold_triggers = df_trigger['hits'] > 0
    vfunc = np.vectorize(sub_fc_timestamp)
    custom_diff_arr = vfunc(df_trigger['timestamp'][1:], df_trigger['timestamp'][:-1])
    both_trigger = (np.array(cosmic_triggers[:-1])) & (np.array(threshold_triggers[1:])) & (np.array(custom_diff_arr) <= 10)
    
    # Okay so both_trigger gives us the index of the last trigger in the consecutive chain. We need
    # the index of the first trigger. We could just loop through and do this...Although that's really not
    # ideal. But we have very few cases where we trigger both. 
    # Let's do this for now and then change it later.
    both_trigger = np.append(both_trigger, False)
    actual_both_trigger_idx = []
    for i_bt in np.where(both_trigger)[0]:
        temp_i = i_bt
        while(custom_diff_arr[temp_i] <= 10):
            temp_i -= 1
        actual_both_trigger_idx.append(temp_i + 1)
    actual_both_trigger = np.zeros(len(both_trigger), dtype=bool)
    actual_both_trigger[actual_both_trigger_idx] = True
            
    indices = np.searchsorted(np.array(single_timestamp), np.array(df_trigger['timestamp']), side='left')
    indices[indices >= len(single_timestamp)] = -1  # Replace out-of-bounds indices with -1

    valid_indices = np.where(np.array((indices != -1) & (single_timestamp[indices] == np.array(df_trigger['timestamp']))), indices, -1)

    #adcs_split_both_trigger = ak.Array(adcs_split)[np.array(valid_indices[actual_both_trigger], dtype = int)]
    #adcs_split_cosmic_trigger = ak.Array(adcs_split)[np.array(valid_indices[cosmic_triggers & ~actual_both_trigger & (valid_indices != -1)], dtype = int)]
    #adcs_split_threshold_trigger = ak.Array(adcs_split)[np.array(valid_indices[threshold_triggers & ~actual_both_trigger & (valid_indices != -1)], dtype = int)]

    threshold_trigger_indices = np.array(valid_indices[threshold_triggers & ~actual_both_trigger & (valid_indices != -1)], dtype = int)
    cosmic_trigger_indices = np.array(valid_indices[cosmic_triggers & ~actual_both_trigger & (valid_indices != -1)], dtype = int)
    both_trigger_indices = np.array(valid_indices[actual_both_trigger], dtype = int)

    # Return each chunk
    return splits, threshold_trigger_indices, cosmic_trigger_indices, both_trigger_indices


def get_adc_split_array(df, eight_channel, by_bunch_count = True, new_format = False, ror_length = 12):

    # Returns an array with nchannel entries in left to right order, 
    # each with an array of ror_length arrays, for each event
    all_adc_split_arrays = []
    for i in range(12):
        lane = SIX_CHANNEL_LIST[i] // 6
        ch = SIX_CHANNEL_LIST[i] % 6
        if (eight_channel) :
            lane = EIGHT_CHANNEL_LIST[i] // 8
            ch = EIGHT_CHANNEL_LIST[i] % 8
        plot_id = i
        chunk_size = 1
        charge_array = adc_to_Q(np.array(df['adc%d'%(ch)][(df['lane'] == lane)]))
        split_array = group_bunches(df[df['lane'] == lane], np.array(charge_array), by_bunch_count=by_bunch_count, new_format = new_format, ror_length = ror_length)
        all_adc_split_arrays.append(split_array)
    return ak.Array(all_adc_split_arrays)


def get_adc_split_array_sort_triggers(df, df_trigger, eight_channel, new_format = False, ror_length = 12):
    all_adc_split_arrays_cosmic = []
    all_adc_split_arrays_threshold = []
    all_adc_split_arrays_both = []


    for i in range(12):
        lane = SIX_CHANNEL_LIST[i] // 6
        ch = SIX_CHANNEL_LIST[i] % 6
        if (eight_channel) :
            lane = EIGHT_CHANNEL_LIST[i] // 8
            ch = EIGHT_CHANNEL_LIST[i] % 8
        plot_id = i
        chunk_size = 1
        charge_array = adc_to_Q(np.array(df['adc%d'%(ch)][(df['lane'] == lane)]))
        split_array = group_bunches_sort_triggers(df[df['lane'] == lane], df_trigger, np.array(charge_array), new_format = new_format, ror_length = ror_length)
        all_adc_split_arrays_threshold.append(split_array[0])
        all_adc_split_arrays_cosmic.append(split_array[1])
        all_adc_split_arrays_both.append(split_array[2])
    return ak.Array(all_adc_split_arrays_threshold),ak.Array(all_adc_split_arrays_cosmic),ak.Array(all_adc_split_arrays_both)
