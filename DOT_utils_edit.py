# This file holds functions utilized primarily in DOTnbeam.py

# all imports
import pandas as pd
import numpy as np
import logging
import pickle
import time
import os
import glob
import argparse
import sys
# export PYTHONPATH=$PYTHONPATH:/path/to/your/module
import blimpy as bl
# print(bl.__file__)
import logging
from pathlib import Path
from numba import jit

# hdf_reader.examine_h5(None)
# logging function
def setup_logging(log_filename):
    if not os.path.exists(log_filename):
        # create parent folder (if dont split then turns filename into a dir)
        Path('/'.join(log_filename.split('/')[:-1])).mkdir(parents=True, exist_ok=True)
        # create file
        open(log_filename, 'w+').close()

    # Import the logging module and configure the root logger
    logging.basicConfig(level=logging.INFO, filemode='w', format='%(message)s')

    # Get the root logger instance
    logger = logging.getLogger()

    # Remove any existing handlers from the logger
    for handler in logger.handlers:
        logger.removeHandler(handler)

    # Create a console handler that writes to sys.stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    # Create a file handler that writes to the specified file
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # Set the logger as the default logger for the logging module
    logging.getLogger('').handlers = [console_handler, file_handler]
    return None

def get_specific_logger(logfile):
    """
    Allows to have multiple logger files. Creates root logger if does not exist. 
    Allows multiple processes to log without having to synchronize them so they don't block each other. 
    """
    if not os.path.exists(logfile):
        # create parent folder (if dont split then turns filename into a dir)
        Path('/'.join(logfile.split('/')[:-1])).mkdir(parents=True, exist_ok=True)
        # create file
        open(logfile, 'w+').close()

    try: # setup logging if haven't yet
        root_logger = logging.getLogger()
    except:
        setup_logging(logfile)
        root_logger = logging.getLogger()

    # creates named logger
    specific_logger = logging.getLogger(logfile)

    # doesn't already exist
    if not specific_logger.handlers:
        # Create a file handler that writes to the specified file
        file_handler = logging.FileHandler(logfile)
        file_handler.setLevel(logging.INFO)
        specific_logger.addHandler(file_handler)
        # TODO do i need this ?
        # root_logger.addHandler(file_handler)

    return specific_logger

# elapsed time function
def get_elapsed_time(start=0):
    end = time.time() - start
    time_label = 'seconds'    
    if end > 3600:
        end = end/3600
        time_label = 'hours'
    elif end > 60:
        end = end/60
        time_label = 'minutes'
    return end, time_label

# check log file for completeness
def check_logs(log, bliss=False):
    status="fine"
    if bliss: # prints a million times
        # print("Logging disabled (likely for functionality between bliss and NBeamAnalysis), see also edit to get_dats")
        return status
    else:
        if not os.path.exists(log):
            logging.info("Can't find log file...")
            return "incomplete"
        searchfile = open(log,'r').readlines()
        if searchfile[-1]!='===== END OF LOG\n':
            status="incomplete"
    return status

def get_beam_code(fil_string):
    return fil_string.split('beam')[-1].split('.')[0]

# retrieve a list of .dat files, each with the full path of each .dat file for the target beam
def get_dats(root_dir,beam,bliss):
    """Recursively finds all files with the '.dat' extension in a directory
    and its subdirectories, and returns a list of the full paths of files 
    where each file corresponds to the target beam."""
    errors=0
    dat_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # if 'test' in dirpath:
        #     print(f'Skipping "test" folder:\n{dirpath}')
        #     continue
        if bliss: # print only once
            print("Logging disabled (likely for functionality between bliss and NBeamAnalysis), see also edit to get_dats")
        for f in filenames:
            if f.endswith('.dat') and f.split('beam')[-1].split('.')[0]==beam:
                if not bliss: # currently bliss does not have log files
                    log_file = os.path.join(dirpath, f).replace('.dat','.log')
                    if check_logs(log_file, bliss=bliss)=="incomplete": #or not os.path.isfile(log_file): <---- this is the edit to get_dats()
                        logging.info(f"{log_file} is incomplete. Please check it. Skipping this file...")
                        errors+=1
                        continue
                dat_files.append(os.path.join(dirpath, f))
    return dat_files,errors

# load the data from the input .dat file for the target beam and its corresponding .fil files 
# for all beams formed in the same observation and make a single concatenated dataframe 
def load_dat_df(dat_file,filtuple):
    # make a dataframe of all the data in the .dat file below the headers
    #NOTE: This assumes a standard turboSETI .dat file format with the listed headers
    dat_df = pd.read_csv(dat_file, 
                delim_whitespace=True, 
                names=['Top_Hit_#','Drift_Rate','SNR', 'Uncorrected_Frequency','Corrected_Frequency','Index',
                        'freq_start','freq_end','SEFD','SEFD_freq','Coarse_Channel_Number','Full_number_of_hits'],
                skiprows=9)
    # initiate the final dataframe as a subset of the relevant bits of the .dat dataframe
    full_dat_df = dat_df[['Drift_Rate','SNR', 'Index', 'Uncorrected_Frequency','Corrected_Frequency',
                            'freq_start','freq_end','Coarse_Channel_Number','Full_number_of_hits']]
    # add the path and filename of the .dat file to the dataframe
    full_dat_df = full_dat_df.assign(dat_name = dat_file)
    # loop over each .fil file in the tuple to add to the dataframe
    for i,fil in enumerate(filtuple):
        ext = os.path.splitext(fil)[1]
        col_name = 'fil_'+fil.split('beam')[-1].split(ext)[0]
        full_dat_df[col_name] = fil
    # calculate the drift rate in nHz for each hit and add it to the dataframe
    full_dat_df['normalized_dr'] = full_dat_df['Drift_Rate'] / (full_dat_df[['freq_start','freq_end']].max(axis=1) / 10**3)
    return full_dat_df

# use blimpy to grab the data slice from the filterbank file over the frequency range provided
"""ATTENTION - this takes .5 sec about 1300 calls"""
def wf_data(fil,f1,f2):
    # print("\t**in wf_data calling bl.waterfall**", flush=True)
    # return bl.Waterfall(fil,f1,f2).grab_data(f1,f2)
    wat = bl.Waterfall(fil,f1,f2)
    print(f"time: ({wat.container.t_start}, {wat.container.t_stop})")
    print(f"chan: ({wat.container.chan_start_idx}, {wat.container.chan_stop_idx})")
    return wat.grab_data(f1,f2)

# def wf_blob_data(fil,f1,f2):
#     idx_start, idx_stop = self.get_frequency_indices(f1, f2)

#     # Define blob dimensions (time, beams, freq range)
#     blob_dim = (self.reader.n_ints_in_file, 1, idx_stop - idx_start)

#     # Use `read_blob` to extract the data
#     blob = self.reader.read_blob(blob_dim, n_blob=0)

# get the normalization factor of a 2D array
@jit(nopython=True) # should be automatic for vectorized multi dim
def ACF(s1):
    return ((s1*s1).sum(axis=1)).sum()/np.shape(s1)[0]/np.shape(s1)[1]

# correlate two 2D arrays with a dot product and return the correlation score
@jit(nopython=True)
def sig_cor(s1,s2):
    ACF1=ACF(s1)
    ACF2=ACF(s2)
    DOT =((s1*s2).sum(axis=1)).sum()/np.shape(s1)[0]/np.shape(s1)[1]
    x=DOT/np.sqrt(ACF1*ACF2)
    return x

# get the median of the "noise" after removing the bottom and top 5th percentile of data
@jit(nopython=True)
def noise_median(data_array,p=5):
    return np.median(mid_90(data_array,p))

# get the standard deviation of the "noise" after removing the bottom and top 5th percentile of data
@jit(nopython=True)
def noise_std(data_array,p=5):
    return np.std(mid_90(data_array,p))

# remove the bottom and top 5th percentile from a data array
@jit(nopython=True)
def mid_90(da,p=5):
    # return da[(da>np.percentile(da,p))&(da<np.percentile(da,100-p))]
    # jit compatible
    lower, upper = np.percentile(da,p), np.percentile(da,100-p)
    filtered_values = []
    for x in da.ravel():  # Loop over flattened array
        if lower < x < upper:
            filtered_values.append(x)
    return np.array(filtered_values, dtype=da.dtype)

# identify signals significantly above the "noise" in the data
@jit(nopython=True)
def signals_sig_above_noise(power, std_noise):
    # power[(power>10*std_noise)&(power>np.percentile(power,95))] 
    percentile_95 = np.percentile(power, 95)
    threshold = 10 * std_noise
    filtered_values = []
    for x in power.ravel():  # Loop over flattened array
        if x > threshold and x > percentile_95:
            filtered_values.append(x)
    return np.array(filtered_values, dtype=power.dtype)

# My method for calculating SNR
@jit(nopython=True)
def mySNR(power):
    # get the median of the noise
    median_noise=noise_median(power)
    # assume the middle 90 percent of the array represent the noise
    noise_els=mid_90(power)        
    # median_noise=np.median(mid_90)
    # zero out the noise by subtracting off the median
    zeroed_noise=noise_els-median_noise     
    # get the standard deviation of the noise using median instead of mean
    std_noise=np.sqrt(np.median((zeroed_noise)**2))
    # identify signals significantly above the "noise" in the data
    # signal_els=power[(power>10*std_noise)&(power>np.percentile(power,95))] 
    signal_els = signals_sig_above_noise(power, std_noise)
    if not bool(signal_els.size):
        # if there are no "signals" popping out above the noise
        # this will result in an SNR of 1
        signal=std_noise
    else:
        # the signal is calculated as the median of the highest N elements in the signal candidates
        # where N is the number of time bins or rows in the data matrix
        # signal=np.median(sorted(signal_els)[-np.shape(power)[0]:])-median_noise 
        signal=np.median(np.sort(signal_els)[-np.shape(power)[0]:])-median_noise # jit compatible - sorted makes python list
    # subtract off the median (previous step) and divide by the standard deviation to get the SNR
    SNR = signal/std_noise
    return SNR

# extract index and dataframe from pickle files to resume from last checkpoint
def resume(pickle_file, df):
    index = 0 # initialize at 0
    if os.path.exists(pickle_file):
        # If a checkpoint file exists, load the dataframe and row index from the file
        with open(pickle_file, "rb") as f:
            index, df = pickle.load(f)
        logging.info(f'\t***pickle checkpoint file found. Resuming from step {index+1}\n')
    return index, df


# comb through each hit in the dataframe and look for corresponding hits in each of the beams.
# def comb_df(df, outdir='./', obs='UNKNOWN', resume_index=None, pickle_off=False, sf=4):
def comb_df(df, outdir='./', obs='UNKNOWN', resume_index=None, pickle_off=False, sf=4, proc_count=0): # TODO debug only
    if sf==None:
        sf=4
    """
    Same target_fil for every row in a given data frame = always same metadata
    Don't recalc metadata constants for each hit
    """

    print("in DOT_utils_edit")
    # identify the target beam .fil file
    first_row = df.iloc[0]
    matching_col = first_row.filter(like='fil_').apply(lambda x: x == first_row['dat_name']).idxmax()
    target_fil = first_row[matching_col]

    """
    Access process specific log file so can write within its process wihtout needing to synchronize
    """
    node_name = first_row['dat_name'].split("/")[-2]
    # TODO doens't apply to other nbeam users
    fil_name = first_row['dat_name'].split("/")[-1][10:15] # scan identifier
    logfile=outdir+f'/{node_name}_out.txt'
    curr_proc_logger = get_specific_logger(logfile)

    try:
        print(f"[{node_name}] fetching meta for {target_fil}")
        # curr_proc_logger.info(f"fetching meta for {target_fil}")
        fil_meta = bl.Waterfall(target_fil,load_data=False)
    except Exception as e:
        curr_proc_logger.warning(f"Failed to load fil meta with bl.Waterfall for {target_fil} - skipping this node...")
        curr_proc_logger.info(e)
        return
    
    # determine the frequency boundaries in the .fil file
    minimum_frequency = fil_meta.container.f_start
    maximum_frequency = fil_meta.container.f_stop
    print(f"time: ({fil_meta.container.t_start}, {fil_meta.container.t_stop})")
    print(f"chan: ({fil_meta.container.chan_start_idx}, {fil_meta.container.chan_stop_idx})")
    # calculate the narrow signal window using the reported drift rate and metadata
    tsamp = fil_meta.header['tsamp']    # time bin length in seconds
    obs_length=fil_meta.n_ints_in_file * tsamp # total length of observation in seconds

    # # get a list of all the other fil files for all the other beams
    other_cols = first_row.loc[first_row.index.str.startswith('fil_') & (first_row.index != matching_col)]
    # # iteritems is deprecated, also don't use colname 
    # # for col_name, other_fil in other_cols.iteritems():
    other_fils = other_cols.values # this is index object - can use tolist()
    # other_wfs_full = [get_wf(other_fil) for other_fil in other_fils]

    beam_codes = [get_beam_code(fil) for fil in other_fils]
    col_name_corrs=[f'corrs_{beam}' for beam in beam_codes] # TODO this is always 0001 ?
    col_name_SNRr=[f'SNR_ratio_{beam}' for beam in beam_codes]

    # this is pointer to those waterfall objects 

    print(f"[{node_name}] Beginning loop through hits...\n")
    num_rows = len(df)
    for r,row in df.iterrows(): # each hit
        if r%200==0: print(f"\t[{proc_count}] [{node_name}] [{fil_name}] {r}/{num_rows}") # TODO for debug only 
        if resume_index is not None and r < resume_index:
            continue  # skip rows before the resume index

        # calculate the narrow signal window using the reported drift rate and metadata
        DR = row['Drift_Rate']              # reported drift rate
        padding=1+np.log10(row['SNR'])/10   # padding based on reported strength of signal
        # calculate the amount of frequency drift with some padding
        half_span=abs(DR)*obs_length*padding  
        if half_span<250:
            half_span=250 # minimum 500 Hz span window
        fmid = row['Corrected_Frequency']
        # signal may not be centered, could drift up or down in frequency space
        # so the frequency drift is added to both sides of the central frequency
        # to ensure it is contained within the window
        f1=round(max(fmid-half_span*1e-6,minimum_frequency),6)
        f2=round(min(fmid+half_span*1e-6,maximum_frequency),6)

        # now set f_start and f_stop of the waterfall and call read_data 
        # then grab data for frange,s0
        frange,s0=wf_data(target_fil,f1,f2) # bl.Waterfall(fil,f1,f2).grab_data(f1,f2)
        # frange,s0=wf_data_range(target_wf_full, f1,f2) # bl.Waterfall(fil).grab_data(f1,f2)
        
        # calculate the SNR
        SNR0 = mySNR(s0)
        
        # get a list of all the other fil files for all the other beams
        # other_cols = row.loc[row.index.str.startswith('fil_') & (row.index != matching_col)]
        # initialize empty lists for appending
        corrs=[]
        mySNRs=[SNR0]
        SNR_ratios=[]
        for other_fil in other_fils: #iteritems deprecated
            # grab the signal data from the non-target fil in the same location
            _,s1=wf_data(other_fil,f1,f2)
            # just grabbing data in that range without completely reloading wf again
            # calculate and append the SNR for the same location in the other beam
            off_SNR = mySNR(s1)
            mySNRs.append(off_SNR)
            # calculate and append the SNR ratio
            SNR_ratios.append(SNR0/off_SNR)
            # calculate and append the correlation score
            corrs.append(sig_cor(s0-noise_median(s0),s1-noise_median(s1)))

        df.loc[r,col_name_corrs] = corrs
        df.loc[r,col_name_SNRr] = SNR_ratios
        df.loc[r,'mySNRs'] = str(mySNRs)

        # calculate and add average values to the dataframe (useful for N>2 beams)
        if len(SNR_ratios)>0:
            df.loc[r,'corrs'] = sum(corrs)/len(corrs) 
            df.loc[r,'SNR_ratio'] = sum(SNR_ratios)/len(SNR_ratios)  
        # pickle the dataframe and row index for resuming
        if not pickle_off:
            with open(outdir+f'{obs}_comb_df.pkl', 'wb') as f:
                pickle.dump((r, df), f)
    
    # remove the pickle checkpoint file after all loops complete
    if os.path.exists(outdir+f"{obs}_comb_df.pkl"):
        os.remove(outdir+f"{obs}_comb_df.pkl") 
    return df

# cross reference hits in the target beam dat with the other beams dats for identical signals
def cross_ref(input_df,sf):
    if len(input_df)==0:
        logging.info("\tNo hits in the input dataframe to cross reference.")
        return input_df
    # first, make sure the indices are reset
    input_df=input_df.reset_index(drop=True)
    # Extract directory path from the first row of the dat_name column
    dat_path = os.path.dirname(input_df['dat_name'].iloc[0])
    # Find all dat files in the directory
    dat_files = [f for f in os.listdir(dat_path) if f.endswith('.dat')]
    # Load the hits from the other dat files into a separate dataframe and store in a list
    dat_dfs = []
    for dat_file in dat_files:
        if dat_file == os.path.basename(input_df['dat_name'].iloc[0]):
            continue  # Skip the dat file corresponding to the dat_name column
        dat_df = pd.read_csv(os.path.join(dat_path, dat_file), delim_whitespace=True,
                             names=['Top_Hit_#','Drift_Rate','SNR','Uncorrected_Frequency',
                                    'Corrected_Frequency','Index','freq_start','freq_end',
                                    'SEFD','SEFD_freq','Coarse_Channel_Number',
                                    'Full_number_of_hits'], skiprows=9)
        dat_dfs.append(dat_df)
    # Iterate through the rows in the input dataframe and prune matching hits
    rows_to_drop = []
    for idx, row in input_df.iterrows():
        drop_row = False # don't drop it unless there's a match
        # Check if values are within tolerance in any of the dat file dataframes
        for dat_df in dat_dfs:
            # check if frequencies match and if reported SNRs are similar within a factor of the expected attenuation
            within_tolerance = ((dat_df['Corrected_Frequency'] - row['Corrected_Frequency']).abs() < 2e-6) & \
                               ((((dat_df['freq_start'] - row['freq_start']).abs() < 2e-6) & \
                               ((dat_df['freq_end'] - row['freq_end']).abs() < 2e-6)) | \
                               ((dat_df['Drift_Rate'] - row['Drift_Rate']).abs() < 1/16)) & \
                               ((dat_df['SNR'] / row['SNR']).abs() >= 1/sf) & \
                               ((row['SNR'] / dat_df['SNR']).abs() <= sf)
            if within_tolerance.any():
                drop_row = True # drop it like it's hot
                break
        # Add the row index to the list of rows that should be dropped
        if drop_row:
            rows_to_drop.append(idx)
    # Drop the rows that were identified as within matching tolerance
    trimmed_df = input_df.drop(rows_to_drop)
    # Return the trimmed dataframe with a reset index
    return trimmed_df.reset_index(drop=True)

# a weak attempt at filtering out duplicate hits due to fscrunching
# not complete or implemented anywhere
def drop_fscrunch_duplicates(input_df,frez=1,time_rez=16):
    if len(input_df)==0:
        logging.info("\tNo hits in the input dataframe to cross reference.")
        return input_df
    # first, make sure the indices are reset
    input_df=input_df.reset_index(drop=True)
    # Extract directory path from the first row of the dat_name column
    dat_path = os.path.dirname(input_df['dat_name'].iloc[0])
    # Find all dat files in the directory
    dat_files = [f for f in os.listdir(dat_path) if f.endswith('.dat')]
    # Load the hits from the other dat files into a separate dataframe and store in a list
    dat_dfs = []
    for dat_file in dat_files:
        if dat_file == os.path.basename(input_df['dat_name'].iloc[0]):
            continue  # Skip the dat file corresponding to the dat_name column
        dat_df = pd.read_csv(os.path.join(dat_path, dat_file), delim_whitespace=True,
                             names=['Top_Hit_#','Drift_Rate','SNR','Uncorrected_Frequency',
                                    'Corrected_Frequency','Index','freq_start','freq_end',
                                    'SEFD','SEFD_freq','Coarse_Channel_Number',
                                    'Full_number_of_hits'], skiprows=9)
        dat_dfs.append(dat_df)
    # Iterate through the rows in the input dataframe and prune matching hits
    rows_to_drop = []
    for idx, row in input_df.iterrows():
        drop_row = False # don't drop it unless there's a match
        # Check if values are within tolerance in any of the dat file dataframes
        for dat_df in dat_dfs:
            # check if frequencies match and if reported SNRs are similar within a factor of the expected attenuation
            within_tolerance = ((dat_df['Corrected_Frequency'] - row['Corrected_Frequency']).abs() < 10e-6) & \
                               ((dat_df['Drift_Rate'] - row['Drift_Rate']).abs() < frez/time_rez) & \
                               ((dat_df['SNR'] / row['SNR']).abs() >= 1/sf) & \
                               ((row['SNR'] / dat_df['SNR']).abs() <= sf)
            if within_tolerance.any():
                drop_row = True # drop it like it's hot
                break
        # Add the row index to the list of rows that should be dropped
        if drop_row:
            rows_to_drop.append(idx)
    # Drop the rows that were identified as within matching tolerance
    trimmed_df = input_df.drop(rows_to_drop)
    # Return the trimmed dataframe with a reset index
    return trimmed_df.reset_index(drop=True)
