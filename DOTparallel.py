'''
This is the parallelized version of the DOTnbeam code. It does the same thing over multiple cores, 
but currently lacks the ability to pickle states for resuming an interrupted process.

This program uses a dot product to correlate power in target and off-target beams 
in an attempt to quantify the localization of identified signals.
Additionally, the SNR-ratio between the beams is evaluated as well
for comparison with the expected attenuation value.

DOT_utils.py and plot_target_utils.py are required for modularized functions.

The outputs of this program are:
    1. csv of the full dataframe of hits, used for plotting with plot_DOT_hits.py
    2. diagnostic histogram plot
    3. plot of SNR-ratio vs Correlation Score
    4. logging output text file

Typical basic command line usage looks something like:
    python NbeamAnalysis/DOTparallel.py <dat_dir> -f <fil_dir> -sf -o <output_dir>

NOTE: the subdirectory tree structure for the <dat_dir> and <fil_dir> must be identical
'''

    # Import Packages
import pandas as pd
import numpy as np
import time
import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter
import DOT_utils as DOT
import logging
import psutil
import threading
from multiprocessing import Pool, Manager, Lock, Process
from plot_utils import diagnostic_plotter

    # Define Functions
# monitor and print CPU usage during the parallel execution
def monitor_cpu_usage(samples):
    while not exit_flag.is_set():
        cpu_usage = psutil.cpu_percent(interval=1, percpu=True)
        samples.append(cpu_usage)

# parse the input arguments:
def parse_args():
    parser = argparse.ArgumentParser(description='Process ATA 2beam filterbank data.')
    parser.add_argument('datdir', metavar='/observation_base_dat_directory/', type=str, nargs=1,
                        help='full path of observation directory with subdirectories for integrations and seti-nodes containing dat tuples')
    parser.add_argument('-f','--fildir', metavar='/observation_base_fil_directory/', type=str, nargs=1,
                        help='full path of directory with same subdirectories leading to fil files, if different from dat file location.')
    parser.add_argument('-o', '--outdir',metavar='/output_directory/', type=str, nargs=1,default='./',
                        help='output target directory')
    parser.add_argument('-b', '--beam',metavar='target_beam',type=str,nargs=1,default='0',
                        help='target beam, 0 or 1. Default is 0.')
    parser.add_argument('-tag', '--tag',metavar='tag',type=str,nargs=1,default=None,
                        help='output files label')
    parser.add_argument('-ncore', type=int, nargs='?', default=None,
                        help='number of cpu cores to use in parallel')
    parser.add_argument('-sf', type=float, nargs='?', const=4, default=None,
                        help='flag to turn on spatial filtering with optional attenuation value for filtering')
    parser.add_argument('-before', '--before', type=str,nargs=1,default=None,
                        help='MJD before which observations should be processed')
    parser.add_argument('-after', '--after', type=str,nargs=1,default=None,
                        help='MJD after which observations should be processed')
    parser.add_argument('-bliss', '--bliss', action="store_true",
                        help='set this flag to True if using bliss instead of turboSETI')
    parser.set_defaults(bliss=False)
    args = parser.parse_args()

    args = check_cmd_args(args)
    
    return args

def check_cmd_args(args):
    # Check for trailing slash in the directory path and add it if absent
    odict = args if type(args) == dict else vars(args)
    assert "datdir" in odict, print("No required data directory!")
    # odict["datdir"] += "/"
    datdir = odict["datdir"]
    if datdir[-1] != "/":
        datdir += "/"
    odict["datdir"] = datdir  
    if odict["fildir"]:
        # fildir = odict["fildir"][0]
        fildir = odict["fildir"]
        if fildir[-1] != "/":
            fildir += "/"
        odict["fildir"] = fildir  
    else:
        odict["fildir"] = datdir
    if odict["outdir"]:
        # outdir = odict["outdir"][0]
        outdir = odict["outdir"]
        if outdir[-1] != "/":
            outdir += "/"
        odict["outdir"] = outdir  
    else:
        odict["outdir"] = ""

    # Set defaults when not using parser bc not using cmd line args
    if "beam" not in odict:
        odict["beam"] = "0"
    if "ncore" not in odict:
        odict["ncore"] = None
    if "after" not in odict:
        odict["after"] = None
    if "bliss" not in odict:
        odict["bliss"] = False
    # Returns the input argument as a labeled array
    return odict

    # dat processing function for parallelization.
# perform cross-correlation to pare down the list of hits if flagged with sf, 
# and put the remaining hits into a dataframe.
def dat_to_dataframe(args):
    dat, datdir, fildir, outdir, obs, sf, count_lock, proc_count, ndats, before, after = args
    start = time.time()

    dd_time_dst = f"/mnt/primary/scratch/igerrard/ASP/"+r"2024-12-13-02:09:48/"+"Finer/DOTParallel2/data_to_dataframe/"+dat+"/"
    dataframe_profiler = profile_manager.start_profiler("proc", "1_dat_to_dataframe", dd_time_dst, restart=RESTART)

    """ATTENTION - 0.0, 132m, 136m, 76m, 64m, 84m, 125m, 190m, 75m, 0.0, 3m, 69m, 38m
    BUT maybe not and is profiler issue ???
    maybe only taking a while because dont close profiler if not fils cases 
    """
    dataframe_profiler.add_section("with count_lock")
    with count_lock:
        count_lock_profiler = profile_manager.start_profiler("proc", "2_with_count_lock", dd_time_dst, restart=RESTART)
        dat_name = "/".join(dat.split("/")[-2:])
        # get the common subdirectories with trailing "/"
        count_lock_profiler.add_section("get the common subdirectories with trailing ")
        subdirectories="/".join(dat.replace(datdir,"").split("/")[:-1])+"/"
        fil_MJD="_".join(dat.split('/')[-1].split("_")[:3])
        proc_count.value += 1
        # optionally skip if outside input MJD bounds
        count_lock_profiler.add_section("Optionally skip if outside input MJD bounds")
        if before and float(".".join(fil_MJD[4:].split("_"))[:len(before[0])]) >= float(".".join(before[0].split("_"))):
            logging.info(f'Skipping dat file {proc_count.value}/{ndats} occurring after input MJD ({before[0]}):\n\t{dat_name}')
            count_lock_profiler.end_and_save_profiler()
            profile_manager.active_profilers.remove(count_lock_profiler)
            dataframe_profiler.end_and_save_profiler()
            profile_manager.active_profilers.remove(dataframe_profiler)
            return pd.DataFrame(),0,1,0
        if after and float(".".join(fil_MJD[4:].split("_"))[:len(after[0])]) <= float(".".join(after[0].split("_"))):
            logging.info(f'Skipping dat file {proc_count.value}/{ndats} occurring before input MJD ({after[0]}):\n\t{dat_name}')
            count_lock_profiler.end_and_save_profiler()
            profile_manager.active_profilers.remove(count_lock_profiler)
            dataframe_profiler.end_and_save_profiler()
            profile_manager.active_profilers.remove(dataframe_profiler)
            return pd.DataFrame(),0,1,0
        logging.info(f'\nProcessing dat file {proc_count.value}/{ndats}\n\t{dat_name}')
        hits,skipped,exact_matches=0,0,0
        # make a tuple with the corresponding fil/h5 files
        # fils=sorted(glob.glob(fildir+subdirectories+fil_MJD+'*fil'))
        count_lock_profiler.add_section("Sorting subdirectories")
        fils=sorted(glob.glob(fildir+subdirectories+os.path.basename(os.path.splitext(dat)[0])[:-4]+'????*fil'))
        if not fils:
            print("No fil files. Looking for fbh5/h5 instead.")
            # TODO this step is taking a while - probs because using sorting
            # also pretty sure the sorting is doing something weird anyway
            count_lock_profiler.add_section("If not fils - Sorting subdirectories again ?")
            fils=sorted(glob.glob(fildir+subdirectories+os.path.basename(os.path.splitext(dat)[0])[:-4]+'????*h5'))
        if not fils:
            count_lock_profiler.add_section("Skipping because could not locate filterbank files ?")
            logging.info(f'\tWARNING! Could not locate filterbank files in:\n\t{fildir+dat.split(datdir)[-1].split(dat.split("/")[-1])[0]}')
            logging.info(f'\tSkipping...\n')
            skipped+=1
            mid, time_label = DOT.get_elapsed_time(start)
            logging.info(f"Finished processing in %.2f {time_label}." %mid)
            count_lock_profiler.end_and_save_profiler()
            profile_manager.active_profilers.remove(count_lock_profiler)
            dataframe_profiler.end_and_save_profiler()
            profile_manager.active_profilers.remove(dataframe_profiler)
            return pd.DataFrame(),hits,skipped,exact_matches
        elif len(fils)==1:
            logging.info(f'\tWARNING! Could only locate 1 filterbank file in:\n\t{fildir+dat.split(datdir)[-1].split(dat.split("/")[-1])[0]}')
            logging.info(f'\tProceeding with caution...')
        count_lock_profiler.end_and_save_profiler()
        profile_manager.active_profilers.remove(count_lock_profiler)
        """END - ATTENTION"""

        # make a dataframe containing all the hits from all the dat files in the tuple and sort them by frequency
        """OKAY - load_dat_sf takes 0.0 seconds"""
        dataframe_profiler.add_section("DOT.load_dat_df")
        load_dat_profiler = profile_manager.start_profiler("proc", "3_load_dat_df", dd_time_dst, restart=RESTART)
        load_dat_profiler.add_section("DOT.load_dat_df")
        df0 = DOT.load_dat_df(dat,fils)
        load_dat_profiler.add_section("Sort by Corrected_frequency")
        df0 = df0.sort_values('Corrected_Frequency').reset_index(drop=True)
        if df0.empty:
            load_dat_profiler.add_section("df0.empty True")
            logging.info(f'\tWARNING! No hits found in this dat file.')
            logging.info(f'\tSkipping...')
            skipped+=1
            mid, time_label = DOT.get_elapsed_time(start)
            logging.info(f"Finished processing in %.2f {time_label}." %mid)
            load_dat_profiler.end_and_save_profiler()
            profile_manager.active_profilers.remove(load_dat_profiler)
            dataframe_profiler.end_and_save_profiler()
            profile_manager.active_profilers.remove(dataframe_profiler)
            return pd.DataFrame(),hits,skipped,exact_matches
        load_dat_profiler.end_and_save_profiler()
        profile_manager.active_profilers.remove(load_dat_profiler)
        """END - load_dat_sf takes 0.0 seconds"""

        """ATTENTION - Takes 1.5 min per NODE, 5s, 3s, 7s, 8s, 7s, 0, 0, 0, 1, 0, 0, 14, 1, 0, 0"""
        sf_profiler = profile_manager.start_profiler("proc", "4_sf", dd_time_dst, restart=RESTART)
        dataframe_profiler.add_section("Apply spatial filtering if turned on with sf flag")
        # apply spatial filtering if turned on with sf flag (default is off)
        if sf!=None:  
            sf_profiler.add_section("DOT.cross_ref")
            df = DOT.cross_ref(df0,sf)
            exact_matches+=len(df0)-len(df)
            hits+=len(df0)
            mid, time_label = DOT.get_elapsed_time(start)
            logging.info(f"\t{len(df0)-len(df)}/{len(df0)} hits removed as exact frequency matches in %.2f {time_label}." %mid)
            start = time.time()
        else:
            sf_profiler.add_section("sf == None")
            df = df0
            hits+=len(df0)
        sf_profiler.end_and_save_profiler()
        profile_manager.active_profilers.remove(sf_profiler)
        """END - Takes 1.5 min per NODE"""

        """ATTENTION - takes 2 min, 4.5 min, 70 min, 48s, 1min40, 3m16, 4m30, 1m60, 1m40, 38m, 34m, 15m, 36
        12221.138 12221.138 12221.138 12221.138 {method 'enable' of '_lsprof.Profiler' objects}
        """
        comb_profiler = profile_manager.start_profiler("proc", "5_comb_and_correlate", dd_time_dst, restart=RESTART)
        dataframe_profiler.add_section("Comb through the dataframe, correlate beam power for each hit and calculate attenuation with SNR-ratio")
        # comb through the dataframe, correlate beam power for each hit and calculate attenuation with SNR-ratio
        if df.empty:
            comb_profiler.add_section("Empty dataframe")
            logging.info(f'\tWARNING! Empty dataframe constructed after spatial filtering of dat file.')
            logging.info(f'\tSkipping this dat file because there are no remaining hits to comb through...')
            skipped+=1
            mid, time_label = DOT.get_elapsed_time(start)
            logging.info(f"Finished processing in %.2f {time_label}." %mid)
            comb_profiler.end_and_save_profiler()
            profile_manager.active_profilers.remove(comb_profiler)
            dataframe_profiler.end_and_save_profiler()
            profile_manager.active_profilers.remove(dataframe_profiler)
            return pd.DataFrame(),hits,skipped,exact_matches
        else:
            """ATTENTION - NO THIS IS THE PART TAKING TIME 
            """
            comb_df_profiler = profile_manager.start_profiler("proc", "6_DOT.comb_df", dd_time_dst, restart=RESTART)
            comb_profiler.add_section(f"\tCombing through the remaining {len(df)} hits.")
            logging.info(f"\tCombing through the remaining {len(df)} hits.")
            print(np.shape(df))
            temp_df = DOT.comb_df(df,outdir,obs,pickle_off=True,sf=sf)
            comb_df_profiler.end_and_save_profiler()
            profile_manager.active_profilers.remove(comb_df_profiler)
            """END - ATTENTION"""
        mid, time_label = DOT.get_elapsed_time(start)
        logging.info(f"Finished processing in %.2f {time_label}." %mid)
        comb_profiler.end_and_save_profiler()
        profile_manager.active_profilers.remove(comb_profiler)
        dataframe_profiler.end_and_save_profiler()
        profile_manager.active_profilers.remove(dataframe_profiler)
        return temp_df,hits,skipped,exact_matches


    # Main program execution
def main(cmd_args):
    scan_time_dst = f"/mnt/primary/scratch/igerrard/ASP/"+r"2024-12-13-02:09:48/"+"Finer/DOTParallel2/"
    dp_profiler = profile_manager.start_profiler("scan", 0, scan_time_dst, dataset = SCAN, restart=RESTART)

    """OKAY - Threading takes 0.0 seconds"""
    dp_profiler.add_section("Threading to monitor CPU usafe during parallel execution")
    start=time.time()

    global exit_flag
    exit_flag = threading.Event()
    samples=[]  # Store CPU usage samples
    # Start a thread to monitor CPU usage during parallel execution
    monitor_thread = threading.Thread(target=monitor_cpu_usage, args=(samples,))
    monitor_thread.start()
    """END - Threading takes 0.0 seconds"""

    """OKAY - Args + File Management takes 0.0 seconds"""
    dp_profiler.add_section("Args + file management stuff")
    # parse the command line arguments
    # cmd_args = parse_args()
    datdir = cmd_args["datdir"]     # required input
    fildir = cmd_args["fildir"]     # optional (but usually necessary)
    beam = cmd_args["beam"][0]      # optional, default = 0
    beam = str(int(beam)).zfill(4)  # force beam format as four char string with leading zeros. Ex: '0010'
    outdir = cmd_args["outdir"]     # optional (defaults to current directory)
    tag = cmd_args["tag"]           # optional file label, default = None
    ncore = cmd_args["ncore"]       # optional, set number of cores to use, default = all
    sf = cmd_args["sf"]             # optional, flag to turn off spatial filtering
    before = cmd_args["before"]     # optional, MJD to limit observations
    after = cmd_args["after"]       # optional, MJD to limit observations
    bliss = cmd_args["bliss"]       # optional, set True if using bliss

    # create the output directory if the specified path does not exist
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    # set a unique file identifier if not defined by input
    if tag == None:
        try:
            obs="obs_"+"-".join([i.split('-')[1:3] for i in datdir.split('/') if ':' in i][0])
        except:
            obs="obs_UNKNOWN"
    else:
        obs = tag[0]
    """END - Args + File Management takes 0.0 seconds"""

    """OKAY - Configure logging takes 0.0 seconds"""
    dp_profiler.add_section("Configure Logging")
    # configure the output log file
    logfile=outdir+f'{obs}_out.txt'
    completion_code="Program complete!"
    if os.path.exists(logfile):
        searchfile=open(logfile,'r').readlines()
        for line in searchfile:
            if completion_code in line:
                os.remove(logfile)
                break
    
    
    # file_handler = logging.FileHandler(log_filename) giving permission denied 
    # logfile = f"{os.getcwd()}/{logfile}"
    logfile = logfile
    DOT.setup_logging(logfile)
    logger = logging.getLogger()
    logging.info("\nExecuting program...")
    logging.info(f"Initial CPU usage for each of the {os.cpu_count()} cores:\n{psutil.cpu_percent(percpu=True)}")
    """END - Configure logging takes 0.0 seconds"""

    # find and get a list of tuples of all the dat files corresponding to each subset of the observation
    """
    Dot.get_dats takes <1 second (~.7s)
    Because of os.walk
    """
    dats_profiler = profile_manager.start_profiler("proc", "1_DOT.get_dats", scan_time_dst, restart=RESTART)
    dp_profiler.add_section("DOT.get_dats : find and get a list of tuples of all the dat files corresponding to each subset of the observation")
    dats_profiler.add_section("DOT.get_dats")
    dat_files,errors = DOT.get_dats(datdir,beam,bliss)

    # make sure dat_files is not empty
    if not dat_files:
        logging.info(f'\n\tERROR: No .dat files found in subfolders.'+
                f'Please check the input directory and/or beam number, and then try again:\n{datdir}\n')
        sys.exit()
    if errors:
        logging.info(f'{errors} errors when gathering dat files in the input directory. Check the log for skipped files.')

    if sf==None:
        logging.info("\nNo spatial filtering being applied since sf flag was not toggled on input command.\n")
    dats_profiler.end_and_save_profiler()
    profile_manager.active_profilers.remove(dats_profiler)
    """END - Dot.get_dats takes <1 second (~.7s)"""

    """ATTENTION - Parallelization is taking all the time - about 3.5 hours"""
    dp_profiler.add_section("Start Parallelization")
    parallel_profiler = profile_manager.start_profiler("proc", "2_parallelization", scan_time_dst, restart=RESTART)
    ndats=len(dat_files)

    """OKAY - takes 0.0 seconds"""
    parallel_profiler.add_section("Get num_processes")
    # Here's where things start to get fancy with parellelization
    if ncore==None:
        num_processes = os.cpu_count()
    else:
        num_processes = ncore
    logging.info(f"\n{num_processes} cores requested by user for parallel processing.")
    """OKAY - takes 0.0 seconds"""

    parallel_profiler.add_section("Initialize manager for shared variables")
    # Initialize the Manager object for shared variables
    manager = Manager()
    count_lock = manager.Lock()
    proc_count = manager.Value('i', 0)  # Shared integer to track processed count
    # Execute the parallelized function

    """ATTENTION - takes 3+ HOURS"""
    parallel_profiler.add_section("Execute parallelized function")
    input_args = [(dat_file, datdir, fildir, outdir, obs, sf, count_lock, proc_count, ndats, before, after) for dat_file in dat_files]
    with Pool(num_processes) as pool:
        results = pool.map(dat_to_dataframe, input_args)

    parallel_profiler.end_and_save_profiler()
    profile_manager.active_profilers.remove(parallel_profiler)
    """END - Parallelization is taking all the time - about 3.5 hours"""

    # TODO
    """BREAKS HERE"""
    dp_profiler.add_section("Processing results")
    results_profiler = profile_manager.start_profiler("proc", "3_results", scan_time_dst, restart=RESTART)
    # Process the results as needed
    results_profiler.add_section("Process results as needed")
    result_dataframes, hits, skipped, exact_matches = zip(*results)

    # Concatenate the dataframes into a single dataframe
    results_profiler.add_section("Concatenate the dataframes into a single dataframe")
    full_df = pd.concat(result_dataframes, ignore_index=True)
    test_dst = os.path.join(os.getcwd(), f"{outdir}{obs}_DOTnbeam.csv")
    full_df.to_csv(test_dst)
    print(np.shape(full_df))

    # Do something with the counters if needed
    results_profiler.add_section("Do something with counters if needed")
    total_hits = sum(hits)
    total_skipped = sum(skipped)
    total_exact_matches = sum(exact_matches)

    results_profiler.end_and_save_profiler()
    profile_manager.active_profilers.remove(results_profiler)

    if sf==None:
        sf=4 

    dp_profiler.add_section("Handling SNR_ratio")

    if 'SNR_ratio' in full_df.columns and full_df['SNR_ratio'].notnull().any():
        snr_profiler = profile_manager.start_profiler("proc", "4_snr_ratio", scan_time_dst, restart=RESTART)
        snr_profiler.add_section("Plot histograms for hits within the target beam")
        # plot the histograms for hits within the target beam
        diagnostic_plotter(full_df, obs, saving=True, outdir=outdir)

        # plot the SNR ratios vs the correlation scores for each hit in the target beam
        snr_profiler.add_section("Plot theSNR ratios vs the correlation scores for each hit in the target beam")
        x = full_df.corrs
        SNRr = full_df.SNR_ratio
        fig,ax=plt.subplots(figsize=(12,10))
        plt.scatter(x,SNRr,color='orange',alpha=0.5,edgecolor='k')
        plt.xlabel('Correlation Score')
        plt.ylabel('SNR-ratio')
        ylims=plt.gca().get_ylim()
        xlims=plt.gca().get_xlim()
        xcutoff=np.linspace(-0.05,1.05,1000)
        ycutoff=np.array([0.9*sf*max(j-0.05,0)**(1/3) for j in xcutoff])
        plt.plot(xcutoff,ycutoff,linestyle='--',color='k',alpha=0.5,label='Nominal Cutoff')
        plt.axhspan(sf,max(ylims[1],10),color='green',alpha=0.25,label='Attenuated Signals')
        plt.axhspan(1/sf,sf,color='grey',alpha=0.25,label='Similar SNRs')
        plt.axhspan(1/max(ylims[1],10),1/sf,color='brown',alpha=0.25,label='Off-beam Attenuated')
        plt.ylim(1/max(ylims[1],10),max(ylims[1],10))
        plt.yscale('log')
        ax.yaxis.set_major_formatter(ScalarFormatter())
        plt.xlim(-0.1,1.1)
        plt.legend().get_frame().set_alpha(0) 
        plt.grid(which='major', axis='both', alpha=0.5,linestyle=':')
        plt.savefig(outdir + f'{obs}_SNRx.png',
                    bbox_inches='tight',format='png',dpi=fig.dpi,facecolor='white', transparent=False)
        plt.close()

        above_cutoff=0
        for i,score in enumerate(x):
            if np.interp(score,xcutoff,ycutoff)<SNRr[i]:
                above_cutoff+=1

        snr_profiler.end_and_save_profiler()
        profile_manager.active_profilers.remove(snr_profiler)
    logging.info(f"\n**Final results:")
    
    # Final print block
    dp_profiler.add_section("File print block")
    if total_skipped>0:
        logging.info(f'\n\t{total_skipped}/{ndats} dat files skipped. Check the log for skipped filenames.\n')
    end, time_label = DOT.get_elapsed_time(start)
    logging.info(f"\t{len(dat_files)} dats with {total_hits} total hits cross referenced and {total_exact_matches} hits removed as exact matches.")
    logging.info(f"\tThe remaining {total_hits-total_exact_matches} hits were correlated and processed in \n\n\t\t%.2f {time_label}.\n" %end)
    if 'SNR_ratio' in full_df.columns and full_df['SNR_ratio'].notnull().any():
        logging.info(f"\t{len(full_df[full_df.SNR_ratio>sf])}/{len(full_df)} hits above a SNR-ratio of {sf:.1f}\n")
        logging.info(f"\t{above_cutoff}/{len(full_df)} hits above the nominal cutoff.")
    elif 'SNR_ratio' in full_df.columns:
        logging.info(f"\n\tSNR_ratios missing for some hits, possibly due to missing filterbank files. Please check the log.")
    elif 'mySNRs'==full_df.columns[-1]:
        logging.info(f"\n\tSingle SNR calculated, possibly due to only one filterbank file being found. Please check the log.")
    
    if 'SNR_ratio' not in full_df.columns or full_df['SNR_ratio'].isnull().any():
        # save the broken dataframe to csv
        logging.info(f"\nScores in full dataframe not filled out correctly. Please check it:\n{outdir}{obs}_DOTnbeam.csv")
    else:
        logging.info(f"\nThe full dataframe was saved to: {outdir}{obs}_DOTnbeam.csv")

    # Signal the monitoring thread to exit
    dp_profiler.add_section("Signal the monitoring thread to exit")
    exit_flag.set()

    # Allow some time for the monitoring thread to finish
    dp_profiler.add_section("Allow some time for the monitoring thread to finish")
    monitor_thread.join(timeout=5)  # Adjust the timeout if needed

    # Calculate and print the average CPU usage over time
    dp_profiler.add_section("Calculate and print the average CPU usage over time")
    if samples:
        num_samples = len(samples)
        num_cores = len(samples[0])
        avg_cpu_usage = [round(sum(cpu_usage[i] for cpu_usage in samples) / num_samples,1) for i in range(num_cores)]

        logging.info(f"\nFinal average CPU usage over {num_cores} cores:")
        logging.info(avg_cpu_usage)

    logging.info(f"\n\tProgram complete!\n")
    return None

#%% run it!
NIGHT = r"2024-12-13-02:09:48/"
SCAN = r"fil_60657_39349_70813781_radec5.389,23.168_0001/"
RESTART = False

if __name__ == "__main__":
    # in this case the arguments are given as command line arguments
    # cmd_args = parse_args()
    # main(cmd_args)

    sys.path.append("../")
    from time_profiler import ProfileManager, TimeProfiler

    night, scan = NIGHT, SCAN
    datdir = f"/mnt/primary/scratch/igerrard/ASP/bliss_outputs/{night}/{scan}/" # required input
    # datdir = f"{os.getcwd()}/bliss_outputs/{night}/{scan}/LoB.C1120/" # required input # TODO 
    outdir = f"/mnt/primary/scratch/igerrard/ASP/Nbeam_output_FULLSCAN/" # -o # optional (defaults to current directory) 
    tag = "ASP_test_FULLSCAN" # optional file label, default = None 
    # fildir = "." # -f # optional (but usually necessary)
    fildir = f"/mnt/primary/scratch/igerrard/ASP/" # -f # optional (but usually necessary)
    before = 0 # -b # optional, MJD to limit observations
    sf = 5.29 # optional, flag to turn off spatial filtering
    bliss = True
    # equivalent to cmd_args = parse_args() in Dotparallel
    cmd_args = {"datdir":datdir, "outdir":outdir, "fildir":fildir, "tag":tag, "before":before, "sf":sf, "bliss":bliss}
    # and set defaults 
    cmd_args = check_cmd_args(cmd_args)

    profile_manager = ProfileManager()

    try:
        main(cmd_args)
    except KeyboardInterrupt:
        print("\n\tExiting with Ctrl+C\n\tCleaning up...")
    finally:
        # Clean up active profilers
        profile_manager.stop_and_save_all()

    
