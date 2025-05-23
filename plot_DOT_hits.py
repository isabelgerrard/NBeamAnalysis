'''
This program plots hits listed in the csv that is output from DOTnbeam or DOTparallel.
The output waterfall plots are generated using information in the csv dataframe to access and 
plot the filterbank data for each beam over the time and frequency of the hit detected in the 
target beam. The waterfall colormap of each subplot is scaled to the data in the target beam.

Note that although the only input file is a csv file, that csv file should contain filepaths 
to the filterbank files needed to be accessed in order to generate the plots.

plot_utils.py is required for modularized functions. Various extra argument flags can be used
through terminal commands and are interpreted with the argparse package.

Typical basic command line usage looks something like:
    python NbeamAnalysis/plot_DOT_hits.py <path_to_csv> -o <output_dir>

There are 4 main plotting modes:
    1.  Default plotting first filters all hits above the nominal cutoff, calculated as 
        SNR-ratios above 0.9*sf*max(x-0.05,0)^(1/3), where sf (default=4.0) is the attenuation value, 
        x is the correlation score and 0.9 and 0.05 are included for padding. If there are more 
        than 500 (this number is adjustable with the -cutoff input flag) hits above this cutoff, 
        the hits are sorted by SNR-ratio. If more than 500 have SNR-ratios above the attenuation value, 
        sf, then they are sorted by correlation score and the 500 lowest scoring hits are plotted. 

    2.  Custom column sorting and plotting is available through optional input arguments.
        All hits can be plotted through clever employment of the optional input arguments, if desired.
        See the parse_args() function for details.

    3.  Individual plots over specified a frequency range can be made with the "-freqs" flag 
        by specifying the start and end frequencies. Note that without additional sorting/filtering, 
        this will produce a plot for each unique dat/fil in the dataframe that span the specified frequencies.
    
    4.  A stack of subplots can be plotted up using other integrations in the same observation.
        With the "-stack" argument flag plus the number of adjacent integrations to plot, 
        the program will use all the filepath information in the csv to determine the common 
        observation path and add subplots for other integrations before and after the target 
        observation, with a red frame around the target observation. The default value is 1
        when using the stack flag without an integer.
        Note: this significantly increases plotting time. 
'''

    # Import packages
import numpy as np
import pandas as pd
import blimpy as bl
import os, glob, sys
import math
import argparse
import time
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
import plot_utils as ptu

    # Define Functions
# parse the input arguments. 
# required: csv containing the hits to be plotted 
def parse_args():
    parser = argparse.ArgumentParser(description='Process ATA 2beam filterbank data.')
    parser.add_argument('csv', metavar='/plot_target_csv/', type=str, nargs=1,
                        help='CSV file with correlation scores for target beam hits')
    parser.add_argument('-o', '--outdir',metavar="", type=str, nargs=1,
                        help='output target directory')
    parser.add_argument('-col',metavar="", type=str, default=None,
                        help='name of the column in the csv to filter on')
    parser.add_argument('-op',metavar="", type=str, choices=['lt', 'gt', 'is'], default=None,
                        help='operator for filtering: less than (lt), greater than (gt), or equal to (is)')
    parser.add_argument('-val',metavar="", type=str, default=None, help='Filter value')
    parser.add_argument('-nbeams', type=int, default=2,
                        help='optional number of beams, default = 2')
    parser.add_argument('-tbeam', type=int, default=0,
                        help='optional set target beam number, default = 0')
    parser.add_argument('-stack', nargs='?', const=1, type=int,
                        help='optional flag to plot other integrations in observation as subplots')
    parser.add_argument('-zoom', nargs='?', const=10, type=float,
                        help='optional flag to zoom in/out plot in frequency space')
    parser.add_argument('-sf', type=float, default=4.0,
                        help='optional attenuation value for filtering')
    parser.add_argument('-num_backup_plots', type=int, default=1,
                        help='specifies how many plots to create if none pass the sf threshold')
    parser.add_argument('-cutoff', type=int, default=500,
                        help='optionally set the number of plots above cutoff in default plotting mode')
    parser.add_argument('-clobber', action='store_true',
                        help='overwrite files if they already exist')
    parser.add_argument('-pdf', action='store_true',
                        help='plot file format as pdf (default is png)')
    parser.add_argument('-before', '--before', type=str,nargs=1,default=None,
                        help='MJD before which observations should be plotted')
    parser.add_argument('-after', '--after', type=str,nargs=1,default=None,
                        help='MJD after which observations should be plotted')
    parser.add_argument('-fbelow', '--fbelow', type=float,nargs=1,default=None,
                        help='frequency below which hits should be plotted')
    parser.add_argument('-fabove', '--fabove', type=float,nargs=1,default=None,
                        help='frequency above which hits should be plotted')
    parser.add_argument("-freqs", type=list_parser, nargs="+", help="optional tuple of frequencies")
    args = parser.parse_args()
    odict = vars(args)
    if odict["outdir"]:
        outdir = odict["outdir"][0]
        if outdir[-1] != "/":
            outdir += "/"
        odict["outdir"] = outdir  
    return odict

def check_cmd_args(args):
    # Check for trailing slash in the directory path and add it if absent
    odict = args if type(args) == dict else vars(args)
    if odict["outdir"]:
        # outdir = odict["outdir"][0]
        outdir = odict["outdir"]
        if outdir[-1] != "/":
            outdir += "/"
        odict["outdir"] = outdir  
    else:
        odict["outdir"] = "" # this should actually be default to datdir in csv

    # Set defaults when not using parser bc not using cmd line args
    if "col" not in odict:
        odict["col"] = None
    if "op" not in odict:
        odict["op"] = None
    if "val" not in odict:
        odict["val"] = None
    if "stack" not in odict:
        odict["stack"] = None
    if "zoom" not in odict:
        odict["zoom"] = None
    if "nbeams" not in odict:
        odict["nbeams"] = 2
    if "tbeam" not in odict:
        odict["tbeam"] = 0
    if "sf" not in odict:
        odict["sf"] = 4.0
    if "num_backup_plots" not in odict:
        odict["num_backup_plots"] = 1
    if "cutoff" not in odict:
        odict["cutoff"] = 500
    if "clobber" not in odict:
        odict["clobber"] = False
    if "pdf" not in odict:
        odict["pdf"] = False
    if "before" not in odict:
        odict["before"] = None
    if "after" not in odict:
        odict["after"] = None
    if "fbelow" not in odict:
        odict["fbelow"] = None
    if "fabove" not in odict:
        odict["fabove"] = None
    if "freqs" not in odict:
        odict["freqs"] = None
    # Returns the input argument as a labeled array
    return odict

# function to help parse space separated tuple input from command line 
def list_parser(input_str):
    try:
        # Split the input string by spaces and convert each element to a float
        parsed_list = list(map(float, input_str.split()))
        return parsed_list[0]
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid list format. Use space-separated floating-point numbers.")

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

# clobber function to remove old files (optional)
def purge(dir, pattern):
    old_plots = glob.glob(dir+pattern)
    print(f'\nRemoving {len(old_plots)} old plots from {dir}\n')
    for f in old_plots:
        os.remove(f)

# filter dataframe on optional input parameters
def filter_df(df,column,operator,value):
    # Infer the data type of the column from the dataframe
    column_dtype = df[column].dtype
    # Convert filter value to the correct data type
    value = column_dtype.type(value)
    # Apply the filter based on the operator
    if operator == 'lt':
        signals_of_interest = df[df[column] < value]
    elif operator == 'gt':
        signals_of_interest = df[df[column] > value]
    elif operator == 'is':
        signals_of_interest = df[df[column] == value]
    # signals_of_interest = signals_of_interest.sort_values(by='x').reset_index(drop=True)
    signals_of_interest = signals_of_interest.sort_values(by=column,ascending=False).reset_index(drop=True)
    return signals_of_interest

# default filtering
def default_filter(df,sf,cutnum,num_backup_plots):
    print(f"\nDefault filtering:")
    print(f"Up to the {cutnum} lowest correlation scores with an SNR-ratio above the attenuation value of {sf:.2f}\n")
    print(f"Barring that, up to the {num_backup_plots} above the cutoff will be plotted")
    xcutoff=np.linspace(-0.05,1.05,1000)
    ycutoff=np.array([0.9*sf*max(j-0.05,0)**(1/3) for j in xcutoff])
    dfx=df[np.interp(df.corrs,xcutoff,ycutoff)<df.SNR_ratio].reset_index(drop=True)
    # Sort by SNR *ratio* between on and off - descending
    dfx=dfx.sort_values(by='SNR_ratio',ascending=False).reset_index(drop=True)
    dfsf=dfx[dfx.SNR_ratio>sf].reset_index(drop=True)
    # if more than cutnum hits above the SNR ratio sf 
    if len(dfsf)>cutnum:
        # then sort by correlation score (x axis)
        signals_of_interest = dfsf.sort_values(by='corrs',ascending=True).reset_index(drop=True).iloc[:cutnum]
    else:
        signals_of_interest = dfsf.iloc[:cutnum]
    # or else if nothing above sf then ignore cutnum and sort by correlation
    if signals_of_interest.empty==True:
        print(f"Warning: Default filtering produced an empty dataset. Reverting to lowest scores of the original dataset.\n")
        signals_of_interest = dfx.sort_values(by='corrs',ascending=True).reset_index(drop=True).iloc[:num_backup_plots]
    return signals_of_interest

# optional, skip hits that are outside input MJD bounds
def limit_by_MJD(df,before,after):
    # print(before,after)
    # tot=len(df)
    for index, row in df.iterrows():
        # print(index,tot)
        MJD_nums="_".join((row["dat_name"].split("/")[-1]).split("_")[1:3]) # MJD in number of secs
        if before: 
            before_float=float(".".join(before[0].split("_")))
            MJD_float=float(".".join(MJD_nums.split("_"))[:len(before[0])])
            if MJD_float >= before_float:
                df=df.drop(index=index)
                continue
        if after:
            after_float=float(".".join(after[0].split("_")))
            MJD_float=float(".".join(MJD_nums.split("_"))[:len(after[0])])
            if MJD_float <= after_float:
                df=df.drop(index=index)
                continue
    df=df.reset_index(drop=True)
    if len(df)==0:
        print(f"\n\tNo hits found within specified time window.")
        sys.exit()
    return df

# optional, skip hits that are outside input frequency bounds
def limit_by_frange(df,fbelow,fabove):
    # print(fbelow,fabove)
    print(f"{len(df)} hits between {min(df.Corrected_Frequency)} and {max(df.Corrected_Frequency)} MHz.")
    for index, row in df.iterrows():
        if fbelow: 
            if row['Corrected_Frequency'] >= fbelow[0]:
                df=df.drop(index=index)
                continue
        if fabove:
            if row['Corrected_Frequency'] <= fabove[0]:
                df=df.drop(index=index)
                continue
    df=df.reset_index(drop=True)
    if len(df)==0:
        print(f"\n\tNo hits found within specified frequency window.")
        sys.exit()
    return df


    # Main program execution
def main(cmd_args):
    print("\nExecuting program...")
    start=time.time()

    # parse any command line arguments
    # cmd_args = parse_args()
    csv = cmd_args["csv"][0]            # required input
    outdir = cmd_args["outdir"]         # optional, default to datdir listed in csv
    column = cmd_args["col"]            # optional, default None
    operator = cmd_args["op"]           # optional, default None
    value = cmd_args["val"]             # optional, default None
    stack = cmd_args["stack"]           # optional, default None
    zoom = cmd_args["zoom"]             # optional, default None
    nbeams = cmd_args["nbeams"]         # optional, default = 2
    tbeam = cmd_args["tbeam"]           # optional, default = 0
    sf = cmd_args["sf"]                 # optional, default = 4.0
    num_backup_plots = cmd_args["num_backup_plots"] #optional, default = 1
    cutnum = cmd_args["cutoff"]         # optional, default = 500
    clobber = cmd_args["clobber"]       # optional, flag on or default off
    pdf = cmd_args["pdf"]               # optional, flag on or default off
    freqs = cmd_args["freqs"]           # optional, custom frequency span
    before = cmd_args["before"]         # optional, MJD to limit observation plots
    after = cmd_args["after"]           # optional, MJD to limit observation plots
    fbelow = cmd_args["fbelow"]         # optional, MJD to limit observation plots
    fabove = cmd_args["fabove"]           # optional, MJD to limit observation plots
    paths_cleared=[]                    # clobber counter

    # set the path where the plots will be saved (optionally remove old plots with clobber)
    if not outdir:
        path="/".join(csv.split("/")[:-1])+f'/{csv.split("/")[-1].split(".")[0]}_plots/'
    else:
        path=outdir
    if not os.path.exists(path):
        os.mkdir(path)
    if clobber and path not in paths_cleared:
        if pdf:
            ext='*.pdf'
        else:
            ext='*.png'
        purge(path,ext)
        paths_cleared.append(path)
        
    # load the csv into a dataframe and filter based on the input parameters
    df = pd.read_csv(csv)
    
    # if stack flagged, determine the common observing directory for additional integrations.
    # computed outside of loop to save time
    if stack:
        obs_dir=ptu.get_obs_dir(sorted(set(df.fil_0000)))
    
    # plotting mode 1: circumvent regular plotting in the case of bespoke frequency bounds 
    if freqs:
        obs_dir=ptu.get_obs_dir(sorted(set(df.fil_0000)))
        if zoom:
            fmid=(max(freqs)+min(freqs))/2
            fspan=(max(freqs)-min(freqs))*zoom
            freqs=[round(fmid+fpsan,6),round(fmid-fspan,6)]
        if not column or not operator or not value:
            if before or after: # optional, limit hits within MJD range
                df=limit_by_MJD(df,before,after)
            num_plots = ptu.plot_by_freqs(df,obs_dir,freqs,stack,nbeams,tbeam,path=path,pdf=pdf)
        else:
            signals_of_interest = filter_df(df,column,operator,value)
            if before or after: # optional, limit hits within MJD range
                df=limit_by_MJD(df,before,after)
            num_plots = ptu.plot_by_freqs(signals_of_interest,obs_dir,freqs,stack,nbeams,tbeam,path=path,pdf=pdf)
        end, time_label = get_elapsed_time(start)
        print(f"\n\t{num_plots} hits plotted in %.2f {time_label}.\n" %end)
        # This is where the program ends for this plotting mode.
        return None
    # plotting mode 2: default plotting
    elif not column or not operator or not value:
        signals_of_interest = default_filter(df,sf,cutnum, num_backup_plots)
    # plotting mode 3: custom dataframe filtering from input arguments
    else:
        signals_of_interest = filter_df(df,column,operator,value)

    if before or after: # optional, limit hits within MJD range
        signals_of_interest=limit_by_MJD(signals_of_interest,before,after)

    if fbelow or fabove: # optional, limit hits within frequency range
        signals_of_interest=limit_by_frange(signals_of_interest,fbelow,fabove)

    signals_of_interest=signals_of_interest.sort_values(by=['dat_name','Corrected_Frequency'])
    print(f"{len(signals_of_interest)} hits selected out of {len(df)} total from the input dataframe.")

    # iterate through the csv of interesting hits and plot row by row
    for index, row in signals_of_interest.reset_index(drop=True).iterrows():
        # make an array of the filenames for each beam based on the column names
        fil_names = [row[i] for i in list(signals_of_interest) if i.startswith('fil_')]
        
        # determine frequency range for plot
        fstart = row['freq_start']
        fend = row['freq_end']

        # determine frequency span over which to plot based on drift rate
        target_fil=fil_names[tbeam]
        fil_meta = bl.Waterfall(target_fil,load_data=False)
        minimum_frequency = fil_meta.container.f_start
        maximum_frequency = fil_meta.container.f_stop
        tsamp = fil_meta.header['tsamp']    # time bin length in seconds
        obs_length = fil_meta.n_ints_in_file * tsamp # total length of observation in seconds
        DR = row['Drift_Rate']              # reported drift rate
        padding=1+np.log10(row['SNR'])/10   # padding based on reported strength of signal
        MJD_dec=fil_meta.header['tstart']   # MJD in decimal form
        MJD_nums="_".join(os.path.basename(target_fil).split("_")[1:3]) # MJD in number of secs
        # calculate the amount of frequency drift with some padding
        half_span=abs(DR)*obs_length*padding  
        if zoom:
            half_span*=zoom
        if half_span<250:
            half_span=250 # minimum 500 Hz span window
        fmid = row['Corrected_Frequency']
        # fstart=round(max(fmid-half_span*1e-6,minimum_frequency),6)
        # fend=round(min(fmid+half_span*1e-6,maximum_frequency),6)
        
        # grab the other integrations to stack as subplots, if stack flagged on
        if stack:
            # add the relevant fil files to the array from the other integrations
            fil_names = ptu.get_stacks(fil_names,obs_dir,nbeams,stack) 
            if len(fil_names)/nbeams%1==0:
                nstacks=int(len(fil_names)/nbeams) # get number of subplot rows
            else: # assert a subplot for every row and column
                print("\n\tERROR: Number of subplots not evenly divisible by number of beams.")
                print("\tPlease check inputs and retry.\n")
                sys.exit()
            # the fil array is sorted in chronological order from get_stacks so...
            target=fil_names.index(target_fil) # identify which fil contains the target signal in the array
        else:
            nstacks=1   # one row of plots if not stacking
            target=0
        fstart=round(max(fmid-half_span*1e-6*nstacks,minimum_frequency),6)
        fend=round(min(fmid+half_span*1e-6*nstacks,maximum_frequency),6)

        # plot
        print(f'Plotting {index+1}/{len(signals_of_interest)} with central frequency {fmid:.6f} MHz,'+
                f' and SNR ratio: {row["SNR_ratio"]:.3f}{[f", in {nstacks} rows of subplots" if nstacks>1 else ""][0]}')
        ptu.plot_beams(fil_names,
                    fstart,
                    fend,
                    drift_rate=row['Drift_Rate'],
                    nstacks=nstacks,
                    nbeams=nbeams,
                    MJD=MJD_nums,
                    target=target,
                    SNR=row['SNR'],
                    corrs=row['corrs'],
                    SNRr=row['SNR_ratio'],
                    path=path,
                    pdf=pdf)

    # Print the elapsed time of the entire program.
    end, time_label = get_elapsed_time(start)
    print(f"\n\t{len(signals_of_interest)} hits plotted in %.2f {time_label}.\n" %end)
    return None
# run it!
if __name__ == "__main__":
    cmd_args = parse_args()
    main(cmd_args)
