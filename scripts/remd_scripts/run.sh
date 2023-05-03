#!/bin/bash

#PBS -N ipi_styrene
#PBS -q long_sky
#PBS -l nodes=1:ppn=28
#PBS -o ipi_run.out
#PBS -e ipi_run.err


#: Options
# For DFTB+, avoid multiple instances of DFTB+ to
# overlap by limiting the number of omp threads
export OMP_NUM_THREADS=2
WorkDir="/local/e.cignoni/ipi/styrene"
NReplicas=16
LogFile=ipi_driver.log
OrigDir="$PBS_O_WORKDIR"


#: Functions
log_date() {
    msg=$1
    logfile=$2

    date=`date +"%m/%d/%Y %H:%M"`
    echo "@SIMULATION $msg AT: $date" >> $logfile
}

create_folder() {
    local folder=$1

    if [ ! -d $folder ] ; then
        mkdir -p $visual
    fi
}

create_dftb_folders_and_copy_input() {
    for idx in `seq 0 $(($NReplicas - 1))` ; do
        if [ ! -d dftb_${idx} ] ; then
            mkdir dftb_${idx}
        fi
        cp dftb_in.hsd dftb_${idx}
        cp init.gen dftb_${idx}
    done
}


delete_folder_if_success() {
    local exitcode=$1
    local folder=$2

    if [ $exitcode -ne 0 ] ; then
        echo "ERROR: Error while copying back files from node" >> $LogFile
        echo "ERROR: Not deleting the local folder." >> $LogFile
    else
        rm -r $folder
    fi
}

#: Run

# source conda
# environment with DFTB+ and i-PI
source /home/e.cignoni/software/miniconda3/etc/profile.d/conda.sh
conda activate dftbp

# where are we running
cat $PBS_NODEFILE > ${OrigDir}/pbs_node

# create local folder if not present
crate_folder $WorkDir

# copy content locally on the node
rsync -a $OrigDir/* $WorkDir/

# go local on the node
cd $WorkDir

# Create a separate folder for each dftb+ instance
create_dftb_folders_and_copy_input

# initialize log
echo "" > $LogFile

log_date STARTED $LogFile

# run the simulation in parallel 
# (multiple instances of DFTB+)
i-pi input.xml >> $LogFile & sleep 1; for idx in `seq 0 $(($NReplicas - 1))` ; do cd dftb_${idx}; dftb+ > dftb.log & sleep 1; cd ..; done; wait

log_date FINISHED $LogFile

# copy back the simulation
rsync -a $WorkDir/* $OrigDir/

# if you cannot copy back the results, do not remove the local folder and report the problem
delete_folder_if_success $? $WorkDir

exit 0
