#!/bin/bash

#PBS -N ipi_styrene
#PBS -q long_sky
#PBS -l nodes=1:ppn=28
#PBS -o ipi_run.out
#PBS -e ipi_run.err


log_date() {
    msg=$1
    logfile=$2

    date=`date +"%m/%d/%Y %H:%M"`
    echo "@SIMULATION $msg AT: $date" >> $logfile
}


# source conda
source /home/e.cignoni/software/miniconda3/etc/profile.d/conda.sh

# environment with DFTB+ and i-PI
conda activate dftbp

cat $PBS_NODEFILE > ${PBS_O_WORKDIR}/pbs_node

# copy the folder locally on the node and work there
workdir="/local/e.cignoni/ipi/styrene"
origdir="$PBS_O_WORKDIR"

# create local folder if not present
if [ ! -d $workdir ] ; then
    mkdir -p $workdir
fi

# copy content locally on the node
rsync -a $origdir/* $workdir/

# go local on the node
cd $workdir

# For DFTB+, avoid multiple instances of DFTB+ to
# overlap
export OMP_NUM_THREADS=2
NReplicas=16
LogFile=ipi_driver.log

# Create a separate folder for each dftb+ instance
for idx in `seq 0 $(($NReplicas - 1))` ; do
    if [ ! -d dftb_${idx} ] ; then
        mkdir dftb_${idx}
    fi
    cp dftb_in.hsd dftb_${idx}
    cp init.gen dftb_${idx}
done


echo "" > $LogFile

log_date STARTED $LogFile

# run the simulation in parallel 
# (multiple instances of DFTB+)
i-pi input.xml >> $LogFile & sleep 1; for idx in `seq 0 $(($NReplicas - 1))` ; do cd dftb_${idx}; dftb+ > dftb.log & sleep 1; cd ..; done; wait

log_date FINISHED $LogFile

# copy back the simulation
rsync -a $workdir/* $origdir/

# if you cannot copy back the results, do not remove the local folder and report the problem
if [ $? -ne 0 ] ; then
    echo "ERROR: Error while copying back files from node" >> $LogFile
    echo "ERROR: Not deleting the local folder." >> $LogFile
else
    rm -r $workdir
fi

exit 0
