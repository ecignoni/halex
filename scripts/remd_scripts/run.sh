#!/bin/bash

#PBS -N ipi_styrene
#PBS -q long_sky
#PBS -l nodes=1:ppn=24
#PBS -o ipi_test.out
#PBS -e ipi_test.err


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

# change to folder where the script is launched
cd $PBS_O_WORKDIR

cat $PBS_NODEFILE > pbs_node

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
i-pi input.xml >> $LogFile & sleep 1; for idx in `seq 0 $(($NReplicas - 1))` ; do cd dftb_${idx}; dftb+ > /dev/null & sleep 1; cd ..; done; wait

log_date FINISHED $LogFile
