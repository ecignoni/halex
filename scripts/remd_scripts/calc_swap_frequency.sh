#!/bin/bash

if [ ${#@} -ne 1 ] ; then
    echo "ERROR: no arguments provided."
    echo "./calc_swap_frequency.sh IPI_LOGFILE"
    exit 1
fi

ipi_log=$1

get_total_swaps() {
    logfile=$1

    tot_swaps=`grep "@ PT:  SWAP" $logfile | wc -l`
}

get_swapped() {
    logfile=$1

    swapped=`grep "@ PT:  SWAPPING" $logfile | wc -l`
}

get_rejected() {
    logfile=$1

    rejected=`grep "@ PT:  SWAP REJECTED" $logfile | wc -l`
}


get_total_swaps $ipi_log
get_swapped $ipi_log
get_rejected $ipi_log

echo "# SWAP attempts        $tot_swaps"
echo "# SWAP accepted        $swapped"
echo "# SWAP rejected        $rejected"

ratio=`awk -v tot=$tot_swaps -v acc=$swapped 'BEGIN{print (acc/tot)*100}'`

echo "SWAP accept ratio      $ratio %"
