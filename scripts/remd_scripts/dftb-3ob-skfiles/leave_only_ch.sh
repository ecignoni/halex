#!/bin/bash

get_elements() {
    local skfile=$1

    OLD_IFS=$IFS

    # strip extension
    skfile="${skfile%.*}"

    IFS="-"
    read -a elements <<< "$skfile"

    IFS=$OLD_IFS
}



for file in `ls *.skf` ; do
    get_elements $file

    # C-C
    if [ "${elements[0]}" == C ] && [ ${elements[1]} == C ] ; then
        :

    # H-H
    elif [ "${elements[0]}" == H ] && [ ${elements[1]} == H ] ; then
        :

    # C-H
    elif [ "${elements[0]}" == C ] && [ ${elements[1]} == H ] ; then
        :

    # H-C
    elif [ "${elements[0]}" == H ] && [ ${elements[1]} == C ] ; then
        :

    else
        rm $file
    fi
done
