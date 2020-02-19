#!/usr/bin/env bash

deg=$1
cc=$2

tmpfile_deg=$(mktemp /tmp/deg.XXXXXX)
tmpfile_cc=$(mktemp /tmp/cc.XXXXXX)

cat ${deg} | tr ' ' '\n' > $tmpfile_deg
cat ${cc} | tr ' ' '\n' > $tmpfile_cc

norma=$(cat $tmpfile_deg | tr '.' ',' | ~/miniconda3/bin/datamash sum 1)
absolute=$(paste $tmpfile_deg $tmpfile_cc | awk '{print $1*$2}' | tr '.' ',' | ~/miniconda3/bin/datamash sum 1)
echo $absolute $norma | awk '{print $1/$2}'
