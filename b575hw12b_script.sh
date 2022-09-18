#!/bin/bash
echo "#==SCRIPT STARTED"

awk '{if (NR%4==2) print$0}' sample.fastq > sample_rseq.txt
awk '{if (NR%4==2) print$0}' sample_cutadapt.fastq > sample_cutadapt_rseq.txt

echo "#== RETRIEVE READ SEQUENCE DONE"

sed 's/AGATCGGAAGAGC.*//' sample_rseq.txt > sample_rseq_CUT.txt

echo "#== CUT ADAPTER DONE"

diff sample_rseq_CUT.txt sample_cutadapt_rseq.txt

echo "#== SCRIPT DONE"
The last time the script was ran is Sun Sep 18 07:20:50 PM EDT 2022
