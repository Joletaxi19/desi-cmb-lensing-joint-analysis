#!/bin/bash -l
for sample in 1 2 3 4
do
	wget https://data.desi.lbl.gov/public/papers/c3/lrg_xcorr_2023/v1/maps/main_lrg/lrg_s0${sample}_del.hpx2048.fits.gz
	wget https://data.desi.lbl.gov/public/papers/c3/lrg_xcorr_2023/v1/maps/main_lrg/lrg_s0${sample}_msk.hpx2048.fits.gz
    gzip -d lrg_s0${sample}_del.hpx2048.fits.gz
    gzip -d lrg_s0${sample}_msk.hpx2048.fits.gz
    mv lrg_s0${sample}_del.hpx2048.fits lrg_s0${sample}_del.hpx2048.public.fits
    mv lrg_s0${sample}_msk.hpx2048.fits masks/lrg_s0${sample}_msk.hpx2048.public.fits
done