#!/bin/bash -l
for sample in 1 2 3 4
do
	wget https://data.desi.lbl.gov/public/papers/c3/lrg_xcorr_2023/v1/maps/main_lrg/lrg_s0${sample}_del.hpx2048.fits.gz
	wget https://data.desi.lbl.gov/public/papers/c3/lrg_xcorr_2023/v1/maps/main_lrg/lrg_s0${sample}_msk.hpx2048.fits.gz
    mv lrg_s0${sample}_del.hpx2048.fits.gz lrg_s0${sample}_del.hpx2048.public.fits.gz
    mv lrg_s0${sample}_msk.hpx2048.fits.gz masks/lrg_s0${sample}_msk.hpx2048.public.fits.gz
    
    wget https://data.desi.lbl.gov/public/papers/c3/lrg_xcorr_2023/v1/maps/extended_lrg/lrg_s0${sample}_del.hpx2048.fits.gz
	wget https://data.desi.lbl.gov/public/papers/c3/lrg_xcorr_2023/v1/maps/extended_lrg/lrg_s0${sample}_msk.hpx2048.fits.gz
    mv lrg_s0${sample}_del.hpx2048.fits.gz lrg_s0${sample}_del.hpx2048.extended.fits.gz
    mv lrg_s0${sample}_msk.hpx2048.fits.gz masks/lrg_s0${sample}_msk.hpx2048.extended.fits.gz
done