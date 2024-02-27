#!/bin/bash -l
wget -O COM_Lensing-SimMap_4096_R3.00.tar "http://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_Lensing-SimMap_4096_R3.00.tar"
tar xopf COM_Lensing-SimMap_4096_R3.00.tar
cd COM_Lensing-SimMap_4096_R3.00
for file in MV_sim_klm_*
do
    tar xopf ${file}
done