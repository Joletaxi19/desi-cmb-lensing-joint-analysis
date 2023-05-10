#!/usr/bin/env python3
#
# Extract the imaging data from Rongpu's photometry files and
# select target types and randoms.
# Use the combination to generate a density map
# and a (binary) mask.
#
import numpy  as np
import healpy as hp
import glob
import sys
from   astropy.table import Table


from assign_randoms_weights import get_randoms_weights


from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc= comm.Get_size()







def toMag(depth,ext,ebv):
    """A 'safe' conversion of depth to magnitude."""
    dd = np.sqrt( depth.clip(1e-30,1e30) )
    mag= -2.5*(np.log10(5/dd)-9) - ext*ebv
    return(mag)
    #




if __name__=="__main__":
    if len(sys.argv)==1:
        # Set a default sample.
        isamp = 1
    elif len(sys.argv)==2:
        # Use the argument as isamp.
        isamp = int(sys.argv[1])
    else:
        print("Usage: "+sys.argv[0]+" [isamp]")
        comm.Barrier()
        comm.Abort(1)
    # Set up the pz_bin based on isamp.
    pz_bin = isamp%10
    # Write a log file, to get around annoying buffering issues
    # at NERSC.
    if rank==0:
        flog = open("make_lrg_maps_s{:02d}.log".format(isamp),"w")
        flog.write("Running "+sys.argv[0]+" on "+str(nproc)+" ranks.\n")
        flog.write("Generating sample "+str(isamp)+"\n")
        flog.write("Setting pz_bin to "+str(pz_bin)+"\n")
    #
    # We will make the maps with a different resolution and ordering than
    # HPXPIXEL that is already provided (typically NSIDE=64, NEST=True).
    # We will convert from celestial to galactic coordinates.
    # Typically we make the systematics maps with lower nside than we
    # want the final maps to be, so put a flag for writing systmaps.
    nside = 2048
    npix  = 12*nside**2
    isnest= False
    if rank==0:
        flog.write("Will write {:d} pixels (Nside={:d}).\n".format(npix,nside))
        flog.write("Format isnest="+str(isnest)+"\n\n")
    ##rot = hp.rotator.Rotator(coord=['C','G'])
    #
    # Set up the data release and version info we'll use.
    #
    release = 'dr9'
    version = '1.0.0'
    #
    # Cuts on extinction and stellar density to be applied throughout.
    ebv_cut,star_cut = 0.15,2500.
    #
    # There is some useful information in the "pixweight" files.
    # The pixweight file is in "Celestial" coordinates and in
    # the Nest format at Nside=256.
    db  = '/global/cfs/cdirs/desi/target/catalogs/'
    db += release+'/'+version+'/pixweight/main/resolve/dark/'
    fn  = db+'pixweight-1-dark.fits'
    pxw_nside = 256
    pxw_nest  = True
    # Want to only read this once if possible.
    if rank==0:
        flog.write("Pix-weight file: "+fn+"\n")
        pxw = Table.read(fn)
    else:
        pxw = None
    pxw = comm.bcast(pxw,root=0)
    # Downgrade the stellar density map to remove the many holes
    # and islands that exist at Nside=256.
    stars_nside = 64
    stars = hp.ud_grade(pxw['STARDENS'],stars_nside,\
                        order_in='NEST',order_out='NEST')
    #
    # Start with the data.
    #
    db = '/global/cfs/cdirs/desi/users/rongpu/data/lrg_xcorr/catalogs/'
    fn = 'dr9_lrg_pzbins_20230509.fits'
    if rank==0:
        flog.write("Cutting on E(B-V)<{:f}\n".format(ebv_cut))
        flog.write("Cutting on STARDENS<{:f}\n".format(star_cut))
        flog.write("Star map ud_graded to Nside={:d}\n\n".format(stars_nside))
        flog.write("Data file path: "+db+"\n")
        flist = glob.glob(db+fn)
        flist = sorted(flist)
        flog.write("Will read {:d} data files...\n".format(len(flist)))
        flog.flush()
    else:
        flist = None
    flist = comm.bcast(flist,root=0)
    #
    dmap = np.zeros(npix,dtype='f8')
    for i,fn in enumerate(flist):
      if i%nproc==rank:
        tt = Table.read(fn)
        # First select the sample.
        if isamp==0:  # isamp==0 is the whole sample
            tt = tt[ (tt['pz_bin']>0)&(tt['pz_bin']<5) ]
        else:
            tt = tt[ tt['pz_bin']==pz_bin ]
        tt = tt[ tt['lrg_mask']==0 ]
        # Restrict to the "observed" area.
        tt = tt[(tt['PIXEL_NOBS_G']>1)&(tt['PIXEL_NOBS_R']>1)&\
                (tt['PIXEL_NOBS_Z']>1)]
        # Cut out regions of "high" extinction.
        tt = tt[(tt['EBV']<ebv_cut)]
        # Apply any other cuts we want here, e.g. on stellar density.
        theta,phi = np.radians(90-tt['DEC']),np.radians(tt['RA'])
        pixnum    = hp.ang2pix(stars_nside,theta,phi,nest=pxw_nest)
        sdens     = stars[pixnum]
        tt        = tt[sdens<star_cut]
        # generate the partial map:
        if len(tt)>0:
            theta,phi = np.radians(90-tt['DEC']),np.radians(tt['RA'])
            ##theta,phi = rot(theta,phi) # C->G coordinates
            pixnum    = hp.ang2pix(nside,theta,phi,nest=isnest)
            tmp, _    = np.histogram(pixnum,bins=np.arange(npix+1)-0.5)
            dmap     += tmp
    dtot = np.zeros(npix,dtype='f8')
    comm.Reduce(dmap,dtot,op=MPI.SUM,root=0)
    dmap = dtot
    # Print some summary statistics.
    if rank==0:
        msk = np.nonzero(dmap>0)[0]
        flog.write("Read and assigned {:e} galaxies.\n".format(np.sum(dmap)))
        flog.write("Have {:d} non-empty pixels, covering {:f} sq.deg.\n".\
           format(len(msk),len(msk)*hp.nside2pixarea(nside,True)))
        flog.write("Done with galaxy files.\n\n")
        flog.flush()
    #
    # Now select randoms.
    #
    version = '0.49.0'
    #
    mb  = '/global/cfs/cdirs/desi/users/rongpu/desi_mask/randoms/lrgmask_v1.1/'
    rb  = '/global/cfs/cdirs/desi/target/catalogs/'
    rb += release+'/'+version+'/randoms/resolve/'
    if rank==0:
        flog.write("Mask   file path: "+mb+"\n")
        flog.write("Random file path: "+rb+"\n")
        flist = glob.glob(rb+r'randoms-[0-9]-[0-9].fits')
        flist+= glob.glob(rb+r'randoms-[0-9]-1[0-9].fits')
        flist = sorted(flist)
        if len(flist)>nproc:
            flist = flist[:nproc]   # Otherwise it takes too long!
        flog.write("Will read {:d} rand files...\n".format(len(flist)))
        flog.flush()
    else:
        flist = None
    flist = comm.bcast(flist,root=0)
    # It is useful to know the randoms per sq. deg., which is
    # stored in the FITS header.
    if rank==0:
        tt = Table.read(flist[0])
        ran_dens = tt.meta['DENSITY']
        flog.write("Random density "+str(ran_dens)+' per sq.deg. [per file].\n')
        flog.flush()
    #
    wmap = np.zeros(npix,dtype='f8')
    rmap = np.zeros(npix,dtype='f8')
    for i,fn in enumerate(flist):
      if i%nproc==rank:
        tt = Table.read(fn)
        mf = mb + fn[len(rb):-5] + '-lrgmask_v1.1.fits.gz'
        mm = Table.read(mf)
        tt['lrg_mask']=mm['lrg_mask']
        # Restrict to the "observed" area.
        tt = tt[(tt['NOBS_G']>1)&(tt['NOBS_R']>1)&(tt['NOBS_Z']>1)]
        # Cut out regions of "high" extinction.
        tt = tt[(tt['EBV']<ebv_cut)]
        # and select LRG objects.
        tt = tt[ tt['lrg_mask']==0 ]
        # Apply any other cuts we want here, e.g. on stellar density.
        theta,phi = np.radians(90-tt['DEC']),np.radians(tt['RA'])
        pixnum    = hp.ang2pix(stars_nside,theta,phi,nest=pxw_nest)
        sdens     = stars[pixnum]
        tt        = tt[sdens<star_cut]
        # Throw away the little islands below the NGC.
        msk = (tt['RA']>135)&(tt['RA']<170)&(tt['DEC']<-10.5)&(tt['DEC']>-31)
        msk|= (tt['RA']>215)&(tt['RA']<225)&(tt['DEC']<-10.5)&(tt['DEC']>-20)
        tt  = tt[~msk]
        # Random weights want these fields.
        tt['galdepth_gmag_ebv'] = toMag(tt['GALDEPTH_G' ],3.214,tt['EBV'])
        tt['galdepth_rmag_ebv'] = toMag(tt['GALDEPTH_R' ],2.165,tt['EBV'])
        tt['galdepth_zmag_ebv'] = toMag(tt['GALDEPTH_Z' ],1.211,tt['EBV'])
        tt['psfdepth_w1mag_ebv']= toMag(tt['PSFDEPTH_W1'],0.184,tt['EBV'])
        tt['psfdepth_w2mag_ebv']= toMag(tt['PSFDEPTH_W2'],0.113,tt['EBV'])
        # Now weight the randoms (or not).
        if True:
            weights_path = '/global/cfs/cdirs/desi/users/rongpu/data'
            weights_path+= '/lrg_xcorr/imaging_weights/main_lrg'
            weights_path+= '/main_lrg_linear_coeffs_pz.yaml'
            wt = get_randoms_weights(tt,weights_path,pz_bin)
            if i==0: # which should be rank 0.
                flog.write("Using weights from "+\
                           weights_path+"\n")
        else:
            wt = np.ones(len(tt))
            if i==0: flog.write("Ignoring weights\n")
        #
        # Now bin the randoms into the map, here we convert coordinate
        # systems.
        theta,phi = np.radians(90-tt['DEC']),np.radians(tt['RA'])
        ##theta,phi = rot(theta,phi) # C->G coordinates
        pixnum    = hp.ang2pix(nside,theta,phi,nest=isnest)
        # Produce weighted and unweighted maps.
        tmp, _ = np.histogram(pixnum,weights=wt,bins=np.arange(npix+1)-0.5)
        rmap  += tmp
        tmp, _ = np.histogram(pixnum,bins=np.arange(npix+1)-0.5)
        wmap  += tmp
    rtot = np.zeros(npix,dtype='f8')
    comm.Reduce(rmap,rtot,op=MPI.SUM,root=0)
    rmap = rtot
    wtot = np.zeros(npix,dtype='f8')
    comm.Reduce(wmap,wtot,op=MPI.SUM,root=0)
    wmap = wtot
    wmap = rmap/(wmap+1e-30)
    #
    if rank==0:
        flog.write("Done with random files.\n\n")
        #
        # Compute the average number of randoms per pixel.
        msk  = np.nonzero(rmap>0)[0]
        avg  = np.mean(rmap[msk])
        #
        # Set the mask to be regions with >=1/5 of the average,
        # i.e. this is an "inclusion mask".
        # We allow relatively large corrections to make sure we don't
        # have lots of small holes that give ringing.
        msk  = np.nonzero(rmap>0.20*avg)[0]
        #
        print("Have {:e} total galaxies in masked region.".\
               format(np.sum(dmap[msk])))
        print("Have {:8.2f} galaxies per masked pixel.".\
                format(np.mean(dmap[msk])))
        print("Have {:8.2f} randoms  per masked pixel.".\
                format(np.mean(wtot[msk])))
        #
        flog.write("Have {:e} total galaxies in masked region.\n".\
               format(np.sum(dmap[msk])))
        flog.write("Have {:8.2f} galaxies per masked pixel.\n".\
               format(np.mean(dmap[msk])))
        flog.write("Have {:8.2f} randoms  per masked pixel.\n".\
               format(np.mean(wtot[msk])))
        # Now fill in the masked region.
        omap      = np.zeros(npix,dtype='f8')
        omap[msk] = dmap[msk]/rmap[msk]
        omap[msk] = omap[msk]/np.mean(omap[msk]) - 1
        omap      = omap.astype('f4')   # Don't need full precision.
        mask      = np.zeros(npix,dtype='f4')
        mask[msk] = 1.0
        # Print some useful numbers.
        shot = np.sum(dmap[msk]/wmap[msk])**2/np.sum(dmap[msk]/wmap[msk]**2)
        shot = np.sum(mask)*hp.nside2pixarea(nside,False)/shot
        ninv = np.sum(mask)*hp.nside2pixarea(nside,False)/np.sum(dmap[msk])
        nbar = 1.0/ninv * (np.pi/180.)**2 # Per sq.deg.
        print("Mean of omap is {:e}.".format(np.mean(omap)))
        print("nbar         is {:f}/deg2".format(nbar))
        print("1/nbar       is {:e}".format(ninv))
        print("Shot noise   is {:e}".format(shot))
        print("Sky fraction is {:f}.".format(np.sum(mask)/mask.size))
        #
        flog.write("Mean of omap is {:e}.\n".format(np.mean(omap)))
        flog.write("nbar         is {:f}/deg2\n".format(nbar))
        flog.write("1/nbar       is {:e}\n".format(ninv))
        flog.write("Shot noise   is {:e}\n".format(shot))
        flog.write("Sky fraction is {:f}\n".format(np.sum(mask)/mask.size))
        flog.close()
        # Write the basic maps we always want.  Do this at multiple
        # resolutions.
        for outnside in [256,2048]:
            pref = "lrg_s{:02d}".format(isamp)
            hpex = ".hpx{:04d}.fits".format(outnside)
            nmap = hp.ud_grade(omap,outnside)
            hp.write_map(pref+"_del"+hpex,nmap,dtype='f4',\
                     nest=isnest,coord='G',overwrite=True)
            nmap = hp.ud_grade(mask,outnside)
            hp.write_map(pref+"_msk"+hpex,nmap,dtype='f4',\
                     nest=isnest,coord='G',overwrite=True)
    #
