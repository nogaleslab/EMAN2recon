#!/usr/bin/env python
# Modified Richard J. Hall 07/2010 rjhall@berkeley.edu
# Author: Pawel A.Penczek, 09/09/2006 (Pawel.A.Penczek@uth.tmc.edu)
# Copyright (c) 2000-2006 The University of Texas - Houston Medical School
#
# This software is issued under a joint BSD/GNU license. You may use the
# source code in this file under either license. However, note that the
# complete EMAN2 and SPARX software packages have some GPL dependencies,
# so you are responsible for compliance with the licenses of these packages
# if you opt to use BSD licensing. The warranty disclaimer below holds
# in either instance.
#
# This complete copyright notice must be included in any revised version of the
# source code. Additional authorship citations may be added, but existing
# author citations must be preserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  2111-1307 USA
#
#


import os
import global_def
from global_def import *
from optparse import OptionParser
import sys
#import cProfile
#import lsprofcalltree


def main():
        arglist = []
        for arg in sys.argv:
                arglist.append( arg )
        progname = os.path.basename(arglist[0])
        usage = progname + " stack ref_vol outdir <maskfile> --ir=inner_radius --ou=outer_radius --rs=ring_step --xr=x_range --yr=y_range  --ts=translational_search_step  --delta=angular_step --an=angular_neighborhood --center=center_type --maxit=max_iter --term=pixel_error --CTF --snr=SNR  --ref_a=S --sym=c1 --multimodel --sort --cutoff --pix_cutoff --model_jump --recon_start --full_output --compare_repro --compare_ref_free=fin --ref_free_cutoff --debug --recon_pad --MPI"
        parser = OptionParser(usage,version=SPARXVERSION)
        parser.add_option("--ir",       type= "int",   default= 1,                  help="inner radius for rotational correlation > 0 (set to 1)")
        parser.add_option("--ou",       type= "int",   default= -1,                 help="outer radius for rotational correlation < int(nx/2)-1 (set to the radius of the particle)")
        parser.add_option("--rs",       type= "int",   default= 1,                  help="step between rings in rotational correlation >0  (set to 1)" )
        parser.add_option("--xr",       type="string", default= "4 2 1 1 1",        help="range for translation search in x direction, search is +/xr")
        parser.add_option("--yr",       type="string", default= "-1",               help="range for translation search in y direction, search is +/yr (default = same as xr)")
        parser.add_option("--ts",       type="string", default= "1 1 1 0.5 0.25",   help="step size of the translation search in both directions, search is -xr, -xr+ts, 0, xr-ts, xr, can be fractional")
        parser.add_option("--delta",    type="string", default= "10 6 4 3 2",       help="angular step of reference projections")
        parser.add_option("--an",       type="string", default= "-1",               help="angular neighborhood for local searches")
        parser.add_option("--center",   type="float",  default= 0,                  help="0: average shift method; 0: no centering; 1: center of gravity (default=0)")
        parser.add_option("--maxit",    type="float",  default= 5,                  help="maximum number of iterations performed for each angular step (set to 5) ")
	parser.add_option("--term",     type="float",  default= 95,                 help="When 95% of images has < 1 pixel error, move to next angular step")
        parser.add_option("--CTF",      action="store_true", default=False,         help="Consider CTF correction during the alignment ")
	parser.add_option("--fourvar",  action="store_true", default=False,         help="compute Fourier variance")
        parser.add_option("--snr",      type="float",  default= 1.0,                help="Signal-to-Noise Ratio of the data")
        parser.add_option("--ref_a",    type="string", default= "S",                help="method for generating the quasi-uniformly distributed projection directions (default S)")
        parser.add_option("--sym",      type="string", default= "c1",               help="symmetry of the refined structure")
	parser.add_option("--sort",      action="store_true", default=False,        help="if true images can only be included in one model")
	parser.add_option("--cutoff",   type="float",        default=999.99,        help="image to include in reconstruction given as sigma value")
	parser.add_option("--pix_cutoff", type="string",    default="0",    help="discount particles by pixel error (1=True, 0=False)")
	parser.add_option("--two_tail", action="store_true", default=False,	    help="two tail cutoff, discount images above and below sigma value")
	parser.add_option("--model_jump",type="string", default="1 1 1 1 1",        help="allow images to select between models for specific round 1 (true)")
	parser.add_option("--restart",   action="store_true", default=False,        help="restart a refinement, uses header values to generate first model")
	parser.add_option("--full_output", action="store_true", default=False,      help="full output of all alignment and angle parameters at each stage")
	parser.add_option("--compare_repro", action="store_true", default=False,    help="output reprojections of the current model, together with average of particles in that view")
	parser.add_option("--compare_ref_free", type="string", default="-1",        help="compare reprojections of the model with previously calculated reference free class averages")
	parser.add_option("--ref_free_cutoff", type="string", default="-1 -1 -1 -1 -1",  help="cutoff based on reference free class averages given as a fraction of 3D resolution")
	parser.add_option("--recon_pad", type="float", default=4,                     help="changed padding for fourier interporlation, should be left at default unless you know what you are doing")
	parser.add_option("--save_half", action="store_true", default=False,        help="save the even/odd maps")

	## options for microtubule helical processing
	parser.add_option("--protos",    type="string", help="if microtubule, protofilament number of initial model(s)")
	parser.add_option("--oplane",   type="float",   help="out of plane angle limiting the range of projections")
	parser.add_option("--lmask",    type="float",   default= 240,               help="length of mask to apply along helix in Angstroms (default=240)")
	parser.add_option("--ilmask",   type="float",   default= 50,                help="radius of inner mask to apply along helix in Angstroms (default=50)")
	parser.add_option("--findseam",        action="store_true", default=False,         help="use for reconstructions that have a seam")
	parser.add_option("--vertstep", type="float",   help="vertical step for vertical symmetry (in Angstroms)")
	parser.add_option("--hpars",    type="string",  default= "-1",              help="twist rise for each volume (separate by spaces, not commas)")
	parser.add_option("--hsearch",  type="string",  default= "73.0 170.0",      help="inner & outer radii for helical search (in Angstroms, default=73.0 170.0)")
	parser.add_option("--wcmask",  type="string",   help="specifies an additional cylinder mask, specified as x, y, and cylinder radius in Angstroms")
        parser.add_option("--MPI",      action="store_true", default=False,         help="whether to use MPI version")
        parser.add_option("--debug",    action="store_true", default=False,         help="debug")
        (options, args) = parser.parse_args(arglist[1:])
        if len(args) < 3 or len(args) > 4:
                print "usage: " + usage
                print "Please run '" + progname + " -h' for detailed options"
        else:
                if len(args) == 3 :
                        mask = None
                else:
                        mask = args[3]
                if options.MPI:
                        from mpi import mpi_init
                        sys.argv = mpi_init(len(sys.argv), sys.argv)

                if global_def.CACHE_DISABLE:
                        from utilities import disable_bdb_cache
                        disable_bdb_cache()

                else:
                        from functions import ali3d
                        global_def.BATCH = True
                        ali3d(args[0], args[1], args[2], mask, options.ir, options.ou, options.rs, options.xr,
                        options.yr, options.ts, options.delta, options.an,
                        options.center, options.maxit, options.term, options.CTF, options.fourvar, options.snr, options.ref_a, options.sym, 
			options.sort, options.cutoff, options.pix_cutoff,  options.two_tail, options.model_jump, options.restart, options.save_half,
			options.protos, options.oplane, options.lmask, options.ilmask, options.findseam, options.vertstep, options.hpars, options.hsearch,
                        options.full_output, options.compare_repro, options.compare_ref_free, options.ref_free_cutoff,
			options.wcmask, options.debug, options.recon_pad, options.MPI)
                        global_def.BATCH = False
		if options.MPI:
			from mpi import mpi_finalize
			mpi_finalize()

if __name__ == "__main__":
#	from mpi import mpi_init, mpi_comm_rank, MPI_COMM_WORLD, mpi_finalize
#        sys.argv = mpi_init(len(sys.argv), sys.argv)
#        myid = mpi_comm_rank(MPI_COMM_WORLD)
#        fout = "trace_%02d.kgrind"%myid
#        p = cProfile.Profile()
#        p.run('main(sys.argv)')
#        k = lsprofcalltree.KCacheGrind(p)
#        data = open(fout, 'w+')
#        k.output(data)
#        data.close()
#        mpi_finalize()

        main()





