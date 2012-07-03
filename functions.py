# Author: Pawel A.Penczek, 09/09/2006 (Pawel.A.Penczek@uth.tmc.edu)
# Modified: Nogales lab

from EMAN2_cppwrap import *
from global_def import *
import sys
import types


def ali3d(stack, ref_vol, outdir, maskfile = None, ir = 1, ou = -1, rs = 1, 
	    xr = "4 2 2 1", yr = "-1", ts = "1 1 0.5 0.25", delta = "10 6 4 4", an = "-1", 
	    center = 0, maxit = 5, term = 95, CTF = False, fourvar = False, snr = 1.0,  ref_a = "S", sym = "c1", 
	    sort=True, cutoff=999.99, pix_cutoff="0", two_tail=False, model_jump = "1 1 1 1 1", restart=False, save_half=False,
	    protos=None, oplane=None, lmask=-1, ilmask=-1, findseam=False, vertstep=None, hpars="-1", hsearch="73.0 170.0",
	    full_output = False, compare_repro = False, compare_ref_free = "-1" ,ref_free_cutoff = "-1 -1 -1 -1",
	    wcmask = None, debug = False, recon_pad = 4, MPI = False):
	if MPI:
		ali3d_MPI(stack, ref_vol, outdir, maskfile, ir, ou, rs, xr, yr, ts,
	       		delta, an, center, maxit,term, CTF, fourvar, snr, ref_a, sym, 
			sort, cutoff, pix_cutoff, two_tail, model_jump,restart, save_half,
			protos, oplane, lmask, ilmask, findseam, vertstep, hpars, hsearch, 
			full_output, compare_repro, compare_ref_free, ref_free_cutoff,
			wcmask, debug, recon_pad)
		return
		print_end_msg("ali3d")

def ali3d_MPI(stack, ref_vol, outdir, maskfile = None, ir = 1, ou = -1, rs = 1, 
	    xr = "4 2 2 1", yr = "-1", ts = "1 1 0.5 0.25", delta = "10 6 4 4", an = "-1",
	    center = 0, maxit = 5, term = 95, CTF = False, fourvar = False, snr = 1.0,  ref_a = "S", sym = "c1", 
	    sort=True, cutoff=999.99, pix_cutoff="0", two_tail=False, model_jump="1 1 1 1 1", restart=False, save_half=False,
	    protos=None, oplane=None, lmask=-1, ilmask=-1, findseam=False, vertstep=None, hpars="-1", hsearch="73.0 170.0",
	    full_output = False, compare_repro = False, compare_ref_free = "-1", ref_free_cutoff= "-1 -1 -1 -1",
	    wcmask = None, debug = False, recon_pad = 4):

	from alignment      import Numrinit, prepare_refrings
	from utilities      import model_circle, get_image, drop_image, get_input_from_string
	from utilities      import bcast_list_to_all, bcast_number_to_all, reduce_EMData_to_root, bcast_EMData_to_all 
	from utilities      import send_attr_dict
	from utilities      import get_params_proj, file_type
	from fundamentals   import rot_avg_image
	import os
	import types
	from utilities      import print_begin_msg, print_end_msg, print_msg
	from mpi	    import mpi_bcast, mpi_comm_size, mpi_comm_rank, MPI_FLOAT, MPI_COMM_WORLD, mpi_barrier, mpi_reduce
	from mpi	    import mpi_reduce, MPI_INT, MPI_SUM, mpi_finalize
	from filter	 import filt_ctf
	from projection     import prep_vol, prgs
	from statistics     import hist_list, varf3d_MPI, fsc_mask
	from numpy	  import array, bincount, array2string, ones

	number_of_proc = mpi_comm_size(MPI_COMM_WORLD)
	myid	   = mpi_comm_rank(MPI_COMM_WORLD)
	main_node = 0
	if myid == main_node:
		if os.path.exists(outdir):  ERROR('Output directory exists, please change the name and restart the program', "ali3d_MPI", 1)
		os.mkdir(outdir)
	mpi_barrier(MPI_COMM_WORLD)

	if debug:
		from time import sleep
		while not os.path.exists(outdir):
			print  "Node ",myid,"  waiting..."
			sleep(5)

		info_file = os.path.join(outdir, "progress%04d"%myid)
		finfo = open(info_file, 'w')
	else:
		finfo = None
	mjump = get_input_from_string(model_jump)
	xrng	= get_input_from_string(xr)
	if  yr == "-1":  yrng = xrng
	else	  :  yrng = get_input_from_string(yr)
	step	= get_input_from_string(ts)
	delta       = get_input_from_string(delta)
	ref_free_cutoff = get_input_from_string(ref_free_cutoff)	
	pix_cutoff = get_input_from_string(pix_cutoff)
	
	lstp = min(len(xrng), len(yrng), len(step), len(delta))
	if an == "-1":
		an = [-1] * lstp
	else:
		an = get_input_from_string(an)
	# make sure pix_cutoff is set for all iterations
	if len(pix_cutoff)<lstp:
		for i in xrange(len(pix_cutoff),lstp):
			pix_cutoff.append(pix_cutoff[-1])
	# don't waste time on sub-pixel alignment for low-resolution ang incr
	for i in range(len(step)):
		if (delta[i] > 4 or delta[i] == -1) and step[i] < 1:
			step[i] = 1

	first_ring  = int(ir)
	rstep       = int(rs)
	last_ring   = int(ou)
	max_iter    = int(maxit)
	center      = int(center)

	nrefs   = EMUtil.get_image_count( ref_vol )
	nmasks = 0
	if maskfile:
		# read number of masks within each maskfile (mc)
		nmasks   = EMUtil.get_image_count( maskfile )
		# open masks within maskfile (mc)
		maskF   = EMData.read_images(maskfile, xrange(nmasks))
	vol     = EMData.read_images(ref_vol, xrange(nrefs))
	nx      = vol[0].get_xsize()

	## make sure box sizes are the same
	if myid == main_node:
		im=EMData.read_images(stack,[0])
		bx = im[0].get_xsize()
		if bx!=nx:
			print_msg("Error: Stack box size (%i) differs from initial model (%i)\n"%(bx,nx))
			sys.exit()
		del im,bx
	
	# for helical processing:
	helicalrecon = False
	if protos is not None or hpars != "-1" or findseam is True:
		helicalrecon = True
		# if no out-of-plane param set, use 5 degrees
		if oplane is None:
			oplane=5.0
	if protos is not None:
		proto = get_input_from_string(protos)
		if len(proto) != nrefs:
			print_msg("Error: insufficient protofilament numbers supplied")
			sys.exit()
	if hpars != "-1":
		hpars = get_input_from_string(hpars)
		if len(hpars) != 2*nrefs:
			print_msg("Error: insufficient helical parameters supplied")
			sys.exit()
	## create helical parameter file for helical reconstruction
	if helicalrecon is True and myid == main_node:
		from hfunctions import createHpar
		# create initial helical parameter files
		dp=[0]*nrefs
		dphi=[0]*nrefs
		vdp=[0]*nrefs
		vdphi=[0]*nrefs
		for iref in xrange(nrefs):
			hpar = os.path.join(outdir,"hpar%02d.spi"%(iref))
			params = False
			if hpars != "-1":
				# if helical parameters explicitly given, set twist & rise
				params = [float(hpars[iref*2]),float(hpars[(iref*2)+1])]
			dp[iref],dphi[iref],vdp[iref],vdphi[iref] = createHpar(hpar,proto[iref],params,vertstep)

	# get values for helical search parameters
	hsearch = get_input_from_string(hsearch)
	if len(hsearch) != 2:
		print_msg("Error: specify outer and inner radii for helical search")
		sys.exit()

	if last_ring < 0 or last_ring > int(nx/2)-2 :	last_ring = int(nx/2) - 2

	if myid == main_node:
	#	import user_functions
	#	user_func = user_functions.factory[user_func_name]

		print_begin_msg("ali3d_MPI")
		print_msg("Input stack		 : %s\n"%(stack))
		print_msg("Reference volume	    : %s\n"%(ref_vol))	
		print_msg("Output directory	    : %s\n"%(outdir))
		if nmasks > 0:
			print_msg("Maskfile (number of masks)  : %s (%i)\n"%(maskfile,nmasks))
		print_msg("Inner radius		: %i\n"%(first_ring))
		print_msg("Outer radius		: %i\n"%(last_ring))
		print_msg("Ring step		   : %i\n"%(rstep))
		print_msg("X search range	      : %s\n"%(xrng))
		print_msg("Y search range	      : %s\n"%(yrng))
		print_msg("Translational step	  : %s\n"%(step))
		print_msg("Angular step		: %s\n"%(delta))
		print_msg("Angular search range	: %s\n"%(an))
		print_msg("Maximum iteration	   : %i\n"%(max_iter))
		print_msg("Center type		 : %i\n"%(center))
		print_msg("CTF correction	      : %s\n"%(CTF))
		print_msg("Signal-to-Noise Ratio       : %f\n"%(snr))
		print_msg("Reference projection method : %s\n"%(ref_a))
		print_msg("Symmetry group	      : %s\n"%(sym))
		print_msg("Fourier padding for 3D      : %i\n"%(recon_pad))
		print_msg("Number of reference models  : %i\n"%(nrefs))
		print_msg("Sort images between models  : %s\n"%(sort))
		print_msg("Allow images to jump	: %s\n"%(mjump))
		print_msg("CC cutoff standard dev      : %f\n"%(cutoff))
		print_msg("Two tail cutoff	     : %s\n"%(two_tail))
		print_msg("Termination pix error       : %f\n"%(term))
		print_msg("Pixel error cutoff	  : %s\n"%(pix_cutoff))
		print_msg("Restart		     : %s\n"%(restart))
		print_msg("Full output		 : %s\n"%(full_output))
		print_msg("Compare reprojections       : %s\n"%(compare_repro))
		print_msg("Compare ref free class avgs : %s\n"%(compare_ref_free))
		print_msg("Use cutoff from ref free    : %s\n"%(ref_free_cutoff))
		if protos:
			print_msg("Protofilament numbers	: %s\n"%(proto))
			print_msg("Using helical search range   : %s\n"%hsearch) 
		if findseam is True:
			print_msg("Using seam-based reconstruction\n")
		if hpars != "-1":
			print_msg("Using hpars		  : %s\n"%hpars)
		if vertstep != None:
			print_msg("Using vertical step    : %.2f\n"%vertstep)
		if save_half is True:
			print_msg("Saving even/odd halves\n")
		for i in xrange(100) : print_msg("*")
		print_msg("\n\n")
	if maskfile:
		if type(maskfile) is types.StringType: mask3D = get_image(maskfile)
		else:				  mask3D = maskfile
	else: mask3D = model_circle(last_ring, nx, nx, nx)

	numr	= Numrinit(first_ring, last_ring, rstep, "F")
	mask2D  = model_circle(last_ring,nx,nx) - model_circle(first_ring,nx,nx)

	fscmask = model_circle(last_ring,nx,nx,nx)
	if CTF:
		from filter	 import filt_ctf
	from reconstruction_rjh import rec3D_MPI_noCTF

	if myid == main_node:
		active = EMUtil.get_all_attributes(stack, 'active')
		list_of_particles = []
		for im in xrange(len(active)):
			if active[im]:  list_of_particles.append(im)
		del active
		nima = len(list_of_particles)
	else:
		nima = 0
	total_nima = bcast_number_to_all(nima, source_node = main_node)

	if myid != main_node:
		list_of_particles = [-1]*total_nima
	list_of_particles = bcast_list_to_all(list_of_particles, source_node = main_node)

	image_start, image_end = MPI_start_end(total_nima, number_of_proc, myid)

	# create a list of images for each node
	list_of_particles = list_of_particles[image_start: image_end]
	nima = len(list_of_particles)
	if debug:
		finfo.write("image_start, image_end: %d %d\n" %(image_start, image_end))
		finfo.flush()

	data = EMData.read_images(stack, list_of_particles)

	t_zero = Transform({"type":"spider","phi":0,"theta":0,"psi":0,"tx":0,"ty":0})
	transmulti = [[t_zero for i in xrange(nrefs)] for j in xrange(nima)]

	for iref,im in ((iref,im) for iref in xrange(nrefs) for im in xrange(nima)):
		if nrefs == 1:
			transmulti[im][iref] = data[im].get_attr("xform.projection")
		else:
			# if multi models, keep track of eulers for all models
			try:
				transmulti[im][iref] = data[im].get_attr("eulers_txty.%i"%iref)
			except:
				data[im].set_attr("eulers_txty.%i"%iref,t_zero)

	scoremulti = [[0.0 for i in xrange(nrefs)] for j in xrange(nima)] 
	pixelmulti = [[0.0 for i in xrange(nrefs)] for j in xrange(nima)] 
	ref_res = [0.0 for x in xrange(nrefs)] 
	apix = data[0].get_attr('apix_x')

	# for oplane parameter, create cylindrical mask
	if oplane is not None and myid == main_node:
		from hfunctions import createCylMask
		cmaskf=os.path.join(outdir, "mask3D_cyl.mrc")
		mask3D = createCylMask(data,ou,lmask,ilmask,cmaskf)
		# if finding seam of helix, create wedge masks
		if findseam is True:
			wedgemask=[]
			for pf in xrange(nrefs):
				wedgemask.append(EMData())
			# wedgemask option
			if wcmask is not None:
				wcmask = get_input_from_string(wcmask)
				if len(wcmask) != 3:
					print_msg("Error: wcmask option requires 3 values: x y radius")
					sys.exit()

	# determine if particles have helix info:
	try:
		data[0].get_attr('h_angle')
		original_data = []
		boxmask = True
		from hfunctions import createBoxMask
	except:
		boxmask = False

	# prepare particles
	for im in xrange(nima):
		data[im].set_attr('ID', list_of_particles[im])
		data[im].set_attr('pix_score', int(0))
		if CTF:
			# only phaseflip particles, not full CTF correction
			ctf_params = data[im].get_attr("ctf")
			st = Util.infomask(data[im], mask2D, False)
			data[im] -= st[0]
			data[im] = filt_ctf(data[im], ctf_params, sign = -1, binary=1)
			data[im].set_attr('ctf_applied', 1)
		# for window mask:
		if boxmask is True:
			h_angle = data[im].get_attr("h_angle")
			original_data.append(data[im].copy())
			bmask = createBoxMask(nx,apix,ou,lmask,h_angle)
			data[im]*=bmask
			del bmask
	if debug:
		finfo.write( '%d loaded  \n' % nima )
		finfo.flush()
	if myid == main_node:
		# initialize data for the reference preparation function
		ref_data = [ mask3D, max(center,0), None, None, None, None ]
		# for method -1, switch off centering in user function

	from time import time	

	#  this is needed for gathering of pixel errors
	disps = []
	recvcount = []
	disps_score = []
	recvcount_score = []
	for im in xrange(number_of_proc):
		if( im == main_node ):  
			disps.append(0)
			disps_score.append(0)
		else:		  
			disps.append(disps[im-1] + recvcount[im-1])
			disps_score.append(disps_score[im-1] + recvcount_score[im-1])
		ib, ie = MPI_start_end(total_nima, number_of_proc, im)
		recvcount.append( ie - ib )
		recvcount_score.append((ie-ib)*nrefs)

	pixer = [0.0]*nima
	cs = [0.0]*3
	total_iter = 0
	volodd = EMData.read_images(ref_vol, xrange(nrefs))
	voleve = EMData.read_images(ref_vol, xrange(nrefs))

	if restart:
		# recreate initial volumes from alignments stored in header
		itout = "000_00"
		for iref in xrange(nrefs):
			if(nrefs == 1):
				modout = ""
			else:
				modout = "_model_%02d"%(iref)	
	
			if(sort): 
				group = iref
				for im in xrange(nima):
					imgroup = data[im].get_attr('group')
					if imgroup == iref:
						data[im].set_attr('xform.projection',transmulti[im][iref])
			else: 
				group = int(999) 
				for im in xrange(nima):
					data[im].set_attr('xform.projection',transmulti[im][iref])
			
			fscfile = os.path.join(outdir, "fsc_%s%s"%(itout,modout))

			vol[iref], fscc, volodd[iref], voleve[iref] = rec3D_MPI_noCTF(data, sym, fscmask, fscfile, myid, main_node, index = group, npad = recon_pad)

			if myid == main_node:
				if helicalrecon:
					from hfunctions import processHelicalVol

					vstep=None
					if vertstep is not None:
						vstep=(vdp[iref],vdphi[iref])
					print_msg("Old rise and twist for model %i     : %8.3f, %8.3f\n"%(iref,dp[iref],dphi[iref]))
					hvals=processHelicalVol(vol[iref],voleve[iref],volodd[iref],iref,outdir,itout,
								dp[iref],dphi[iref],apix,hsearch,findseam,vstep,wcmask)
					(vol[iref],voleve[iref],volodd[iref],dp[iref],dphi[iref],vdp[iref],vdphi[iref])=hvals
					print_msg("New rise and twist for model %i     : %8.3f, %8.3f\n"%(iref,dp[iref],dphi[iref]))
					# get new FSC from symmetrized half volumes
					fscc = fsc_mask( volodd[iref], voleve[iref], mask3D, rstep, fscfile)
				else:
					vol[iref].write_image(os.path.join(outdir, "vol_%s.hdf"%itout),-1)

				if save_half is True:
					volodd[iref].write_image(os.path.join(outdir, "volodd_%s.hdf"%itout),-1)
					voleve[iref].write_image(os.path.join(outdir, "voleve_%s.hdf"%itout),-1)

				if nmasks > 1:
					# Read mask for multiplying
					ref_data[0] = maskF[iref]
				ref_data[2] = vol[iref]
				ref_data[3] = fscc
				#  call user-supplied function to prepare reference image, i.e., center and filter it
				vol[iref], cs,fl = ref_ali3d(ref_data)
				vol[iref].write_image(os.path.join(outdir, "volf_%s.hdf"%(itout)),-1)
				if (apix == 1):
					res_msg = "Models filtered at spatial frequency of:\t"
					res = fl
				else:
					res_msg = "Models filtered at resolution of:       \t"
					res = apix / fl	
				ares = array2string(array(res), precision = 2)
				print_msg("%s%s\n\n"%(res_msg,ares))	
			
			bcast_EMData_to_all(vol[iref], myid, main_node)
			# write out headers, under MPI writing has to be done sequentially
			mpi_barrier(MPI_COMM_WORLD)

	# projection matching	
	for N_step in xrange(lstp):
		terminate = 0
		Iter = -1
 		while(Iter < max_iter-1 and terminate == 0):
			Iter += 1
			total_iter += 1
			itout = "%03g_%02d" %(delta[N_step], Iter)
			if myid == main_node:
				print_msg("ITERATION #%3d, inner iteration #%3d\nDelta = %4.1f, an = %5.2f, xrange = %5.2f, yrange = %5.2f, step = %5.2f\n\n"%(N_step, Iter, delta[N_step], an[N_step], xrng[N_step],yrng[N_step],step[N_step]))
	
			for iref in xrange(nrefs):
				if myid == main_node: start_time = time()
				volft,kb = prep_vol( vol[iref] )

				## constrain projections to out of plane parameter
				theta1 = None
				theta2 = None
				if oplane is not None:
					theta1 = 90-oplane
					theta2 = 90+oplane
				refrings = prepare_refrings( volft, kb, nx, delta[N_step], ref_a, sym, numr, MPI=True, phiEqpsi = "Minus", initial_theta=theta1, delta_theta=theta2)
				
				del volft,kb

				if myid== main_node:
					print_msg( "Time to prepare projections for model %i: %s\n" % (iref, legibleTime(time()-start_time)) )
					start_time = time()
	
				for im in xrange( nima ):
					data[im].set_attr("xform.projection", transmulti[im][iref])
					if an[N_step] == -1:
						t1, peak, pixer[im] = proj_ali_incore(data[im],refrings,numr,xrng[N_step],yrng[N_step],step[N_step],finfo)
					else:
						t1, peak, pixer[im] = proj_ali_incore_local(data[im],refrings,numr,xrng[N_step],yrng[N_step],step[N_step],an[N_step],finfo)
					#data[im].set_attr("xform.projection"%iref, t1)
					if nrefs > 1: data[im].set_attr("eulers_txty.%i"%iref,t1)
					scoremulti[im][iref] = peak
					from pixel_error import max_3D_pixel_error
					# t1 is the current param, t2 is old
					t2 = transmulti[im][iref]
					pixelmulti[im][iref] = max_3D_pixel_error(t1,t2,numr[-3])
					transmulti[im][iref] = t1

				if myid == main_node:
					print_msg("Time of alignment for model %i: %s\n"%(iref, legibleTime(time()-start_time)))
					start_time = time()


			# gather scoring data from all processors
			from mpi import mpi_gatherv
			scoremultisend = sum(scoremulti,[])
			pixelmultisend = sum(pixelmulti,[])
			tmp = mpi_gatherv(scoremultisend,len(scoremultisend),MPI_FLOAT, recvcount_score, disps_score, MPI_FLOAT, main_node,MPI_COMM_WORLD)
			tmp1 = mpi_gatherv(pixelmultisend,len(pixelmultisend),MPI_FLOAT, recvcount_score, disps_score, MPI_FLOAT, main_node,MPI_COMM_WORLD)
			tmp = mpi_bcast(tmp,(total_nima * nrefs), MPI_FLOAT,0, MPI_COMM_WORLD)
			tmp1 = mpi_bcast(tmp1,(total_nima * nrefs), MPI_FLOAT,0, MPI_COMM_WORLD)
			tmp = map(float,tmp)
			tmp1 = map(float,tmp1)
			score = array(tmp).reshape(-1,nrefs)
			pixelerror = array(tmp1).reshape(-1,nrefs) 
			score_local = array(scoremulti)
			mean_score = score.mean(axis=0)
			std_score = score.std(axis=0)
			cut = mean_score - (cutoff * std_score)
			cut2 = mean_score + (cutoff * std_score)
			res_max = score_local.argmax(axis=1)
			minus_cc = [0.0 for x in xrange(nrefs)]
			minus_pix = [0.0 for x in xrange(nrefs)]
			minus_ref = [0.0 for x in xrange(nrefs)]
			
			#output pixel errors
			if(myid == main_node):
				from statistics import hist_list
				lhist = 20
				pixmin = pixelerror.min(axis=1)
				region, histo = hist_list(pixmin, lhist)
				if(region[0] < 0.0):  region[0] = 0.0
				print_msg("Histogram of pixel errors\n      ERROR       number of particles\n")
				for lhx in xrange(lhist):
					print_msg(" %10.3f     %7d\n"%(region[lhx], histo[lhx]))
				# Terminate if 95% within 1 pixel error
				im = 0
				for lhx in xrange(lhist):
					if(region[lhx] > 1.0): break
					im += histo[lhx]
				print_msg( "Percent of particles with pixel error < 1: %f\n\n"% (im/float(total_nima)*100))
				term_cond = float(term)/100
				if(im/float(total_nima) > term_cond): 
					terminate = 1
					print_msg("Terminating internal loop\n")
				del region, histo
			terminate = mpi_bcast(terminate, 1, MPI_INT, 0, MPI_COMM_WORLD)
			terminate = int(terminate[0])	
			
			for im in xrange(nima):
				if(sort==False):
					data[im].set_attr('group',999)
				elif (mjump[N_step]==1):
					data[im].set_attr('group',int(res_max[im]))
				
				pix_run = data[im].get_attr('pix_score')			
				if (pix_cutoff[N_step]==1 and (terminate==1 or Iter == max_iter-1)):
					if (pixelmulti[im][int(res_max[im])] > 1):
						data[im].set_attr('pix_score',int(777))

				if (score_local[im][int(res_max[im])]<cut[int(res_max[im])]) or (two_tail and score_local[im][int(res_max[im])]>cut2[int(res_max[im])]):
					data[im].set_attr('group',int(888))
					minus_cc[int(res_max[im])] = minus_cc[int(res_max[im])] + 1

				if(pix_run == 777):
					data[im].set_attr('group',int(777))
					minus_pix[int(res_max[im])] = minus_pix[int(res_max[im])] + 1

				if (compare_ref_free != "-1") and (ref_free_cutoff[N_step] != -1) and (total_iter > 1):
					id = data[im].get_attr('ID')
					if id in rejects:
						data[im].set_attr('group',int(666))
						minus_ref[int(res_max[im])] = minus_ref[int(res_max[im])] + 1	
						
				
			minus_cc_tot = mpi_reduce(minus_cc,nrefs,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD)	
			minus_pix_tot = mpi_reduce(minus_pix,nrefs,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD) 	
			minus_ref_tot = mpi_reduce(minus_ref,nrefs,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD)
			if (myid == main_node):
				if(sort):
					tot_max = score.argmax(axis=1)
					res = bincount(tot_max)
				else:
					res = ones(nrefs) * total_nima
				print_msg("Particle distribution:	     \t\t%s\n"%(res*1.0))
				afcut1 = res - minus_cc_tot
				afcut2 = afcut1 - minus_pix_tot
				afcut3 = afcut2 - minus_ref_tot
				print_msg("Particle distribution after cc cutoff:\t\t%s\n"%(afcut1))
				print_msg("Particle distribution after pix cutoff:\t\t%s\n"%(afcut2)) 
				print_msg("Particle distribution after ref cutoff:\t\t%s\n\n"%(afcut3)) 
					
						
			res = [0.0 for i in xrange(nrefs)]
			for iref in xrange(nrefs):
				if(center == -1):
					from utilities      import estimate_3D_center_MPI, rotate_3D_shift
					dummy=EMData()
					cs[0], cs[1], cs[2], dummy, dummy = estimate_3D_center_MPI(data, total_nima, myid, number_of_proc, main_node)				
					cs = mpi_bcast(cs, 3, MPI_FLOAT, main_node, MPI_COMM_WORLD)
					cs = [-float(cs[0]), -float(cs[1]), -float(cs[2])]
					rotate_3D_shift(data, cs)


				if(sort): 
					group = iref
					for im in xrange(nima):
						imgroup = data[im].get_attr('group')
						if imgroup == iref:
							data[im].set_attr('xform.projection',transmulti[im][iref])
				else: 
					group = int(999) 
					for im in xrange(nima):
						data[im].set_attr('xform.projection',transmulti[im][iref])
				if(nrefs == 1):
					modout = ""
				else:
					modout = "_model_%02d"%(iref)	
				
				fscfile = os.path.join(outdir, "fsc_%s%s"%(itout,modout))
				vol[iref], fscc, volodd[iref], voleve[iref] = rec3D_MPI_noCTF(data, sym, fscmask, fscfile, myid, main_node, index=group, npad=recon_pad)
	
				if myid == main_node:
					print_msg("3D reconstruction time for model %i: %s\n"%(iref, legibleTime(time()-start_time)))
					start_time = time()
	
				# Compute Fourier variance
				if fourvar:
					outvar = os.path.join(outdir, "volVar_%s.hdf"%(itout))
					ssnr_file = os.path.join(outdir, "ssnr_%s"%(itout))
					varf = varf3d_MPI(data, ssnr_text_file=ssnr_file, mask2D=None, reference_structure=vol[iref], ou=last_ring, rw=1.0, npad=1, CTF=None, sign=1, sym=sym, myid=myid)
					if myid == main_node:
						print_msg("Time to calculate 3D Fourier variance for model %i: %s\n"%(iref, legibleTime(time()-start_time)))
						start_time = time()
						varf = 1.0/varf
						varf.write_image(outvar,-1)
				else:  varf = None

				if myid == main_node:
					if helicalrecon:
						from hfunctions import processHelicalVol

						vstep=None
						if vertstep is not None:
							vstep=(vdp[iref],vdphi[iref])
						print_msg("Old rise and twist for model %i     : %8.3f, %8.3f\n"%(iref,dp[iref],dphi[iref]))
						hvals=processHelicalVol(vol[iref],voleve[iref],volodd[iref],iref,outdir,itout,
									dp[iref],dphi[iref],apix,hsearch,findseam,vstep,wcmask)
						(vol[iref],voleve[iref],volodd[iref],dp[iref],dphi[iref],vdp[iref],vdphi[iref])=hvals
						print_msg("New rise and twist for model %i     : %8.3f, %8.3f\n"%(iref,dp[iref],dphi[iref]))
						# get new FSC from symmetrized half volumes
						fscc = fsc_mask( volodd[iref], voleve[iref], mask3D, rstep, fscfile)

						print_msg("Time to search and apply helical symmetry for model %i: %s\n\n"%(iref, legibleTime(time()-start_time)))
						start_time = time()
					else:
						vol[iref].write_image(os.path.join(outdir, "vol_%s.hdf"%(itout)),-1)

					if save_half is True:
						volodd[iref].write_image(os.path.join(outdir, "volodd_%s.hdf"%(itout)),-1)
						voleve[iref].write_image(os.path.join(outdir, "voleve_%s.hdf"%(itout)),-1)

					if nmasks > 1:
						# Read mask for multiplying
						ref_data[0] = maskF[iref]
					ref_data[2] = vol[iref]
					ref_data[3] = fscc
					ref_data[4] = varf
					#  call user-supplied function to prepare reference image, i.e., center and filter it
					vol[iref], cs,fl = ref_ali3d(ref_data)
					vol[iref].write_image(os.path.join(outdir, "volf_%s.hdf"%(itout)),-1)
					if (apix == 1):
						res_msg = "Models filtered at spatial frequency of:\t"
						res[iref] = fl
					else:
						res_msg = "Models filtered at resolution of:       \t"
						res[iref] = apix / fl	
	
				del varf
				bcast_EMData_to_all(vol[iref], myid, main_node)
				
				if compare_ref_free != "-1": compare_repro = True
				if compare_repro:
					outfile_repro = comp_rep(refrings, data, itout, modout, vol[iref], group, nima, nx, myid, main_node, outdir)
					mpi_barrier(MPI_COMM_WORLD)
					if compare_ref_free != "-1":
						ref_free_output = os.path.join(outdir,"ref_free_%s%s"%(itout,modout))
						rejects = compare(compare_ref_free, outfile_repro,ref_free_output,yrng[N_step], xrng[N_step], rstep,nx,apix,ref_free_cutoff[N_step], number_of_proc, myid, main_node)

			# retrieve alignment params from all processors
			par_str = ['xform.projection','ID','group']
			if nrefs > 1:
				for iref in xrange(nrefs):
					par_str.append('eulers_txty.%i'%iref)

			if myid == main_node:
				from utilities import recv_attr_dict
				recv_attr_dict(main_node, stack, data, par_str, image_start, image_end, number_of_proc)
				
			else:	send_attr_dict(main_node, data, par_str, image_start, image_end)

			if myid == main_node:
				ares = array2string(array(res), precision = 2)
				print_msg("%s%s\n\n"%(res_msg,ares))
				dummy = EMData()
				if full_output:
					nimat = EMUtil.get_image_count(stack)
					output_file = os.path.join(outdir, "paramout_%s"%itout)
					foutput = open(output_file, 'w')
					for im in xrange(nimat):
						# save the parameters for each of the models
						outstring = ""
						dummy.read_image(stack,im,True)
						param3d = dummy.get_attr('xform.projection')
						g = dummy.get_attr("group")
						# retrieve alignments in EMAN-format
						pE = param3d.get_params('eman')
						outstring += "%f\t%f\t%f\t%f\t%f\t%i\n" %(pE["az"], pE["alt"], pE["phi"], pE["tx"], pE["ty"],g)
						foutput.write(outstring)
					foutput.close()
				del dummy
			mpi_barrier(MPI_COMM_WORLD)


#	mpi_finalize()	

	if myid == main_node: print_end_msg("ali3d_MPI")
	
	
def MPI_start_end(nima, nproc, myid):
	image_start = int(round(float(nima)/nproc*myid))
	image_end   = int(round(float(nima)/nproc*(myid+1)))
	return image_start, image_end

def ref_ali3d( ref_data ):
	from utilities      import print_msg
	from filter	 import fit_tanh, filt_tanl
	from fundamentals   import fshift
	from morphology     import threshold

	fl = ref_data[2].cmp("dot",ref_data[2], {"negative":0, "mask":ref_data[0]} )
	cs = [0.0]*3
	stat = Util.infomask(ref_data[2], ref_data[0], False)
	volf = ref_data[2] - stat[0]
	Util.mul_scalar(volf, 1.0/stat[1])
	Util.mul_img(volf, ref_data[0])
	fl, aa = fit_tanh(ref_data[3])
	volf = filt_tanl(volf, fl, aa)
	volf.process_inplace("normalize")
	if ref_data[1] == 1:
		cs = volf.phase_cog()
		volf  = fshift(volf, -cs[0], -cs[1], -cs[2])
	return  volf, cs, fl

def comp_rep(refrings, data, itout, modout, vol, group, nima, nx, myid, main_node, outdir):
	import os
	from fundamentals import rot_shift2D
	from utilities    import get_params_proj, params_3D_2D
	from mpi import mpi_reduce, MPI_COMM_WORLD, MPI_FLOAT, MPI_SUM	
	avg = [EMData() for i in xrange(len(refrings))]
	avg_csum = [0.0 for i in xrange(len(refrings))]
	for i in xrange(len(refrings)):
		avg[i] = EMData()
		avg[i].set_size(nx,nx)
		phi   = refrings[i].get_attr("phi")
		theta = refrings[i].get_attr("theta")
		t = Transform({"type":"spider","phi":phi,"theta":theta,"psi":0.0})
		avg[i].set_attr("xform.projection",t)

	for im in xrange(nima):
		iref = data[im].get_attr("assign")
		gim = data[im].get_attr("group")
		if gim == group:
			[phi, theta, psi, s2x, s2y] = get_params_proj(data[im])
			[alpha, sx,sy,mirror] = params_3D_2D(phi,theta,psi,s2x,s2y)
			temp = rot_shift2D(data[im],alpha, sx, sy, mirror, 1.0)
			avg[iref] = avg[iref] + temp
			avg_csum[iref] = avg_csum[iref] + 1
		from utilities import reduce_EMData_to_root
	for i in xrange(len(refrings)):
	    	reduce_EMData_to_root(avg[i], myid, main_node)
		avg_sum = mpi_reduce(avg_csum[i],1,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD)	
		outfile_repro = os.path.join(outdir, "repro_%s%s.hdf"%(itout,modout))
		if myid ==0:
		     	outfile = os.path.join(outdir, "compare_repro_%s%s.hdf"%(itout,modout))
			avg[i].write_image(outfile,-1)
			t = avg[i].get_attr("xform.projection")
			proj = vol.project("pawel",t)
			proj.set_attr("xform.projection",t)
			proj.set_attr("Raw_im_count", float(avg_sum))
			proj.write_image(outfile,-1)
			proj.write_image(outfile_repro,-1)
	return outfile_repro

def compare(compare_ref_free, outfile_repro,ref_free_output,yrng, xrng, rstep,nx,apix,ref_free_cutoff, nproc, myid, main_node):

	from alignment      import   Numrinit, ringwe,  Applyws
	from random	 import   seed, randint
	from utilities      import   get_params2D, set_params2D, model_circle, inverse_transform2, combine_params2
	from fundamentals   import   rot_shift2D
	from mpi	    import   MPI_COMM_WORLD, mpi_barrier, mpi_bcast, MPI_INT
	from statistics     import   fsc_mask
	from filter	 import   fit_tanh
	from numpy	  import   array	

	fout = "%s.hdf" % ref_free_output
	frc_out = "%s_frc" % ref_free_output
	res_out = "%s_res" % ref_free_output
	
	
	nima = EMUtil.get_image_count(compare_ref_free)
	image_start, image_end = MPI_start_end(nima, nproc, myid)
	ima = EMData()
	ima.read_image(compare_ref_free, image_start)
	
	last_ring = nx/2-2
	first_ring = 1
	mask = model_circle(last_ring, nx, nx)

	refi = []
	numref = EMUtil.get_image_count(outfile_repro)
	cnx = nx/2 +1
	cny = cnx
	
	mode = "F"
	numr = Numrinit(first_ring, last_ring, rstep, mode)	
	wr = ringwe(numr, mode)

	ima.to_zero()
	for j in xrange(numref):
		temp = EMData()
		temp.read_image(outfile_repro, j)
		#  even, odd, numer of even, number of images.  After frc, totav
		refi.append(temp)
	#  for each node read its share of data
	data = EMData.read_images(compare_ref_free, range(image_start, image_end))
	for im in xrange(image_start, image_end):
		data[im-image_start].set_attr('ID', im)
		set_params2D(data[im-image_start],[0,0,0,0,1])
	ringref = []
	for j in xrange(numref):
			refi[j].process_inplace("normalize.mask", {"mask":mask, "no_sigma":1}) # normalize reference images to N(0,1)
			cimage = Util.Polar2Dm(refi[j], cnx, cny, numr, mode)
			Util.Frngs(cimage, numr)
			Applyws(cimage, numr, wr)
			ringref.append(cimage)
	
	if myid == main_node: seed(1000)
	data_shift = []	
	frc = []
	res = []
	for im in xrange(image_start, image_end):
		alpha, sx, sy, mirror, scale = get_params2D(data[im-image_start])
		alphai, sxi, syi, scalei = inverse_transform2(alpha, sx, sy, 1.0)
		# normalize
		data[im-image_start].process_inplace("normalize.mask", {"mask":mask, "no_sigma":1}) # subtract average under the mask
		# align current image to the reference
		[angt, sxst, syst, mirrort, xiref, peakt] = Util.multiref_polar_ali_2d(data[im-image_start], ringref, xrng, yrng, 1, mode, numr, cnx+sxi, cny+syi)
		iref = int(xiref)
		[alphan, sxn, syn, mn] = combine_params2(0.0, -sxi, -syi, 0, angt, sxst, syst, (int)(mirrort))
		set_params2D(data[im-image_start], [alphan, sxn, syn, int(mn), scale])
		temp = rot_shift2D(data[im-image_start], alphan, sxn, syn, mn)
		temp.set_attr('assign',iref)
		tfrc = fsc_mask(temp,refi[iref],mask = mask)
		temp.set_attr('frc',tfrc[1])
		res = fit_tanh(tfrc)
		temp.set_attr('res',res)
		data_shift.append(temp)
	
	for node in xrange(nproc):
		if myid == node:
			for image in data_shift:
				image.write_image(fout,-1)
				refindex = image.get_attr('assign')
				refi[refindex].write_image(fout,-1)	
		mpi_barrier(MPI_COMM_WORLD)
	rejects = []
	if myid == main_node:
		a = EMData()
		index = 0
		frc = []
		res = []
		temp = []
		classes = []
		for im in xrange(nima):
			a.read_image(fout, index)
			frc.append(a.get_attr("frc"))
			if ref_free_cutoff != -1: classes.append(a.get_attr("class_ptcl_idxs"))
			tmp = a.get_attr("res")
			temp.append(tmp[0])
			res.append("%12f" %(apix/tmp[0]))
			res.append("\n")
			index = index + 2
		res_num = array(temp)
		mean_score = res_num.mean(axis=0)
		std_score = res_num.std(axis=0)
		std = std_score / 2
		if ref_free_cutoff !=-1:
			cutoff = mean_score - std * ref_free_cutoff
			reject = res_num < cutoff
			index = 0
			for i in reject:
				if i: rejects.extend(classes[index])
				index = index + 1
			rejects.sort()
			length = mpi_bcast(len(rejects),1,MPI_INT,main_node, MPI_COMM_WORLD)	
			rejects = mpi_bcast(rejects,length , MPI_INT, main_node, MPI_COMM_WORLD)
		del a
		fout_frc = open(frc_out,'w')
		fout_res = open(res_out,'w')
		fout_res.write("".join(res))
		temp = zip(*frc)
		datstrings = []
		for i in temp:
			for j in i:
				datstrings.append("  %12f" % (j))
			datstrings.append("\n")
		fout_frc.write("".join(datstrings))
		fout_frc.close()
	
	del refi		
	del ringref
	return rejects

def proj_ali_incore(data, refrings, numr, xrng, yrng, step, finfo=None):
	from utilities    import compose_transform2

	ID = data.get_attr("ID")
	if finfo:
		from utilities    import get_params_proj
		phi, theta, psi, s2x, s2y = get_params_proj(data)
		finfo.write("Image id: %6d\n"%(ID))
		finfo.write("Old parameters: %9.4f %9.4f %9.4f %9.4f %9.4f\n"%(phi, theta, psi, s2x, s2y))
		finfo.flush()

	mode = "F"
	#  center is in SPIDER convention
	nx   = data.get_xsize()
	ny   = data.get_ysize()
	cnx  = nx//2 + 1
	cny  = ny//2 + 1

	#phi, theta, psi, sxo, syo = get_params_proj(data)
	t1 = data.get_attr("xform.projection")
	dp = t1.get_params("spider")
	# get translations from data
	tx = dp["tx"]
	ty = dp["ty"]

	[ang, sxs, sys, mirror, iref, peak] = Util.multiref_polar_ali_2d(data, refrings, xrng, yrng, step, mode, numr, cnx+tx, cny+ty)
	iref = int(iref)
	data.set_attr("assign",iref)
	#[ang,sxs,sys,mirror,peak,numref] = apmq(projdata[imn], ref_proj_rings, xrng, yrng, step, mode, numr, cnx-sxo, cny-syo)
	#ang = (ang+360.0)%360.0
	# The ormqip returns parameters such that the transformation is applied first, the mirror operation second.
	#  What that means is that one has to change the the Eulerian angles so they point into mirrored direction: phi+180, 180-theta, 180-psi
	angb, sxb, syb, ct = compose_transform2(0.0, sxs, sys, 1, -ang, 0.0, 0.0, 1)
	if mirror:
		phi   = (refrings[iref].get_attr("phi")+540.0)%360.0
		theta = 180.0-refrings[iref].get_attr("theta")
		psi   = (540.0-refrings[iref].get_attr("psi")+angb)%360.0
		s2x   = sxb - tx
		s2y   = syb - ty
	else:
		phi   = refrings[iref].get_attr("phi")
		theta = refrings[iref].get_attr("theta")
		psi   = (refrings[iref].get_attr("psi")+angb+360.0)%360.0
		s2x   = sxb - tx
		s2y   = syb - ty
	#set_params_proj(data, [phi, theta, psi, s2x, s2y])
	t2 = Transform({"type":"spider","phi":phi,"theta":theta,"psi":psi})
	t2.set_trans(Vec2f(-s2x, -s2y))

	from pixel_error import max_3D_pixel_error
	pixel_error = max_3D_pixel_error(t1, t2, numr[-3])

	if finfo:
		finfo.write( "New parameters: %9.4f %9.4f %9.4f %9.4f %9.4f %10.5f  %11.3e\n\n" %(phi, theta, psi, s2x, s2y, peak, pixel_error))
		finfo.flush()

	return t2, peak, pixel_error

def proj_ali_incore_local(data, refrings, numr, xrng, yrng, step, an, finfo=None):
	from utilities    import compose_transform2
	#from utilities    import set_params_proj, get_params_proj
	from math	 import cos, sin, pi

	ID = data.get_attr("ID")

	mode = "F"
	nx   = data.get_xsize()
	ny   = data.get_ysize()
	#  center is in SPIDER convention
	cnx  = nx//2 + 1
	cny  = ny//2 + 1

	ant = cos(an*pi/180.0)
	#phi, theta, psi, sxo, syo = get_params_proj(data)
	t1 = data.get_attr("xform.projection")
	dp = t1.get_params("spider")
	# get translations from data
	tx = dp["tx"]
	ty = dp["ty"]
	
	if finfo:
		finfo.write("Image id: %6d\n"%(ID))
		finfo.write("Old parameters: %9.4f %9.4f %9.4f %9.4f %9.4f\n"%(dp["phi"], dp["theta"], dp["psi"], -tx, -ty))
		finfo.flush()

	[ang, sxs, sys, mirror, iref, peak] = Util.multiref_polar_ali_2d_local(data, refrings, xrng, yrng, step, ant, mode, numr, cnx+tx, cny+ty)
	iref=int(iref)
	data.set_attr("assign",iref)
	if iref > -1:
		# The ormqip returns parameters such that the transformation is applied first, the mirror operation second.
		# What that means is that one has to change the the Eulerian angles so they point into mirrored direction: phi+180, 180-theta, 180-psi
		angb, sxb, syb, ct = compose_transform2(0.0, sxs, sys, 1, -ang, 0.0, 0.0, 1)
		if  mirror:
			phi   = (refrings[iref].get_attr("phi")+540.0)%360.0
			theta = 180.0-refrings[iref].get_attr("theta")
			psi   = (540.0-refrings[iref].get_attr("psi")+angb)%360.0
			s2x   = sxb - tx
			s2y   = syb - ty
		else:
			phi   = refrings[iref].get_attr("phi")
			theta = refrings[iref].get_attr("theta")
			psi   = (refrings[iref].get_attr("psi")+angb+360.0)%360.0
			s2x   = sxb - tx
			s2y   = syb - ty

		t2 = Transform({"type":"spider","phi":phi,"theta":theta,"psi":psi})
		t2.set_trans(Vec2f(-s2x, -s2y))

		from pixel_error import max_3D_pixel_error
		pixel_error = max_3D_pixel_error(t1, t2, numr[-3])
		if finfo:
			finfo.write( "New parameters: %9.4f %9.4f %9.4f %9.4f %9.4f %10.5f  %11.3e\n\n" %(phi, theta, psi, s2x, s2y, peak, pixel_error))
			finfo.flush()
		return t2, peak, pixel_error
	else:
		return -1.0e23, 0.0

#=============================
def legibleTime(sec):
	hours = int(sec/3600)
	sec-= hours*3600
	min = int(sec/60)
	sec -= min*60
	timestr = ""
	if hours > 0: timestr+="%ih,"%hours
	timestr += "%im, %is"%(min,sec)
	return timestr

