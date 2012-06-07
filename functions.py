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
	    full_output = False, compare_repro = False, compare_ref_free = "-1" ,ref_free_cutoff = "-1 -1 -1 -1", debug = False, recon_pad = 4, MPI = False):
	if MPI:
		ali3d_MPI(stack, ref_vol, outdir, maskfile, ir, ou, rs, xr, yr, ts,
	       		delta, an, center, maxit,term, CTF, fourvar, snr, ref_a, sym, 
			sort, cutoff, pix_cutoff, two_tail, model_jump,restart, save_half,
			protos, oplane, lmask, ilmask, findseam, vertstep, hpars, hsearch, 
			full_output, compare_repro, compare_ref_free, ref_free_cutoff, debug, recon_pad)
		return
		print_end_msg("ali3d")

def ali3d_MPI(stack, ref_vol, outdir, maskfile = None, ir = 1, ou = -1, rs = 1, 
	    xr = "4 2 2 1", yr = "-1", ts = "1 1 0.5 0.25", delta = "10 6 4 4", an = "-1",
	    center = 0, maxit = 5, term = 95, CTF = False, fourvar = False, snr = 1.0,  ref_a = "S", sym = "c1", 
	    sort=True, cutoff=999.99, pix_cutoff="0", two_tail=False, model_jump="1 1 1 1 1", restart=False, save_half=False,
	    protos=None, oplane=None, lmask=-1, ilmask=-1, findseam=False, vertstep=None, hpars="-1", hsearch="73.0 170.0",
	    full_output = False, compare_repro = False, compare_ref_free = "-1", ref_free_cutoff= "-1 -1 -1 -1", debug = False, recon_pad = 4):

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
	nmods = 0
	if maskfile:
		# read number of masks within each maskfile (mc)
		nmods   = EMUtil.get_image_count( maskfile )
		# open masks within maskfile (mc)
		maskF   = EMData.read_images(maskfile, xrange(nmods))
	vol     = EMData.read_images(ref_vol, xrange(nrefs))
	nx      = vol[0].get_xsize()
	
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
			dp[iref],dphi[iref],vdp[iref],vdphi[iref] = create_hpar(hpar,proto[iref],params,vertstep)

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
		if nmods > 0:
			print_msg("Maskfile (number of masks)  : %s (%i)\n"%(maskfile,nmods))
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
		from reconstruction_rjh import rec3D_MPI
		from filter	 import filt_ctf
	else:   from reconstruction_rjh import rec3D_MPI_noCTF

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
#	nima_per_ref = total_nima / nrefs
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
	for im in xrange( nima ):
		transmulti[im][0] = data[im].get_attr("xform.projection")
	scoremulti = [[0.0 for i in xrange(nrefs)] for j in xrange(nima)] 
	pixelmulti = [[0.0 for i in xrange(nrefs)] for j in xrange(nima)] 
	ref_res = [0.0 for x in xrange(nrefs)] 
	apix = data[0].get_attr('apix_x')

	# for oplane parameter, create cylindrical mask
	if oplane is not None and myid == main_node:
		cmaskf=os.path.join(outdir, "mask3D_cyl.mrc")
		mask3D = createCylMask(data,ou,lmask,ilmask,cmaskf)
		# if finding seam of helix, create wedge masks
		if findseam is True:
			wedgemask=[]
			for pf in xrange(nrefs):
				wedgemask.append(EMData())

	for im in xrange(nima):
		data[im].set_attr('ID', list_of_particles[im])
		data[im].set_attr('pix_score', int(0))
		if CTF:
			ctf_params = data[im].get_attr("ctf")
			st = Util.infomask(data[im], mask2D, False)
			data[im] -= st[0]
			data[im] = filt_ctf(data[im], ctf_params, sign = -1)
			data[im].set_attr('ctf_applied', 1)

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
		if CTF:  vol[0], fscc, volodd[0], voleve[0] = rec3D_MPI(data, snr, sym, fscmask, os.path.join(outdir, "fsc_000_00"), myid, main_node, index = -1,npad = recon_pad)
		else:    vol[0], fscc, volodd[0], voleve[0] = rec3D_MPI_noCTF(data, sym, fscmask, os.path.join(outdir, "fsc_000_00"), myid, main_node, index = -1, npad = recon_pad)
	
		if myid == main_node:
			vol[0].write_image(os.path.join(outdir, "vol_000_00.hdf"),-1)
			if save_half is True:
				volodd[0].write_image(os.path.join(outdir, "volodd_000_00.hdf"),-1)
				voleve[0].write_image(os.path.join(outdir, "voleve_000_00.hdf"),-1)
			ref_data[2] = vol[0]
			ref_data[3] = fscc
			#  call user-supplied function to prepare reference image, i.e., center and filter it
			vol[0], cs,fl = ref_ali3d(ref_data)
			vol[0].write_image(os.path.join(outdir, "volf_000_00.hdf"),-1)
			if (apix == 1):
				res_msg = "Models filtered at spatial frequency of:\t"
				res = fl
			else:
				res_msg = "Models filtered at resolution of:       \t"
				res = apix / fl	
			ares = array2string(array(res), precision = 2)
			print_msg("%s%s\n\n"%(res_msg,ares))	
			
		bcast_EMData_to_all(vol[0], myid, main_node)
		# write out headers, under MPI writing has to be done sequentially
		mpi_barrier(MPI_COMM_WORLD)

# projection matching	
	for N_step in xrange(lstp):
#		if compare_ref_free == "-1": 
#			ref_free_cutoff[N_step] =-1
#			print ref_free_cutoff
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
				if oplane is not None:
					refrings = prepare_refringsHelical( volft, kb, nx, delta[N_step], ref_a, oplane, sym, numr, True)
				else:
					refrings = prepare_refrings( volft, kb, nx, delta[N_step], ref_a, sym, numr, MPI = True)
				
				del volft,kb
				if myid== main_node:
					print_msg( "Time to prepare projections for model %i: %s\n" % (iref, legibleTime(time()-start_time)) )
					start_time = time()
	
				for im in xrange( nima ):
					if an[N_step] == -1:
						data[im].set_attr("xform.projection", transmulti[im][iref])
						t1, peak, pixer[im] = proj_ali_incore(data[im],refrings,numr,xrng[N_step],yrng[N_step],step[N_step],finfo)
					else:
						data[im].set_attr("xform.projection", transmulti[im][iref])
						t1, peak, pixer[im] = proj_ali_incore_local(data[im],refrings,numr,xrng[N_step],yrng[N_step],step[N_step],an[N_step],finfo)
					data[im].set_attr("xform.projection", t1)
					scoremulti[im][iref] = peak
					from pixel_error import max_3D_pixel_error
					# t1 is the current param
					#t1 = data[im].get_attr("xform.projection")
					t2 = transmulti[im][iref]
					pixelmulti[im][iref] = max_3D_pixel_error(t1,t2,numr[-3])
					transmulti[im][iref] = t1

				if myid == main_node:
					print_msg("Time of alignment for model %i: %s\n"%(iref, legibleTime(time()-start_time)))
					start_time = time()
			
			scoremultisend = sum(scoremulti,[])
			pixelmultisend = sum(pixelmulti,[])
			from mpi import mpi_gatherv
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
			#R		if myid == main_node:
			#R			msg = " Average center x = %10.3f	Center y = %10.3f	Center z = %10.3f\n"%(cs[0], cs[1], cs[2])
			#R			print_msg(msg)
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
				if CTF:
					vol[iref], fscc, volodd[iref], voleve[iref] = rec3D_MPI(data, snr, sym, fscmask, fscfile, myid, main_node, index = group, npad = recon_pad)
				else:   
					vol[iref], fscc, volodd[iref], voleve[iref] = rec3D_MPI_noCTF(data, sym, fscmask, fscfile, myid, main_node, index = group, npad = recon_pad)
	
				if myid == main_node:
					print_msg("3D reconstruction time for model %i: %s\n"%(iref, legibleTime(time()-start_time)))
					start_time = time()
	
				# Compute Fourier variance
				if fourvar:
					outvar = os.path.join(outdir, "volVar_%s.hdf"%(itout))
					ssnr_file = os.path.join(outdir, "ssnr_%s"%(itout))
					varf = varf3d_MPI(data, ssnr_text_file=ssnr_file, mask2D=None, reference_structure=vol[iref], ou=last_ring, rw=1.0, npad=1, CTF=CTF, sign=1, sym=sym, myid=myid)
					if myid == main_node:
						print_msg("Time to calculate 3D Fourier variance for model %i: %s\n"%(iref, legibleTime(time()-start_time)))
						start_time = time()
						varf = 1.0/varf
						varf.write_image(outvar,-1)
				else:  varf = None

				if myid == main_node:
					if helicalrecon:
						# save a copy of the unsymmetrized volume
						nosymf = os.path.join(outdir,"volNoSym_%s.hdf"%(itout))
						vol[iref].write_image(nosymf,-1)
						print_msg("Old rise and twist for model %i     : %8.3f, %8.3f\n"%(iref,dp[iref],dphi[iref]))

						# course helical symmetry
						dp[iref],dphi[iref] = hsearchsym(vol[iref],dp[iref],dphi[iref],apix,hsearch[1],hsearch[0],0.05)
						# fine helical symmetry
						dp[iref],dphi[iref] = hsearchsym(vol[iref],dp[iref],dphi[iref],apix,hsearch[1],hsearch[0],0.01)

						# find vertical symmetry
						if vertstep is not None:
						# course & fine vertical symmetry
							vdp[iref],vdphi[iref] = hsearchsym(vol[iref],vdp[iref],vdphi[iref],apix,hsearch[1],hsearch[0],0.05)
							vdp[iref],vdphi[iref] = hsearchsym(vol[iref],vdp[iref],vdphi[iref],apix,hsearch[1],hsearch[0],0.01)

						# inner & outer radii in pixels 
						if findseam is True:
							vol[iref] = applySeamSym(vol[iref],dp[iref],dphi[iref],apix)
							voleve[iref] = applySeamSym(voleve[iref],dp[iref],dphi[iref],apix)
							volodd[iref] = applySeamSym(volodd[iref],dp[iref],dphi[iref],apix)
							vol[iref].write_image(os.path.join(outdir, "volOverSym_%s.hdf"%(itout)),-1)
							# mask out tubulin & apply sym again for seam
							# have to make a new wedgemask for each iteration
							wedgemask[iref]=createWedgeMask(nx,proto[iref],dp[iref],dphi[iref],apix,hpar,3)
							# recreate the microtubule from seam
							vol[iref] = regenerateFromPF(vol[iref],wedgemask[iref],dp[iref],dphi[iref],apix)
							voleve[iref] = regenerateFromPF(voleve[iref],wedgemask[iref],dp[iref],dphi[iref],apix)
							volodd[iref] = regenerateFromPF(volodd[iref],wedgemask[iref],dp[iref],dphi[iref],apix)
						else:
							vol[iref] = vol[iref].helicise(apix, dp[iref], dphi[iref])
							voleve[iref] = voleve[iref].helicise(apix, dp[iref], dphi[iref])
							volodd[iref] = volodd[iref].helicise(apix, dp[iref], dphi[iref])

						# apply vertical symmetry
						if vertstep is not None:
							vol[iref] = vol[iref].helicise(apix, vdp[iref], vdphi[iref])
							voleve[iref] = voleve[iref].helicise(apix, dp[iref], dphi[iref])
							volodd[iref] = volodd[iref].helicise(apix, dp[iref], dphi[iref])

						print_msg("New rise and twist for model %i     : %8.3f, %8.3f\n"%(iref,dp[iref],dphi[iref]))
						hpar = os.path.join(outdir,"hpar%02d.spi"%(iref))
						f=open(hpar,'a')
						f.write("%.6f\t%.6f"%(dphi[iref],dp[iref]))
						if vertstep is not None:
							f.write("\t%.6f\t%.6f"%(vdphi[iref],vdp[iref]))
						f.write("\n")
						f.close()

						print_msg("Time to search and apply helical symmetry for model %i: %s\n\n"%(iref, legibleTime(time()-start_time)))
						start_time = time()

						volmsk = vol[iref]*mask3D
						volmsk = vol[iref]
						volmsk.write_image(os.path.join(outdir, "vol_%s.hdf"%(itout)),-1)

						# get new FSC from symmetrized half volumes
						fscc = fsc_mask( volodd[iref], voleve[iref], mask3D, rstep, fscfile)

						if save_half is True:
							volh = volodd[iref]*mask3D
							volh.write_image(os.path.join(outdir, "volodd_%s.hdf"%(itout)),-1)
							volh = voleve[iref]*mask3D
							volh.write_image(os.path.join(outdir, "voleve_%s.hdf"%(itout)),-1)
							del volh
					else:
						vol[iref].write_image(os.path.join(outdir, "vol_%s.hdf"%(itout)),-1)
						if save_half is True:
							volodd[iref].write_image(os.path.join(outdir, "volodd_%s.hdf"%(itout)),-1)
							voleve[iref].write_image(os.path.join(outdir, "voleve_%s.hdf"%(itout)),-1)
					if nmods > 1:
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
			par_str = ['xform.projection','ID','group']
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
						dummy.read_image(stack,im,True)
						param3d = dummy.get_attr('xform.projection')
						# retrieve alignments in EMAN-format
						paramEMAN = param3d.get_params('eman')
						g = dummy.get_attr("group")
						outstring = "%f\t%f\t%f\t%f\t%f\t%i\n" %(paramEMAN["az"], paramEMAN["alt"], paramEMAN["phi"], paramEMAN["tx"], paramEMAN["ty"], g)
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
	#data.set_attr("xform.projection", t2)
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
		#finfo.write("Old parameters: %9.4f %9.4f %9.4f %9.4f %9.4f\n"%(phi, theta, psi, sxo, syo))
		finfo.write("Old parameters: %9.4f %9.4f %9.4f %9.4f %9.4f\n"%(dp["phi"], dp["theta"], dp["psi"], -tx, -ty))
		finfo.flush()

	#[ang, sxs, sys, mirror, iref, peak] = Util.multiref_polar_ali_2d_local(data, refrings, xrng, yrng, step, ant, mode, numr, cnx-sxo, cny-syo)
	[ang, sxs, sys, mirror, iref, peak] = Util.multiref_polar_ali_2d_local(data, refrings, xrng, yrng, step, ant, mode, numr, cnx+tx, cny+ty)
	iref=int(iref)
	#[ang,sxs,sys,mirror,peak,numref] = apmq_local(projdata[imn], ref_proj_rings, xrng, yrng, step, ant, mode, numr, cnx-sxo, cny-syo)
	#ang = (ang+360.0)%360.0
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

		#set_params_proj(data, [phi, theta, psi, s2x, s2y])
		t2 = Transform({"type":"spider","phi":phi,"theta":theta,"psi":psi})
		t2.set_trans(Vec2f(-s2x, -s2y))
		#data.set_attr("xform.projection", t2)
		from pixel_error import max_3D_pixel_error
		pixel_error = max_3D_pixel_error(t1, t2, numr[-3])
		if finfo:
			finfo.write( "New parameters: %9.4f %9.4f %9.4f %9.4f %9.4f %10.5f  %11.3e\n\n" %(phi, theta, psi, s2x, s2y, peak, pixel_error))
			finfo.flush()
		return t2, peak, pixel_error
	else:
		return -1.0e23, 0.0

#===========================
def createCylMask(data,rmax,lmask,rmin,outfile=None):
	"""
	create a cylindrical mask with gaussian edges
	"""

	from itertools import product
	import math

	apix = data[0].get_attr('apix_x')
	nx = data[0].get_xsize() 

	## convert mask values to pixels
	lmask = int((lmask/apix)/2)
	rmin = int(abs(rmin)/apix)
	cylRadius = (nx/2)-2
	if rmax == -1:
		rmax = int(240/apix)
	falloff_outer = lmask*0.4
	falloff_inner = rmin*0.4

	## first create cylinder with inner & outer mask
	cyl = EMData(nx,nx,nx)
	for i in range(nx):
		mask=EMData(nx,nx)
		mask.to_one()
		## mask the inner & outer radii
		for x,y in product(range(nx),range(nx)):
			dx = abs(x-nx/2)
			dy = abs(y-nx/2)
			r2 = dx**2+dy**2
			if r2 > rmax*rmax:
				wt1 = 0.5*(1 + math.cos(math.pi*min(1,(math.sqrt(r2)-rmax)/falloff_outer)))
				mask.set(x,y,wt1)
			elif r2 < rmin*rmin:
				wt2 = 0.5*(1 + math.cos(math.pi*min(1,(rmin-math.sqrt(r2))/falloff_inner)))
				mask.set(x,y,wt2)
		## mask along length
		dz = abs(i-nx/2)
		if dz > lmask:
			wt3 = 0.5*(1+math.cos(math.pi*min(1,(dz-lmask)/falloff_outer)))
			mask.mult(wt3)
		cyl.insert_clip(mask,(0,0,i))

	if outfile is not None:
		cyl.write_image(outfile)

	return cyl
	
#===========================
def createWedgeMask(nx,csym,rise,twist,apix,hfile,ovlp):
	"""
	a hard-edged wedge that follows helical symmetry
	"""
	import math
	img = EMData(nx,nx)
	img.to_zero()
	#add ovlp degrees to overlap with the neighboring density
	overlap=ovlp*math.pi/180.0
	alpha = math.pi/2 - math.pi/csym
	for x,y in ((x,y) for x in range(0,nx) for y in range(nx/2,nx)):
		dx = abs(x-nx/2)
		dy = abs(y-nx/2)
		# if above the line y = tan(alpha)*x
		inner = dx*math.tan(alpha)
		outer = dx*math.tan(alpha-overlap)
		if dy >= inner:
			img.set(x,y,1)
		elif dy >= outer:
			pos = (inner-dy)/(inner-outer)
			img.set(x,y,1-pos)

	img.process_inplace("mask.sharp",{"outer_radius":nx/2})

	wedge = EMData(nx,nx,nx)
	alpha = 360+(csym*twist)
	lrise = csym*rise
	rot = alpha/lrise*apix
	for z in range(nx):
		finalrot = ((z-nx/2)*rot)/3
		t=Transform()
		t.set_rotation({"type":"2d","alpha":-finalrot})
		newslice=img.process("xform",{"transform":t})
		wedge.insert_clip(newslice,(0,0,z))
	#wedge *= kinesinMask(nx,int(32/apix),54/apix,143/apix,rot)
	#wedge += kinesinMask(nx,int(30/apix),24/apix,164/apix,rot,pos=True)

	# odd-numbered protofilaments are off by 1/2 twist
	if csym%2==1:
		t = Transform({"type":"spider","psi":twist/2})
		wedge.process_inplace("xform",{"transform":t})

	wedge.process_inplace("threshold.binary",{"value":0.00001})
	#wedge.write_image('wedge_mask_p%d.mrc'%csym)

	return wedge

#===========================
def kinesinMask(nx,rad,cx,cy,rot,pos=False):
	# hard-edged cylinder mask for kinesin position
	img = EMData(nx,nx)
	img.to_one()
	if pos is True:
		img.to_zero()
	for x,y in ((x,y) for x in range(nx) for y in range(nx)):
		dx = abs(x-cx)
		dy = abs(y-cy)
		r2 = dx**2+dy**2
		if r2 < rad*rad:
			if pos is True:
				img.set(nx/2-x,nx/2+y,1)
			else:
				img.set(nx/2+x,nx/2+y,0)
	#img.write_image('test.mrc')
	cylmask = EMData(nx,nx,nx)
	for z in range(nx):
		finalrot=((z-nx/2)*rot)/3
		t=Transform()
		t.set_rotation({"type":"2d","alpha":-finalrot})
		newslice=img.process("xform",{"transform":t})
		cylmask.insert_clip(newslice,(0,0,z))
	return cylmask

#===========================
def prepare_refringsHelical( volft, kb, nx, delta, ref_a, oplane, sym, numr, MPI=False):
	"""
	prepare projections for helical processing
	rotation 180 degrees inplane & specified out-of-plane
	"""
	from alignment import ringwe, Applyws
	from projection   import prgs
	from math	 import sin, cos, pi
	from applications import MPI_start_end
	from utilities      import bcast_list_to_all, bcast_number_to_all, reduce_EMData_to_root, bcast_EMData_to_all 
	import re

	# convert csym to integer:
	sym = int(re.sub("\D", "", sym))
	# generate list of Eulerian angles for reference projections
	#  phi, theta, psi
	mode = "F"
	ref_angles = []
	inplane=int((179.99/sym)/delta) + 1
	# first create 0 and positive out-of-plane tilts
	i = 0
	while i < oplane:
		for j in xrange(inplane):
			t = j*delta
			ref_angles.append([t,90.0+i,90.0])
		i+=delta
	# negative out of plane rotation
	i = -(delta)
	while i > -(oplane):
		for j in xrange(inplane):
			t = j*delta
			ref_angles.append([t,90.0+i,90.0])
		i-=delta
	
	wr_four  = ringwe(numr, mode)
	cnx = nx//2 + 1
	cny = nx//2 + 1
	qv = pi/180.
	num_ref = len(ref_angles)

	if MPI:
		from mpi import mpi_comm_rank, mpi_comm_size, MPI_COMM_WORLD
		myid = mpi_comm_rank( MPI_COMM_WORLD )
		ncpu = mpi_comm_size( MPI_COMM_WORLD )
	else:
		ncpu = 1
		myid = 0
	from applications import MPI_start_end
	ref_start,ref_end = MPI_start_end( num_ref, ncpu, myid )

	refrings = []     # list of (image objects) reference projections in Fourier representation

	sizex = numr[ len(numr)-2 ] + numr[ len(numr)-1 ] - 1

	for i in xrange(num_ref):
		prjref = EMData()
		prjref.set_size(sizex, 1, 1)
		refrings.append(prjref)

	for i in xrange(ref_start, ref_end):
		prjref = prgs(volft, kb, [ref_angles[i][0], ref_angles[i][1], ref_angles[i][2], 0.0, 0.0])
		cimage = Util.Polar2Dm(prjref, cnx, cny, numr, mode)  # currently set to quadratic....
		Util.Normalize_ring(cimage, numr)

		Util.Frngs(cimage, numr)
		Applyws(cimage, numr, wr_four)
		refrings[i] = cimage

	if MPI:
		from utilities import bcast_EMData_to_all
		for i in xrange(num_ref):
			for j in xrange(ncpu):
				ref_start,ref_end = MPI_start_end(num_ref,ncpu,j)
				if i >= ref_start and i < ref_end: rootid = j

			bcast_EMData_to_all( refrings[i], myid, rootid )
	for i in xrange(len(ref_angles)):
		n1 = sin(ref_angles[i][1]*qv)*cos(ref_angles[i][0]*qv)
		n2 = sin(ref_angles[i][1]*qv)*sin(ref_angles[i][0]*qv)
		n3 = cos(ref_angles[i][1]*qv)
		refrings[i].set_attr_dict( {"n1":n1, "n2":n2, "n3":n3} )
		refrings[i].set_attr("phi", ref_angles[i][0])
		refrings[i].set_attr("theta", ref_angles[i][1])
		refrings[i].set_attr("psi", ref_angles[i][2])

	return refrings

#===========================
def create_hpar(hpar,pf,params=False,vertstep=None):
	"""
	create a helical symmetry file for Egelman's helical programs
	file is a spider-formatted text file listing the rise & turn in angstroms
	"""
	if params is False:
		if (pf==11):
			ang = -32.47
			rise = 11.08
		elif (pf==12):
			ang = -29.88
			rise = 10.16
		elif (pf==13):
			ang = -27.69
			rise = 9.39
		elif (pf==14):
			ang = -25.77
			rise = 8.72
		elif (pf==15):
			ang = -23.83
			rise = 10.81
		elif (pf==16):
			ang = -22.4
			rise = 10.18
		else:
			ang = -360.0/pf
			rise = 10.0
	else:
		ang=params[0]
		rise=params[1]
	f=open(hpar,'w')
	f.write("%.6f\t%.6f"%(ang,rise))
	vrise,vang = None, None
	if vertstep is not None:
		vrise,vang = vertstep,-0.1
		f.write("\t%.6f\t%.6f"%(-0.1,vertstep))
	f.write("\n")
	f.close()
	return rise,ang,vrise,vang

#===========================
def hsearchsym(vol,dp,dphi,apix,rmax,rmin,hstep):
	"""
	Use old fortran hsearch_lorentz to find helical symmetry
	Seems to work better than the Sparx version for now
	"""
	import shutil,subprocess,os
	from random import randint

	tmpid = randint(0, 1000000)
	volrot = "volrot%i.spi"%tmpid
	tmphpar = "hpar%i.spi"%tmpid

	# volume must be rotated for hsearch
	t = Transform({"type":"spider","theta":90.0,"psi":90.0})
	volcopy = vol.process("xform",{"transform":t})
	volcopy.write_image(volrot,0,EMUtil.ImageType.IMAGE_SINGLE_SPIDER)

	emancmd = "proc3d %s %s spidersingle"%(volrot,volrot)
	subprocess.Popen(emancmd,shell=True).wait()

	# create hpar file
	f = open(tmphpar,'w')
	f.write("; spi/spi\n")
	f.write("    1 2 %11.5f %11.5f\n"%(dphi,dp))
	f.close()

	hsearchexe = subprocess.Popen("which hsearch_lorentz", shell=True, stdout=subprocess.PIPE).stdout.read().strip()
	if not os.path.isfile(hsearchexe):
		printError("executable 'hsearch_lorentz' not found, make sure it's in your path"%h_exe)
		
	hcmd = "%s %s %s %.4f 73.0 170.0 %.3f %.3f"%(hsearchexe,volrot,tmphpar,apix,hstep,hstep)
	print hcmd
	subprocess.Popen(hcmd,shell=True).wait()

	# get new hsearch parameters
	f = open(tmphpar)
	lines = f.readlines()
	f.close()
	pars = lines[-1].strip().split()

	os.remove(tmphpar)
	os.remove(volrot)

	return float(pars[3]),float(pars[2])

#===========================
def applySeamSym(vol,rise,rot,apix):
	"""
	apply seam symmetry based on results from Egelman search
	"""
	# find protofilament number from rotation
	sym=int(round(360.0/abs(rot)))

	rise/=apix
	# apply protofilament symmetry
	sumvol = vol.copy()
	pfoffset=int(sym/2)
	for pnum in range(-pfoffset,sym-pfoffset):
		if pnum==0: continue
		ang = rot*pnum
		trans = -(rise*pnum)
		t = Transform({"type":"spider","psi":ang})
		t.set_trans(0,0,trans)
		volcopy = vol.process("xform",{"transform":t})
		sumvol.add(volcopy)

	sumvol.process_inplace("normalize")
	return sumvol

#===========================
def regenerateFromPF(vol,wedgemask,rise,rot,apix):
	"""
	mask out one protofilament and regenerate the full microtubule
	"""
	from reconstruction_rjh import smart_add

	# convert rise to pixels
	nx = vol.get_xsize()
	rise/=apix
	sym=int(round(360.0/abs(rot)))

	# apply protofilament symmetry
	sumvol = vol*wedgemask

	pfoffset=int(sym/2)
	for pnum in range(-pfoffset,sym-pfoffset):
		if pnum==0:
			continue
		ang = -(rot*pnum)
		trans = rise*pnum
		#print pnum, ang, trans
		t = Transform({"type":"spider","psi":ang})	
		t.set_trans(0,0,trans)
		volcopy = vol.process("xform",{"transform":t})
		seammaskcopy = wedgemask.process("xform",{"transform":t})
		sumvol = sumvol*(1-seammaskcopy)+volcopy*seammaskcopy
		
	sumvol.process_inplace("normalize")
	return sumvol

#===========================
def findHsym_MPI(vol,dp,dphi,apix,rmax,rmin,myid,main_node):
	from alignment import helios7
	from mpi import mpi_comm_size, mpi_recv, mpi_send, MPI_TAG_UB, MPI_COMM_WORLD, MPI_FLOAT

	nproc = mpi_comm_size(MPI_COMM_WORLD)	

	ndp=12
	ndphi=12
	dp_step=0.05
	dphi_step=0.05

	nlprms = (2*ndp+1)*(2*ndphi+1)
	#make sure num of helical search is more than num of processors
	if nlprms < nproc:
		mindp = (nproc/4)+1
		ndp,ndphi = mindp,mindp
	if myid == main_node:
		lprms = []
		for i in xrange(-ndp,ndp+1,1):
			for j in xrange(-ndphi,ndphi+1,1):
				lprms.append( dp   + i*dp_step)
				lprms.append( dphi + j*dphi_step)

		recvpara = []
		for im in xrange(nproc):
			helic_ib,helic_ie= MPI_start_end(nlprms, nproc, im)
			recvpara.append(helic_ib )
			recvpara.append(helic_ie )

	para_start, para_end = MPI_start_end(nlprms, nproc, myid)

	list_dps     = [0.0]*((para_end-para_start)*2)
	list_fvalues = [-1.0]*((para_end-para_start)*1)

	if myid == main_node:
		for n in xrange(nproc):
			if n!=main_node: mpi_send(lprms[2*recvpara[2*n]:2*recvpara[2*n+1]], 2*(recvpara[2*n+1]-recvpara[2*n]), MPI_FLOAT, n, MPI_TAG_UB, MPI_COMM_WORLD)
			else:    list_dps = lprms[2*recvpara[2*0]:2*recvpara[2*0+1]]
	else:
		list_dps = mpi_recv((para_end-para_start)*2, MPI_FLOAT, main_node, MPI_TAG_UB, MPI_COMM_WORLD)

	list_dps = map(float, list_dps)

	local_pos = [0.0, 0.0, -1.0e20]
	fract = 0.67
	for i in xrange(para_end-para_start):
		fvalue = helios7(vol, apix, list_dps[i*2], list_dps[i*2+1], fract, rmax, rmin)
		if(fvalue >= local_pos[2]):
			local_pos = [list_dps[i*2], list_dps[i*2+1], fvalue ]
	if myid == main_node:
		list_return = [0.0]*(3*nproc)
		for n in xrange(nproc):
			if n != main_node:
				list_return[3*n:3*n+3] = mpi_recv(3,MPI_FLOAT, n, MPI_TAG_UB, MPI_COMM_WORLD)
			else:
				list_return[3*main_node:3*main_node+3] = local_pos[:]
	else:
		mpi_send(local_pos, 3, MPI_FLOAT, main_node, MPI_TAG_UB, MPI_COMM_WORLD)

	if myid == main_node:
		maxvalue = list_return[2]
		for i in xrange(nproc):
			if( list_return[i*3+2] >= maxvalue ):
				maxvalue = list_return[i*3+2]
				dp       = list_return[i*3+0]
				dphi     = list_return[i*3+1]
		dp   = float(dp)
		dphi = float(dphi)

		return dp,dphi
	return None,None

def legibleTime(sec):
	hours = int(sec/3600)
	sec-= hours*3600
	min = int(sec/60)
	sec -= min*60
	timestr = ""
	if hours > 0: timestr+="%ih,"%hours
	timestr += "%im, %is"%(min,sec)
	return timestr

