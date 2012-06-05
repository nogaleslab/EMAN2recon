# Adapted Richard J. Hall 11/19/2010 rjhall@berkeley.edu
#
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
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
#
#
from EMAN2_cppwrap import *
from global_def import *

def rec3D_MPI(data, snr, symmetry, mask3D, fsc_curve, myid, main_node = 0, rstep = 1.0, odd_start=0, eve_start=1, finfo=None, index=-1, npad = 4):
	'''
	  This function is to be called within an MPI program to do a reconstruction on a dataset kept 
          in the memory, computes reconstruction and through odd-even, in order to get the resolution
	'''
	import os
	from statistics import fsc_mask
	from utilities  import model_blank, reduce_EMData_to_root, get_image, send_EMData, recv_EMData
	from random     import randint
	from mpi        import mpi_comm_size, mpi_comm_rank, MPI_COMM_WORLD
	from reconstruction import recons_ctf_from_fftvol, prepare_recons_ctf, get_image_size
	nproc = mpi_comm_size(MPI_COMM_WORLD)
	
	if nproc==1:
		assert main_node==0
		main_node_odd = main_node
		main_node_eve = main_node
		main_node_all = main_node
	elif nproc==2:
		main_node_odd = main_node
		main_node_eve = (main_node+1)%2
		main_node_all = main_node

		tag_voleve     = 1000
		tag_fftvol_eve = 1001
		tag_weight_eve = 1002
	else:
		#spread CPUs between different nodes to save memory
		main_node_odd = main_node
		main_node_eve = (int(main_node)+nproc-1)%int(nproc)
		main_node_all = (int(main_node)+nproc//2)%int(nproc)

		tag_voleve     = 1000
		tag_fftvol_eve = 1001
		tag_weight_eve = 1002

		tag_fftvol_odd = 1003
		tag_weight_odd = 1004
		tag_volall     = 1005


        if index !=-1 :
		grpdata = []
		for i in xrange( len(data) ):
		    if data[i].get_attr( 'group' ) == index:
		    	    grpdata.append( data[i] )
        	imgdata = grpdata
        else:
		imgdata = data
	nx = get_image_size( imgdata, myid )
	if nx==0:
		ERROR("Warning: no images were given for reconstruction, this usually means there is an empty group, returning empty volume","rec3D",0)
		return model_blank( 2, 2, 2 ), None, model_blank(nx,nx,nx), model_blank(nx,nx,nx)
	
	fftvol_odd_file,weight_odd_file = prepare_recons_ctf(nx, imgdata, snr, symmetry, myid, main_node_odd, odd_start, 2, finfo, npad)
	fftvol_eve_file,weight_eve_file = prepare_recons_ctf(nx, imgdata, snr, symmetry, myid, main_node_eve, eve_start, 2, finfo, npad)
	del imgdata

	if nproc == 1:
		fftvol = get_image(fftvol_odd_file)
		weight = get_image(weight_odd_file)
		volodd = recons_ctf_from_fftvol(nx, fftvol, weight, snr, symmetry, npad = npad)

		fftvol = get_image(fftvol_eve_file)
		weight = get_image(weight_eve_file)
		voleve = recons_ctf_from_fftvol(nx, fftvol, weight, snr, symmetry, npad = npad)
                
		fscdat = fsc_mask( volodd, voleve, mask3D, rstep, fsc_curve)

		fftvol = get_image( fftvol_odd_file )
		fftvol_tmp = get_image(fftvol_eve_file)
		fftvol += fftvol_tmp
		fftvol_tmp = None

		weight = get_image( weight_odd_file )
		weight_tmp = get_image(weight_eve_file)
		weight += weight_tmp
		weight_tmp = None

		volall = recons_ctf_from_fftvol(nx, fftvol, weight, snr, symmetry, npad = npad)
		os.system( "rm -f " + fftvol_odd_file + " " + weight_odd_file )
		os.system( "rm -f " + fftvol_eve_file + " " + weight_eve_file ) 

		return volall,fscdat,volodd,voleve
  
	if nproc == 2:
		if myid == main_node_odd:
			fftvol = get_image( fftvol_odd_file )
			weight = get_image( weight_odd_file )
			volodd = recons_ctf_from_fftvol(nx, fftvol, weight, snr, symmetry, npad = npad)
			voleve = recv_EMData(main_node_eve, tag_voleve)
			fscdat = fsc_mask( volodd, voleve, mask3D, rstep, fsc_curve)
		else:
			assert myid == main_node_eve
			fftvol = get_image( fftvol_eve_file )
			weight = get_image( weight_eve_file )
			voleve = recons_ctf_from_fftvol(nx, fftvol, weight, snr, symmetry, npad = npad)
			send_EMData(voleve, main_node_odd, tag_voleve)

		if myid == main_node_odd:
			fftvol = get_image( fftvol_odd_file )
			fftvol_tmp = recv_EMData( main_node_eve, tag_fftvol_eve )
			fftvol += fftvol_tmp
			fftvol_tmp = None

			weight = get_image( weight_odd_file )
			weight_tmp = recv_EMData( main_node_eve, tag_weight_eve )
			weight += weight_tmp
			weight_tmp = None
        	
			volall = recons_ctf_from_fftvol(nx, fftvol, weight, snr, symmetry, npad = npad)
			os.system( "rm -f " + fftvol_odd_file + " " + weight_odd_file )
			return volall,fscdat,volodd,voleve
		else:
			assert myid == main_node_eve
			fftvol = get_image( fftvol_eve_file )
			weight = get_image( weight_eve_file )
			send_EMData(fftvol, main_node_odd, tag_fftvol_eve )
			send_EMData(weight, main_node_odd, tag_weight_eve )
			os.system( "rm -f " + fftvol_eve_file + " " + weight_eve_file )
			return model_blank(nx,nx,nx), None, model_blank(nx,nx,nx), model_blank(nx,nx,nx)

	# cases from all other number of processors situations
	if myid == main_node_odd:
		fftvol = get_image( fftvol_odd_file )
		send_EMData(fftvol, main_node_eve, tag_fftvol_odd )

		if not(finfo is None):
			finfo.write("fftvol odd sent\n")
			finfo.flush()

		weight = get_image( weight_odd_file )
		send_EMData(weight, main_node_all, tag_weight_odd )

		if not(finfo is None):
			finfo.write("weight odd sent\n")
			finfo.flush()

		volodd = recons_ctf_from_fftvol(nx, fftvol, weight, snr, symmetry, npad = npad)
		del fftvol, weight
		voleve = recv_EMData(main_node_eve, tag_voleve)
		fscdat = fsc_mask(volodd, voleve, mask3D, rstep, fsc_curve)
		volall = recv_EMData(main_node_all, tag_volall)
		os.system( "rm -f " + fftvol_odd_file + " " + weight_odd_file );
		return volall,fscdat,volodd,voleve

	if myid == main_node_eve:
		ftmp = recv_EMData(main_node_odd, tag_fftvol_odd)
		fftvol = get_image( fftvol_eve_file )
		Util.add_img( ftmp, fftvol )
		send_EMData(ftmp, main_node_all, tag_fftvol_eve )
		del ftmp

		weight = get_image( weight_eve_file )
		send_EMData(weight, main_node_all, tag_weight_eve )

		voleve = recons_ctf_from_fftvol(nx, fftvol, weight, snr, symmetry, npad = npad)
		send_EMData(voleve, main_node_odd, tag_voleve)
		os.system( "rm -f " + fftvol_eve_file + " " + weight_eve_file );

		return model_blank(nx,nx,nx), None, model_blank(nx,nx,nx), model_blank(nx,nx,nx)


	if myid == main_node_all:
		fftvol = recv_EMData(main_node_eve, tag_fftvol_eve)
		if not(finfo is None):
			finfo.write( "fftvol odd received\n" )
			finfo.flush()

		weight = recv_EMData(main_node_odd, tag_weight_odd)
		weight_tmp = recv_EMData(main_node_eve, tag_weight_eve)
		Util.add_img( weight, weight_tmp )
		weight_tmp = None

		volall = recons_ctf_from_fftvol(nx, fftvol, weight, snr, symmetry, npad = npad)
		send_EMData(volall, main_node_odd, tag_volall)

		return model_blank(nx,nx,nx),None, model_blank(nx,nx,nx), model_blank(nx,nx,nx)

        return model_blank(nx,nx,nx),None, model_blank(nx,nx,nx), model_blank(nx,nx,nx)

#==========================================
def rec3D_MPI_noCTF(data, symmetry, mask3D, fsc_curve, myid, main_node = 0, rstep = 1.0, odd_start=0, eve_start=1, finfo=None, index = -1, npad = 4):
	'''
	  This function is to be called within an MPI program to do a reconstruction on a dataset kept in the memory 
	  Computes reconstruction and through odd-even, in order to get the resolution
	  if index > -1, projections should have attribute group set and only those whose group matches index will be used in the reconstruction
	    this is for multireference alignment
	'''
	import os
	from statistics import fsc_mask
	from utilities  import model_blank, reduce_EMData_to_root, get_image,send_EMData, recv_EMData
	from random     import randint
	from mpi        import mpi_comm_size, mpi_comm_rank, MPI_COMM_WORLD
	from reconstruction import recons_from_fftvol, prepare_recons

	nproc = mpi_comm_size(MPI_COMM_WORLD)
       
	if nproc==1:
		assert main_node==0
		main_node_odd = main_node
		main_node_eve = main_node
		main_node_all = main_node
	elif nproc==2:
		main_node_odd = main_node
		main_node_eve = (main_node+1)%2
		main_node_all = main_node

		tag_voleve     = 1000
		tag_fftvol_eve = 1001
		tag_weight_eve = 1002
	else:
		#spread CPUs between different nodes to save memory
		main_node_odd = main_node
		main_node_eve = (int(main_node)+nproc-1)%int(nproc)
		main_node_all = (int(main_node)+nproc//2)%int(nproc)

		tag_voleve     = 1000
		tag_fftvol_eve = 1001
		tag_weight_eve = 1002

		tag_fftvol_odd = 1003
		tag_weight_odd = 1004
		tag_volall     = 1005
 
        nx = data[0].get_xsize()

        fftvol_odd_file,weight_odd_file = prepare_recons(data, symmetry, myid, main_node_odd, odd_start, 2, index, finfo, npad)
        fftvol_eve_file,weight_eve_file = prepare_recons(data, symmetry, myid, main_node_eve, eve_start, 2, index, finfo, npad) 
	
	if nproc == 1:
		fftvol = get_image( fftvol_odd_file )
		weight = get_image( weight_odd_file )
		volodd = recons_from_fftvol(nx, fftvol, weight, symmetry, npad)

		fftvol = get_image( fftvol_eve_file )
		weight = get_image( weight_eve_file )
		voleve = recons_from_fftvol(nx, fftvol, weight, symmetry, npad)

		fscdat = fsc_mask( volodd, voleve, mask3D, rstep, fsc_curve)

		fftvol = get_image( fftvol_odd_file )
		Util.add_img( fftvol, get_image(fftvol_eve_file) )

		weight = get_image( weight_odd_file )
		Util.add_img( weight, get_image(weight_eve_file) )

		volall = recons_from_fftvol(nx, fftvol, weight, symmetry, npad)
		os.system( "rm -f " + fftvol_odd_file + " " + weight_odd_file );
		os.system( "rm -f " + fftvol_eve_file + " " + weight_eve_file );
		return volall,fscdat,volodd,voleve

	if nproc == 2:
		if myid == main_node_odd:
			fftvol = get_image( fftvol_odd_file )
			weight = get_image( weight_odd_file )
			volodd = recons_from_fftvol(nx, fftvol, weight, symmetry, npad)
			voleve = recv_EMData(main_node_eve, tag_voleve)
			fscdat = fsc_mask( volodd, voleve, mask3D, rstep, fsc_curve)
		else:
			assert myid == main_node_eve
			fftvol = get_image( fftvol_eve_file )
			weight = get_image( weight_eve_file )
			voleve = recons_from_fftvol(nx, fftvol, weight, symmetry, npad)
			send_EMData(voleve, main_node_odd, tag_voleve)

		if myid == main_node_odd:
			fftvol = get_image( fftvol_odd_file )
			fftvol_tmp = recv_EMData( main_node_eve, tag_fftvol_eve )
			Util.add_img( fftvol, fftvol_tmp )
			fftvol_tmp = None

			weight = get_image( weight_odd_file )
			weight_tmp = recv_EMData( main_node_eve, tag_weight_eve )
			Util.add_img( weight, weight_tmp )
			weight_tmp = None
			volall = recons_from_fftvol(nx, fftvol, weight, symmetry, npad)
			os.system( "rm -f " + fftvol_odd_file + " " + weight_odd_file );
			return volall,fscdat,volodd,voleve
		else:
			assert myid == main_node_eve
			fftvol = get_image( fftvol_eve_file )
			send_EMData(fftvol, main_node_odd, tag_fftvol_eve )

			weight = get_image( weight_eve_file )
			send_EMData(weight, main_node_odd, tag_weight_eve )
			os.system( "rm -f " + fftvol_eve_file + " " + weight_eve_file );
			return model_blank(nx,nx,nx), None, model_blank(nx,nx,nx), model_blank(nx,nx,nx)
	# cases from all other number of processors situations
	if myid == main_node_odd:
		fftvol = get_image( fftvol_odd_file )
		send_EMData(fftvol, main_node_eve, tag_fftvol_odd )

		if not(finfo is None):
			finfo.write("fftvol odd sent\n")
			finfo.flush()

		weight = get_image( weight_odd_file )
		send_EMData(weight, main_node_all, tag_weight_odd )

		if not(finfo is None):
			finfo.write("weight odd sent\n")
			finfo.flush()

		volodd = recons_from_fftvol(nx, fftvol, weight, symmetry, npad)
		del fftvol, weight
		voleve = recv_EMData(main_node_eve, tag_voleve)
		fscdat = fsc_mask( volodd, voleve, mask3D, rstep, fsc_curve)
		volall = recv_EMData(main_node_all, tag_volall)
		os.system( "rm -f " + fftvol_odd_file + " " + weight_odd_file );
		return volall,fscdat,volodd,voleve

	if myid == main_node_eve:
		ftmp = recv_EMData(main_node_odd, tag_fftvol_odd)
		fftvol = get_image( fftvol_eve_file )
		Util.add_img( ftmp, fftvol )
		send_EMData(ftmp, main_node_all, tag_fftvol_eve )
		del ftmp

		weight = get_image( weight_eve_file )
		send_EMData(weight, main_node_all, tag_weight_eve )

		voleve = recons_from_fftvol(nx, fftvol, weight, symmetry, npad)
		send_EMData(voleve, main_node_odd, tag_voleve)
		os.system( "rm -f " + fftvol_eve_file + " " + weight_eve_file );

		return model_blank(nx,nx,nx), None, model_blank(nx,nx,nx), model_blank(nx,nx,nx)


	if myid == main_node_all:
		fftvol = recv_EMData(main_node_eve, tag_fftvol_eve)
		if not(finfo is None):
			finfo.write( "fftvol odd received\n" )
			finfo.flush()

		weight = recv_EMData(main_node_odd, tag_weight_odd)
		weight_tmp = recv_EMData(main_node_eve, tag_weight_eve)
		Util.add_img( weight, weight_tmp )
		weight_tmp = None

		volall = recons_from_fftvol(nx, fftvol, weight, symmetry, npad)
		send_EMData(volall, main_node_odd, tag_volall)

		return model_blank(nx,nx,nx),None, model_blank(nx,nx,nx), model_blank(nx,nx,nx)

	return model_blank(nx,nx,nx), None, model_blank(nx,nx,nx), model_blank(nx,nx,nx)

#===========================
def smart_add(vol1,vol2):
	# add two volumes seamlessly
	nx = vol1.get_xsize()
	sumvol = EMData(nx,nx,nx)
	sumvol.to_zero()
	for i,j,k in ((i,j,k) for i in range(nx) for j in range(nx) for k in range(nx)):
		if vol1[i,j,k] == 0 or vol2[i,j,k] == 0:
			sumvol.set(i,j,k,vol1[i,j,k]+vol2[i,j,k])
		else:
			# at the overlapping area
			sumvol.set(i,j,k,max(vol1[i,j,k],vol2[i,j,k]))
	return sumvol

#===========================
def applyHsym_seam(vol,wedgemaskBig,wedgemaskSm,hfile,apix):
	"""
	apply pseudo-helical symmetry and mask densities on the seam
	"""
	import shutil,subprocess

	# get helical params
	rot,rise,vrot,vrise=readHsym(hfile)

	# convert rise to pixels
	nx = vol.get_xsize()
	rise/=apix
	sym=int(round(360.0/abs(rot)))

	# apply protofilament symmetry
	vol *= wedgemaskBig
	sumvol = vol.copy()
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
		seammaskcopy = wedgemaskSm.process("xform",{"transform":t})
		seammaskcopy.process_inplace("threshold.binary",{"value":0.00001})
		volcopy *= seammaskcopy
		try:
			sumvol.addsmart(volcopy)
		except:
			sumvol = smart_add(sumvol,volcopy)
		
	sumvol.process_inplace("normalize")
	return sumvol
	
#===========================
def align3Dvols(refvol,vol,apix):
	"""
	Aligns vol to refvol, returns the aligned volume
	Search limited to rotation & translation along z axis,
	and only within 1 protofilament monomer
	"""

	# normalize the vols:
	refvol.process_inplace("normalize")
	vol.process_inplace("normalize")

	rt={}
	rt['maxshift']=40/6.02
	rt['stepdelta']=0
	rt['stepphi']=1
	rt['stepx']=0
	rt['stepy']=0
	rt['stepz']=1

	ali = vol.align("refine.3d",refvol,rt)
	alignvol=vol.process("xform",{"transform":ali["xform.align3d"]})
	
	return alignvol

