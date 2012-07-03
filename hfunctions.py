# helical functions for Nogales lab

from EMAN2_cppwrap import *
from global_def import *
import os

#=======================================
def processHelicalVol(vol,voleve,volodd,iref,outdir,itout,dp,dphi,apix,hsearch,findseam=False,vertstep=None,wcmask=None):
	# save a copy of the unsymmetrized volume
	nosymf = os.path.join(outdir,"volNoSym_%s.hdf"%(itout))
	vol.write_image(nosymf,-1)

	# course & fine helical symmetry
	dp,dphi = hsearchsym(vol,dp,dphi,apix,hsearch[1],hsearch[0],0.05)
	dp,dphi = hsearchsym(vol,dp,dphi,apix,hsearch[1],hsearch[0],0.01)

	# find vertical symmetry
	vdp,vdphi=None,None
	if vertstep is not None:
		print vertstep
		vdp,vdphi=vertstep
		# course & fine vertical symmetry
		vdp,vdphi = hsearchsym(vol,vdp,vdphi,apix,hsearch[1],hsearch[0],0.05)
		vdp,vdphi = hsearchsym(vol,vdp,vdphi,apix,hsearch[1],hsearch[0],0.01)

	# inner & outer radii in pixels 
	if findseam is True:
		vol = applySeamSym(vol,dp,dphi,apix)
		voleve = applySeamSym(voleve,dp,dphi,apix)
		volodd = applySeamSym(volodd,dp,dphi,apix)
		vol.write_image(os.path.join(outdir, "volOverSym_%s.hdf"%(itout)),-1)
		# mask out tubulin & apply sym again for seam
		# have to make a new wedgemask for each iteration
		wedgemask=createWedgeMask(vol.get_xsize(),dp,dphi,apix,3,wcmask)
		# recreate the microtubule from seam
		vol = regenerateFromPF(vol,wedgemask,dp,dphi,apix)
		voleve = regenerateFromPF(voleve,wedgemask,dp,dphi,apix)
		volodd = regenerateFromPF(volodd,wedgemask,dp,dphi,apix)
	else:
		vol = vol.helicise(apix, dp, dphi)
		voleve = voleve.helicise(apix, dp, dphi)
		volodd = volodd.helicise(apix, dp, dphi)

	# apply vertical symmetry
	if vertstep is not None:
		vol = applyVertSym(vol,apix,vdp,vdphi)
		voleve = applyVertSym(voleve,apix,vdp,vdphi)
		volodd = applyVertSym(volodd,apix,vdp,vdphi)

	hpar = os.path.join(outdir,"hpar%02d.spi"%(iref))
	f=open(hpar,'a')
	f.write("%.6f\t%.6f"%(dphi,dp))
	if vertstep is not None:
		f.write("\t%.6f\t%.6f"%(vdphi,vdp))
	f.write("\n")
	f.close()

	vol.process_inplace("normalize")
	vol.write_image(os.path.join(outdir, "vol_%s.hdf"%(itout)),-1)

	return (vol,voleve,volodd,dp,dphi,vdp,vdphi)

#===========================
def applyVertSym(vol,apix,dp,dphi):
	# because helicise only applies hsym in one direction,
	# make a flipped copy to apply in other direction
	vol1 = vol.helicise(apix,dp,dphi)
	t = Transform({"type":"spider","theta":180})
	vol.process_inplace("xform",{"transform":t})
	vol2 = vol.helicise(apix,dp,dphi)
	vol2.process_inplace("xform",{"transform":t})
	vol = vol1+vol2
	del vol1,vol2
	return vol

#===========================
def createBoxMask(nx,apix,rmax,lmask,rot=0.0):
	"""
	create a 2D rectangular mask for helical particles
	"""
	# convert lmask to pixels
	lmask = int(lmask/apix)
	falloff = int(lmask*0.3)

	# rmax is in pixels
	if rmax == -1:
		rmax = int(nx/2-falloff)
	rmax*= 2
	
	mx = rmax+(falloff*2)
	my = lmask+(falloff*2)
	if mx > nx: mx=nx
	if my > nx: my=nx
	mask=EMData(mx,my)
	mask.to_one()
	mask.process_inplace("mask.decayedge2d",{"width":falloff})
	mask = Util.pad(mask,nx,nx,1,0,0,0,"edge")	

	if rot > 0:
		t = Transform({"type":"spider","psi":rot})
		mask.process_inplace("xform",{"transform":t})
	return mask

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
def createWedgeMask(nx,rise,twist,apix,ovlp,wcmask=None):
	"""
	a soft wedge that follows helical symmetry
	"""
	import math
	img = EMData(nx,nx)
	img.to_zero()

	# find csym number from rotation
	csym=int(round(360.0/abs(twist)))

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
		t = Transform()
		t.set_rotation({"type":"2d","alpha":-finalrot})
		newslice=img.process("xform",{"transform":t})
		wedge.insert_clip(newslice,(0,0,z))

	if wcmask is not None:
		# for additional masking of features inside wedge
		xmsk = int(wcmask[0]/apix)
		ymsk = int(wcmask[1]/apix)
		mskrad = int(wcmask[2]/apix)

		# see if mask is near the edge:
		edge=ymsk*math.atan(math.pi/csym)
		if (abs(xmsk)+mskrad)>=edge:
			# distance for corresponding positive mask
			edge = int(2*edge)
			xmsk2 = int(math.copysign(edge-abs(xmsk),xmsk)*-1)
			# take max of 1 mask
			avgr = Averagers.get("minmax",{"max":1})
			avgr.add_image_list([wedge,wedgeCylMask(nx,mskrad,xmsk2,ymsk,rot,pos=True)])
			wedge=avgr.finish()
		# multiply 0 mask
		wedge *= wedgeCylMask(nx,mskrad,xmsk,ymsk,rot)

	# odd-numbered protofilaments are off by 1/2 twist
	if csym%2==1:
		t = Transform({"type":"spider","psi":twist/2})
		wedge.process_inplace("xform",{"transform":t})

	#wedge.write_image('wedge_mask_p%d.mrc'%csym)

	return wedge

#===========================
def wedgeCylMask(nx,rad,cx,cy,rot,pos=False):
	# soft-edged cylinder mask for additional masking
	img = EMData(nx,nx)
	img.to_one()
	if pos is True:
		img.to_zero()

	# outer radius
	orad = (rad+rad*.5)

	if abs(cy) > (nx/2-orad) : cy = int((cy/abs(cy))*(nx/2-orad))
	if abs(cx) > (nx/2-orad) : cx = int((cx/abs(cx))*(nx/2-orad))

	for x,y in ((x,y) for x in range(-nx/2,nx/2) for y in range(-nx/2,nx/2)):
		r2 = x**2+y**2
		if r2 < orad*orad:
			if r2 < rad*rad:
				val = 1
			else:
				diff=orad**2-rad**2
				val=1-((r2-rad*rad)/(diff))
			if pos is True:
				img.set(nx/2-x+cx,nx/2+y+cy,val)
			else:
				img.set(nx/2+x+cx,nx/2+y+cy,1-val)

	#img.write_image('test.mrc')
	wmask = EMData(nx,nx,nx)
	for z in range(nx):
		finalrot=((z-nx/2)*rot)/3
		t = Transform()
		t.set_rotation({"type":"2d","alpha":-finalrot})
		newslice=img.process("xform",{"transform":t})
		wmask.insert_clip(newslice,(0,0,z))
	return wmask

#===========================
def createHpar(hpar,pf,params=False,vertstep=None):
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
	# save a copy of the single pf
	sumvol.write_image("pf.hdf",0)

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

