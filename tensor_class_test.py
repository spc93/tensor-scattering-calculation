import TensorScatteringClass as ten
import numpy as np

#CIFfile='/home/spc93/Dropbox/spc_cifs/ZnO Kisi et al icsd_67454.cif'; #ZnO E1E2
#CIFfile='/media/sf_Dropbox/spc_cifs/ZnO Kisi et al icsd_67454.cif'; TimeEven=True #add time symmetry (later)
#hkl=array([1, 1, 5]); hkln=array([0,0,1]); lam=12.4/9.659 ; sitevec=array([1./3, 2./3, 0.]); mpol='E1E2';K=3; time=+1;parity=-1; ##d_an=d_PG008=3.355/4; pol_theta=arcsin(lam/2/d_an);




t=ten.TensorScatteringClass(CIFfile='/home/spc93/Dropbox/spc_cifs/ZnO Kisi et al icsd_67454.cif', Site='Zn1'); print t
#t=ten.TensorScatteringClass(CIFfile='/media/sf_Dropbox/spc_cifs/GaFeO3_icsd_55840.cif'); print t
#t=ten.TensorScatteringClass(CIFfile='/media/sf_Dropbox/spc_cifs/GaFeO3_icsd_55840.cif', Site='Fe2'); print t
#t.equiv_sites(t.sglist, t.sitevec)


### check sum of phases for GaFeO3 (same in new version as old)

print t.TensorCalc(calctype='tensor', K=2, hkl=np.array([0,0,1]))






