import sys, pprint
from copy import deepcopy
#from numpy import *
from numpy.linalg import inv
from numpy.random import rand
#from scipy.misc import *

import numpy as np

#sys.path.append('/home/spc93/python/PyCifRW-3.1.2')
#sys.path.append('/media/DCF0769CF0767D18/python/PyCifRW-3.1.2')
import CifFile
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, suppress=True)


class TensorScatteringClass():  
    '''
    Python class for resonant tensor scattering.
    While this currently has limited capability for magnetic systems, magnetic symmetry operators are used throughout
    If no Site keyword arg supplied then available sites will be displayed before exiting
    Useful methods:
        latt2b          compute reciprocal or real-space B matrix from lattice
        equiv_sites     compute symmetry-equivalent sites for selected site
        invert          inverts current spacegroup operators and sites
        isGroup(sg)     Returns True if sg forms a group or False and shows message if not (self.isGroup(self.sglist) should return True)
    Useful parameters:
        lattice
        B
        sglist
        pglist
        crystalpglist
    '''

    def __init__(self, CIFfile=None, Site=None, TimeEven=False):
        if CIFfile==None:
            raise ValueError('=== Must give CIFfile keyword argument')
        self.CIFfile = CIFfile
        self.Site = Site
        self.cif_obj = CifFile.CifFile(self.CIFfile)
        firstkey = self.cif_obj.keys()[0]; cb = self.cifblock=self.cif_obj[firstkey]
        self.lattice = [float(cb['_cell_length_a'].partition('(')[0]), float(cb['_cell_length_b'].partition('(')[0]), float(cb['_cell_length_c'].partition('(')[0]), float(cb['_cell_angle_alpha'].partition('(')[0]), float(cb['_cell_angle_beta'].partition('(')[0]), float(cb['_cell_angle_gamma'].partition('(')[0])]
        self.all_labels=', '.join(cb['_atom_site_label'])
        if Site==None:  #stop here if no site specified
            return
            
        self.atom_index = cb['_atom_site_label'].index(Site)
        self.sitevec = np.array([float(cb['_atom_site_fract_x'][self.atom_index]), float(cb['_atom_site_fract_y'][self.atom_index]), float(cb['_atom_site_fract_z'][self.atom_index])])

        try:
            self.symxyz=cb['_symmetry_equiv_pos_as_xyz']
        except:
            self.symxyz=cb['_space_group_symop_operation_xyz'] #assume this is full group, not just generators

        self.sglist=self.spacegroup_list_from_genpos_list(self.symxyz)
        
        if TimeEven==True:  #add time-reversal symmetry operator - doubles size of spacegroup
            __sgnew=deepcopy(self.sglist)
            for __sym in __sgnew:
                __sym[2]=-__sym[2]
            self.sglist+=__sgnew
         
        #calculate B matrix
        self.B=self.latt2b(self.lattice)
        
        self.pglist= self.site_sym(self.sglist, self.sitevec)   #point group of site
        self.crystalpglist = self.crystal_point_sym(self.sglist)
        
                
    def __repr__(self):
        if self.Site==None:
            return "=== Atomic site labels: \n" + self.all_labels + "\n=== Use Site keyword to specific a site, e.g. Site = 'Fe1'"
        self.fmt='\n%28s:  '
        return '\nCrystal properties\n' \
        + (self.fmt+'%s') % ('CIF file',self.CIFfile) \
        + (self.fmt+'%.3f %.3f %.3f %.2f %.2f %.2f') % ('Lattice',self.lattice[0], self.lattice[1], self.lattice[2], self.lattice[3], self.lattice[4], self.lattice[5]) \
        + (self.fmt+'%s')  %  ('All sites', self.all_labels)  \
        + (self.fmt+'%s')  %  ('Site selected', self.Site)  \
        + (self.fmt+'%.3f %.3f %.3f')  % ('Site vector', self.sitevec[0], self.sitevec[1], self.sitevec[2]) \
        + (self.fmt+'%i') % ('No. of spacegroup ops', len(self.sglist)) \
        + (self.fmt+'%i') % ('No. of sym ops at site', len(self.pglist)) \
        + (self.fmt+'%i') % ('No. of equiv. sites in cell', len(self.sglist)/len(self.pglist)) \
        + (self.fmt+'%i') % ('No. of pg ops for crystal', len(self.crystalpglist)) \


    def TensorCalc(self, hkl=np.array([0,0,0]), hkln=np.array([0,0,1]), calctype=None, K=None, Parity=+1, Time=+1):
        '''
        hkl, hkln:   hkl values for reflection and azimuthal reference
        calctype:    keyword argument ('tensor', 'magnetic')
        '''
        if calctype==None:
            raise ValueError('=== Must give calculation type using calctype keyword')
        self.K = K
        self.hkl = hkl
        self.hkln = hkln
        self.calctype = calctype
        self.Parity = Parity
        self.Time = Time
        
        txtoe=['Even', 'Odd', 'Either', 'Either'];
        
        outstr = '\nTensor properties\n'\
            +(self.fmt+'%s') % ('Required parity', self.msg(self.Parity, txtoe)) \
            +(self.fmt+'%s') % ('Required time sym.', self.msg(self.Time, txtoe)) \
        
        outstr += self.SF_symmetry(self.sitevec, self.hkl, self.sglist)
        
        if calctype == 'tensor':
            
            #populate tensor with random complex numbers that satisfy the requirements for a Hermitian tensor
            self.Ts=list(np.zeros(2*self.K+1))
            self.Ts[self.K]=rand()
            for Qp in range(1,self.K+1):
                Qn=-Qp; 
                rndr=rand(); rndi=rand();
                self.Ts[K-Qp]=rndr+rndi*1.J;
                self.Ts[K+Qp]=(-1)**Qp*(rndr-rndi*1.J);
        
            Tc1=self.spherical_to_cart_tensor(self.Ts)   #convert to cartesian tensor of same rank
        
        
        
        
#           +(self.fmt+'%i') % ('Tensor rank K', self.K) \
        
#        if K>=0:
#    print 'Tensor calculation'
#    Ts=list(rand(2*K+1));     #use random numbers for quick and dirty solution... (replace each element with 2K+1 element list in future)
#    #Ts=[0.123456+0.392957*1.J, 0.836836+0.296849*1.J, 1, -0.836836+0.296849*1.J,  0.123456-0.392957*1.J]; #fixed pseudo random K=2 (Te etc)
#
#    #############
#    for Qp in range(1,K+1): #next four optional lines modify previous line to adopt Brouder's relationship between tensor components
#        Qn=-Qp; rndr=rand(); rndi=rand();
#        Ts[K-Qp]=rndr+rndi*1.J;
#        Ts[K+Qp]=(-1)**Qp*(rndr-rndi*1.J);
#    #############
#
#    Tc1=spherical_to_cart_tensor(Ts)   #convert to cartesian tensor of same rank
#    #Tc1=rand(3,3); print 'XXXXXX remove this line! generate random cartesian K=2 tensor'
#    Tc_atom=apply_sym(Tc1, pglist, B, P=parity, T=time);  #apply site symmetry using site point group ops and B matrix
#    Tc_crystal=apply_sym(Tc1, crystalpglist, B, P=parity, T=time);  #apply site symmetry using crystal point group ops and B matrix
#    Fc=norm_array(calc_SF(Tc1, sitevec, hkl, sglist, B, P=parity, T=time));   #calc SF Crt tensor using crystal space group and B matrix
#    
#    #######for diagnostics
#    #for sym_phase in sym_phases[0]:
#    #   mat=sym_phase[0]; phases=sym_phase[1];
#    #    #print mat; print  mean(phases); print dot(mat, mat); print;
#    #   print mat; print phases; print;
#    #print sym_phases[0]
#    ##################
#    Ts_atom=norm_array(cart_to_spherical_tensor(Tc_atom));    #atomic spherical tensor
#    print 'Atomic spherical tensor:'; print Ts_atom
#    Ts_crystal=norm_array(cart_to_spherical_tensor(Tc_crystal));    #crystal spherical tensor
#    print 'Bulk crystal spherical tensor:'; print Ts_crystal
#    Fs=norm_array(cart_to_spherical_tensor(Fc));    #SF spherical tensor
#    print 'Structure factor spherical tensor:'; print Fs
        
        


    
  
        
        
        return outstr
    
    def StoneSphericalToCartConversionCoefs(self, K,Calc=True,k=-1j):
        #Condon&Shortley phase convention (k=-i in Stone's paper)
        #from FortranForm (No - CForm?) First List->array, del other lists,spaces, extra bracket around first level
        #If Calc==False then use these expressions from Mathematica, else calculate them numerically
        if not Calc:
            if K==0:
                C=array(1.0)
            elif K==1:
                C=array(((Complex(0,1)/Sqrt(2),1/Sqrt(2),0),(0,0,Complex(0,1)), (Complex(0,-1)/Sqrt(2),1/Sqrt(2),0)))
            elif K==2:
                C=array((((-0.5,Complex(0,0.5),0),(Complex(0,0.5),0.5,0),(0,0,0)),((0,0,-0.5),(0,0,Complex(0,0.5)),(-0.5,Complex(0,0.5),0)),((1/Sqrt(6),0,0),(0,1/Sqrt(6),0),(0,0,-Sqrt(0.6666666666666666))),((0,0,0.5),(0,0,Complex(0,0.5)),(0.5,Complex(0,0.5),0)),((-0.5,Complex(0,-0.5),0),(Complex(0,-0.5),0.5,0),(0,0,0))))
            elif K==3:
                C=array(((((Complex(0,-0.5)/Sqrt(2),-1/(2.*Sqrt(2)),0),(-1/(2.*Sqrt(2)),Complex(0,0.5)/Sqrt(2),0),(0,0,0)),((-1/(2.*Sqrt(2)),Complex(0,0.5)/Sqrt(2),0),(Complex(0,0.5)/Sqrt(2),1/(2.*Sqrt(2)),0),(0,0,0)),((0,0,0),(0,0,0),(0,0,0))),(((0,0,Complex(0,-0.5)/Sqrt(3)),(0,0,-1/(2.*Sqrt(3))),(Complex(0,-0.5)/Sqrt(3),-1/(2.*Sqrt(3)),0)),((0,0,-1/(2.*Sqrt(3))),(0,0,Complex(0,0.5)/Sqrt(3)),(-1/(2.*Sqrt(3)),Complex(0,0.5)/Sqrt(3),0)),((Complex(0,-0.5)/Sqrt(3),-1/(2.*Sqrt(3)),0),(-1/(2.*Sqrt(3)),Complex(0,0.5)/Sqrt(3),0),(0,0,0))),(((Complex(0,0.5)*Sqrt(0.3),1/(2.*Sqrt(30)),0),(1/(2.*Sqrt(30)),Complex(0,0.5)/Sqrt(30),0),(0,0,Complex(0,-1)*Sqrt(0.13333333333333333))),((1/(2.*Sqrt(30)),Complex(0,0.5)/Sqrt(30),0),(Complex(0,0.5)/Sqrt(30),Sqrt(0.3)/2.,0),(0,0,-Sqrt(0.13333333333333333))),((0,0,Complex(0,-1)*Sqrt(0.13333333333333333)),(0,0,-Sqrt(0.13333333333333333)),(Complex(0,-1)*Sqrt(0.13333333333333333),-Sqrt(0.13333333333333333),0))),(((0,0,Complex(0,1)/Sqrt(10)),(0,0,0),(Complex(0,1)/Sqrt(10),0,0)),((0,0,0),(0,0,Complex(0,1)/Sqrt(10)),(0,Complex(0,1)/Sqrt(10),0)),((Complex(0,1)/Sqrt(10),0,0),(0,Complex(0,1)/Sqrt(10),0),(0,0,Complex(0,-1)*Sqrt(0.4)))),(((Complex(0,-0.5)*Sqrt(0.3),1/(2.*Sqrt(30)),0),(1/(2.*Sqrt(30)),Complex(0,-0.5)/Sqrt(30),0),(0,0,Complex(0,1)*Sqrt(0.13333333333333333))),((1/(2.*Sqrt(30)),Complex(0,-0.5)/Sqrt(30),0),(Complex(0,-0.5)/Sqrt(30),Sqrt(0.3)/2.,0),(0,0,-Sqrt(0.13333333333333333))),((0,0,Complex(0,1)*Sqrt(0.13333333333333333)),(0,0,-Sqrt(0.13333333333333333)),(Complex(0,1)*Sqrt(0.13333333333333333),-Sqrt(0.13333333333333333),0))),(((0,0,Complex(0,-0.5)/Sqrt(3)),(0,0,1/(2.*Sqrt(3))),(Complex(0,-0.5)/Sqrt(3),1/(2.*Sqrt(3)),0)),((0,0,1/(2.*Sqrt(3))),(0,0,Complex(0,0.5)/Sqrt(3)),(1/(2.*Sqrt(3)),Complex(0,0.5)/Sqrt(3),0)),((Complex(0,-0.5)/Sqrt(3),1/(2.*Sqrt(3)),0),(1/(2.*Sqrt(3)),Complex(0,0.5)/Sqrt(3),0),(0,0,0))),(((Complex(0,0.5)/Sqrt(2),-1/(2.*Sqrt(2)),0),(-1/(2.*Sqrt(2)),Complex(0,-0.5)/Sqrt(2),0),(0,0,0)),((-1/(2.*Sqrt(2)),Complex(0,-0.5)/Sqrt(2),0),(Complex(0,-0.5)/Sqrt(2),1/(2.*Sqrt(2)),0),(0,0,0)),((0,0,0),(0,0,0),(0,0,0)))))
            elif K==4:
                C=array((((((0.25,Complex(0,-0.25),0),(Complex(0,-0.25),-0.25,0),(0,0,0)),((Complex(0,-0.25),-0.25,0),(-0.25,Complex(0,0.25),0),(0,0,0)),((0,0,0),(0,0,0),(0,0,0))),(((Complex(0,-0.25),-0.25,0),(-0.25,Complex(0,0.25),0),(0,0,0)),((-0.25,Complex(0,0.25),0),(Complex(0,0.25),0.25,0),(0,0,0)),((0,0,0),(0,0,0),(0,0,0))),(((0,0,0),(0,0,0),(0,0,0)),((0,0,0),(0,0,0),(0,0,0)),((0,0,0),(0,0,0),(0,0,0)))),((((0,0,1/(4.*Sqrt(2))),(0,0,Complex(0,-0.25)/Sqrt(2)),(1/(4.*Sqrt(2)),Complex(0,-0.25)/Sqrt(2),0)),((0,0,Complex(0,-0.25)/Sqrt(2)),(0,0,-1/(4.*Sqrt(2))),(Complex(0,-0.25)/Sqrt(2),-1/(4.*Sqrt(2)),0)),((1/(4.*Sqrt(2)),Complex(0,-0.25)/Sqrt(2),0),(Complex(0,-0.25)/Sqrt(2),-1/(4.*Sqrt(2)),0),(0,0,0))),(((0,0,Complex(0,-0.25)/Sqrt(2)),(0,0,-1/(4.*Sqrt(2))),(Complex(0,-0.25)/Sqrt(2),-1/(4.*Sqrt(2)),0)),((0,0,-1/(4.*Sqrt(2))),(0,0,Complex(0,0.25)/Sqrt(2)),(-1/(4.*Sqrt(2)),Complex(0,0.25)/Sqrt(2),0)),((Complex(0,-0.25)/Sqrt(2),-1/(4.*Sqrt(2)),0),(-1/(4.*Sqrt(2)),Complex(0,0.25)/Sqrt(2),0),(0,0,0))),(((1/(4.*Sqrt(2)),Complex(0,-0.25)/Sqrt(2),0),(Complex(0,-0.25)/Sqrt(2),-1/(4.*Sqrt(2)),0),(0,0,0)),((Complex(0,-0.25)/Sqrt(2),-1/(4.*Sqrt(2)),0),(-1/(4.*Sqrt(2)),Complex(0,0.25)/Sqrt(2),0),(0,0,0)),((0,0,0),(0,0,0),(0,0,0)))),((((-1/(2.*Sqrt(7)),Complex(0,0.25)/Sqrt(7),0),(Complex(0,0.25)/Sqrt(7),0,0),(0,0,1/(2.*Sqrt(7)))),((Complex(0,0.25)/Sqrt(7),0,0),(0,Complex(0,0.25)/Sqrt(7),0),(0,0,Complex(0,-0.5)/Sqrt(7))),((0,0,1/(2.*Sqrt(7))),(0,0,Complex(0,-0.5)/Sqrt(7)),(1/(2.*Sqrt(7)),Complex(0,-0.5)/Sqrt(7),0))),(((Complex(0,0.25)/Sqrt(7),0,0),(0,Complex(0,0.25)/Sqrt(7),0),(0,0,Complex(0,-0.5)/Sqrt(7))),((0,Complex(0,0.25)/Sqrt(7),0),(Complex(0,0.25)/Sqrt(7),1/(2.*Sqrt(7)),0),(0,0,-1/(2.*Sqrt(7)))),((0,0,Complex(0,-0.5)/Sqrt(7)),(0,0,-1/(2.*Sqrt(7))),(Complex(0,-0.5)/Sqrt(7),-1/(2.*Sqrt(7)),0))),(((0,0,1/(2.*Sqrt(7))),(0,0,Complex(0,-0.5)/Sqrt(7)),(1/(2.*Sqrt(7)),Complex(0,-0.5)/Sqrt(7),0)),((0,0,Complex(0,-0.5)/Sqrt(7)),(0,0,-1/(2.*Sqrt(7))),(Complex(0,-0.5)/Sqrt(7),-1/(2.*Sqrt(7)),0)),((1/(2.*Sqrt(7)),Complex(0,-0.5)/Sqrt(7),0),(Complex(0,-0.5)/Sqrt(7),-1/(2.*Sqrt(7)),0),(0,0,0)))),((((0,0,-3/(4.*Sqrt(14))),(0,0,Complex(0,0.25)/Sqrt(14)),(-3/(4.*Sqrt(14)),Complex(0,0.25)/Sqrt(14),0)),((0,0,Complex(0,0.25)/Sqrt(14)),(0,0,-1/(4.*Sqrt(14))),(Complex(0,0.25)/Sqrt(14),-1/(4.*Sqrt(14)),0)),((-3/(4.*Sqrt(14)),Complex(0,0.25)/Sqrt(14),0),(Complex(0,0.25)/Sqrt(14),-1/(4.*Sqrt(14)),0),(0,0,1/Sqrt(14)))),(((0,0,Complex(0,0.25)/Sqrt(14)),(0,0,-1/(4.*Sqrt(14))),(Complex(0,0.25)/Sqrt(14),-1/(4.*Sqrt(14)),0)),((0,0,-1/(4.*Sqrt(14))),(0,0,Complex(0,0.75)/Sqrt(14)),(-1/(4.*Sqrt(14)),Complex(0,0.75)/Sqrt(14),0)),((Complex(0,0.25)/Sqrt(14),-1/(4.*Sqrt(14)),0),(-1/(4.*Sqrt(14)),Complex(0,0.75)/Sqrt(14),0),(0,0,Complex(0,-1)/Sqrt(14)))),(((-3/(4.*Sqrt(14)),Complex(0,0.25)/Sqrt(14),0),(Complex(0,0.25)/Sqrt(14),-1/(4.*Sqrt(14)),0),(0,0,1/Sqrt(14))),((Complex(0,0.25)/Sqrt(14),-1/(4.*Sqrt(14)),0),(-1/(4.*Sqrt(14)),Complex(0,0.75)/Sqrt(14),0),(0,0,Complex(0,-1)/Sqrt(14))),((0,0,1/Sqrt(14)),(0,0,Complex(0,-1)/Sqrt(14)),(1/Sqrt(14),Complex(0,-1)/Sqrt(14),0)))),((((3/(2.*Sqrt(70)),0,0),(0,1/(2.*Sqrt(70)),0),(0,0,-Sqrt(0.05714285714285714))),((0,1/(2.*Sqrt(70)),0),(1/(2.*Sqrt(70)),0,0),(0,0,0)),((0,0,-Sqrt(0.05714285714285714)),(0,0,0),(-Sqrt(0.05714285714285714),0,0))),(((0,1/(2.*Sqrt(70)),0),(1/(2.*Sqrt(70)),0,0),(0,0,0)),((1/(2.*Sqrt(70)),0,0),(0,3/(2.*Sqrt(70)),0),(0,0,-Sqrt(0.05714285714285714))),((0,0,0),(0,0,-Sqrt(0.05714285714285714)),(0,-Sqrt(0.05714285714285714),0))),(((0,0,-Sqrt(0.05714285714285714)),(0,0,0),(-Sqrt(0.05714285714285714),0,0)),((0,0,0),(0,0,-Sqrt(0.05714285714285714)),(0,-Sqrt(0.05714285714285714),0)),((-Sqrt(0.05714285714285714),0,0),(0,-Sqrt(0.05714285714285714),0),(0,0,2*Sqrt(0.05714285714285714))))),((((0,0,3/(4.*Sqrt(14))),(0,0,Complex(0,0.25)/Sqrt(14)),(3/(4.*Sqrt(14)),Complex(0,0.25)/Sqrt(14),0)),((0,0,Complex(0,0.25)/Sqrt(14)),(0,0,1/(4.*Sqrt(14))),(Complex(0,0.25)/Sqrt(14),1/(4.*Sqrt(14)),0)),((3/(4.*Sqrt(14)),Complex(0,0.25)/Sqrt(14),0),(Complex(0,0.25)/Sqrt(14),1/(4.*Sqrt(14)),0),(0,0,-(1/Sqrt(14))))),(((0,0,Complex(0,0.25)/Sqrt(14)),(0,0,1/(4.*Sqrt(14))),(Complex(0,0.25)/Sqrt(14),1/(4.*Sqrt(14)),0)),((0,0,1/(4.*Sqrt(14))),(0,0,Complex(0,0.75)/Sqrt(14)),(1/(4.*Sqrt(14)),Complex(0,0.75)/Sqrt(14),0)),((Complex(0,0.25)/Sqrt(14),1/(4.*Sqrt(14)),0),(1/(4.*Sqrt(14)),Complex(0,0.75)/Sqrt(14),0),(0,0,Complex(0,-1)/Sqrt(14)))),(((3/(4.*Sqrt(14)),Complex(0,0.25)/Sqrt(14),0),(Complex(0,0.25)/Sqrt(14),1/(4.*Sqrt(14)),0),(0,0,-(1/Sqrt(14)))),((Complex(0,0.25)/Sqrt(14),1/(4.*Sqrt(14)),0),(1/(4.*Sqrt(14)),Complex(0,0.75)/Sqrt(14),0),(0,0,Complex(0,-1)/Sqrt(14))),((0,0,-(1/Sqrt(14))),(0,0,Complex(0,-1)/Sqrt(14)),(-(1/Sqrt(14)),Complex(0,-1)/Sqrt(14),0)))),((((-1/(2.*Sqrt(7)),Complex(0,-0.25)/Sqrt(7),0),(Complex(0,-0.25)/Sqrt(7),0,0),(0,0,1/(2.*Sqrt(7)))),((Complex(0,-0.25)/Sqrt(7),0,0),(0,Complex(0,-0.25)/Sqrt(7),0),(0,0,Complex(0,0.5)/Sqrt(7))),((0,0,1/(2.*Sqrt(7))),(0,0,Complex(0,0.5)/Sqrt(7)),(1/(2.*Sqrt(7)),Complex(0,0.5)/Sqrt(7),0))),(((Complex(0,-0.25)/Sqrt(7),0,0),(0,Complex(0,-0.25)/Sqrt(7),0),(0,0,Complex(0,0.5)/Sqrt(7))),((0,Complex(0,-0.25)/Sqrt(7),0),(Complex(0,-0.25)/Sqrt(7),1/(2.*Sqrt(7)),0),(0,0,-1/(2.*Sqrt(7)))),((0,0,Complex(0,0.5)/Sqrt(7)),(0,0,-1/(2.*Sqrt(7))),(Complex(0,0.5)/Sqrt(7),-1/(2.*Sqrt(7)),0))),(((0,0,1/(2.*Sqrt(7))),(0,0,Complex(0,0.5)/Sqrt(7)),(1/(2.*Sqrt(7)),Complex(0,0.5)/Sqrt(7),0)),((0,0,Complex(0,0.5)/Sqrt(7)),(0,0,-1/(2.*Sqrt(7))),(Complex(0,0.5)/Sqrt(7),-1/(2.*Sqrt(7)),0)),((1/(2.*Sqrt(7)),Complex(0,0.5)/Sqrt(7),0),(Complex(0,0.5)/Sqrt(7),-1/(2.*Sqrt(7)),0),(0,0,0)))),((((0,0,-1/(4.*Sqrt(2))),(0,0,Complex(0,-0.25)/Sqrt(2)),(-1/(4.*Sqrt(2)),Complex(0,-0.25)/Sqrt(2),0)),((0,0,Complex(0,-0.25)/Sqrt(2)),(0,0,1/(4.*Sqrt(2))),(Complex(0,-0.25)/Sqrt(2),1/(4.*Sqrt(2)),0)),((-1/(4.*Sqrt(2)),Complex(0,-0.25)/Sqrt(2),0),(Complex(0,-0.25)/Sqrt(2),1/(4.*Sqrt(2)),0),(0,0,0))),(((0,0,Complex(0,-0.25)/Sqrt(2)),(0,0,1/(4.*Sqrt(2))),(Complex(0,-0.25)/Sqrt(2),1/(4.*Sqrt(2)),0)),((0,0,1/(4.*Sqrt(2))),(0,0,Complex(0,0.25)/Sqrt(2)),(1/(4.*Sqrt(2)),Complex(0,0.25)/Sqrt(2),0)),((Complex(0,-0.25)/Sqrt(2),1/(4.*Sqrt(2)),0),(1/(4.*Sqrt(2)),Complex(0,0.25)/Sqrt(2),0),(0,0,0))),(((-1/(4.*Sqrt(2)),Complex(0,-0.25)/Sqrt(2),0),(Complex(0,-0.25)/Sqrt(2),1/(4.*Sqrt(2)),0),(0,0,0)),((Complex(0,-0.25)/Sqrt(2),1/(4.*Sqrt(2)),0),(1/(4.*Sqrt(2)),Complex(0,0.25)/Sqrt(2),0),(0,0,0)),((0,0,0),(0,0,0),(0,0,0)))),((((0.25,Complex(0,0.25),0),(Complex(0,0.25),-0.25,0),(0,0,0)),((Complex(0,0.25),-0.25,0),(-0.25,Complex(0,-0.25),0),(0,0,0)),((0,0,0),(0,0,0),(0,0,0))),(((Complex(0,0.25),-0.25,0),(-0.25,Complex(0,-0.25),0),(0,0,0)),((-0.25,Complex(0,-0.25),0),(Complex(0,-0.25),0.25,0),(0,0,0)),((0,0,0),(0,0,0),(0,0,0))),(((0,0,0),(0,0,0),(0,0,0)),((0,0,0),(0,0,0),(0,0,0)),((0,0,0),(0,0,0),(0,0,0))))))
            else: 
                raise ValueError('No Spherical to Cart conversion availble for rank '+str(K))
        else:
            CS=[];
            for i in range(1,K+1):      #generate coupling sequence CS=[1,2,3...K]
                CS+=[i]
            C=array(StoneCoefficients(CS,k=k)).transpose()   
        return C   
            
    def StoneCoefficients(self, CouplingSequenceList,k=-1j):
        '''
        StoneCoefficients(CouplingSequenceList,k=phase_convention)
        Sympy Spherical-Cartesian conversion coefficients from
        A.J. Stone Molecular Physics 29 1461 (1975) (Equation 1.9)
        CouplingSequenceList is the coupling sequence for spherical tensors, 
            each time coupling to a new vector to form a tensor of given rank
            (sequence always starts with 1)
        k=-I for Condon & Shortley phase convention (default) of k=1 for Racah
        e.g. StoneCoefficients([1,2,3]) returns conversion coefficients for K=3, coupling with 
        maximum rank and Condon & Shortley (default) phase convention
        Example:     C123=StoneCoefficients([1,2,3])    returns conversion matrix for coupling sequence 123 (K=3)
                print lcontract(C123,3,[1,0,0,0,0,0,0]) returns table values for Q=-3
        Numpy version converted from, Sympy version
        '''
        rt=2**0.5               #sqrt(2)
        #Cartesian index sequence: x,y,z; spherical index sequence -q...+q
        C1=[[1j*k/rt,0,-1j*k/rt],[k/rt,0,k/rt],[0,1j*k,0]]    #coefficients for vector (K=1)=C1
        N=len(CouplingSequenceList)                #total number of vectors coupled    
        #diag('line 311',['rt','C1','N'],locals()) 
        if N==0:
            C=[1]    
        elif N>0:
            C=C1
        if N>1:
            if CouplingSequenceList[0]!=1:
                raise ValueError('First rank in sequence must be 1')
            for J in CouplingSequenceList[1:]:        #loop through all J's after the first
                Cnew=StoneCoupleVector(C,J,C1)        #couple to next vector to make tensor of rank J
                C=Cnew     
                #diag('stone coef main loop',['C','J','C1','Cnew'],locals()) 
        return C
       
    
    
    def spherical_to_cart_tensor(self, Ts):
        K=(len(Ts)-1)/2; #spherical tensor rank
        C=self.StoneSphericalToCartConversionCoefs(K).conjugate()
        Tc=C[0]*0.0;    #array of zeros
        for kk in range(-K, K+1):
            Tc=Tc+Ts[kk+K]*C[kk+K]
        return Tc
            
    def SF_symmetry(self, R, hkl, spacegroup_list):
        #analyse symmetry of any possible structure factor (mainly for information)
        #returns [sym_phases, gen_scalar_allowed, site_scalar_allowed, tensor_allowed, Psym, Tsym, PTsym]
        tol=1e-6
        inv_mat=np.array([[-1.0,0.0,0.0],[0.0,-1.0,0.0],[0.0,0.0,-1.0]]); #inversion operator
        identity_mat=np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]); #identity operator
        identy_phase=np.exp(np.pi*2.j * np.dot(hkl, R)); #phase for initial site vector
        sym_phases=[];      #list of  symmetry operators (matrices) and the set of phases. Start with empty list.
        Rgen=rand(3);      #use random number to simulate general position to identify spacegroup forbidden reflections
        sum_phase_all=0;    #sum of all phases for site (to get scalar structure factor for site)
        sum_phase_gen=0;    #sum of phases for geneal (random) position
        for sym in spacegroup_list:
            mat=sym[0]
            vec=sym[1]
            time=sym[2]
            newR=np.dot(mat, R)+vec
            newRgen=np.dot(mat, Rgen)+vec
            phase=np.exp(np.pi*2.j * np.dot(hkl, newR))/identy_phase    #change phases so first one is unity
            #phase=np.exp(np.pi*2.j * np.dot(hkl, newR)); print 'phase: ',  phase;  print    #temp##################
    
            sum_phase_all+=np.exp(np.pi*2.j * np.dot(hkl, newR))              #add new phase for site to sum
            sum_phase_gen+=np.exp(np.pi*2.j * np.dot(hkl, newRgen))              #add new phase for general (random) position to sum
            newsym=1;
            for sym_phase in sym_phases:
                if np.allclose(mat,sym_phase[0], atol=tol) and np.allclose(time,sym_phase[1], atol=tol):              #compare mat with sym op is sym_phase list
                    newsym=0;                                                       #if already in list then not new
                    sym_phase[2]+=[phase];                                  #add new phase to phse list for sym op
                    break
            if newsym==1:                                                         #sym op not in list so make a new entry
                sym_phases+=[[mat, time, [phase]]]
    
    
        sum_all_phases=0;                       #running total of all phases
        sum_Pplus_Tplus=0   #P even, T even etc
        sum_Pminus_Tplus=0
        sum_Pplus_Tminus=0
        sum_Pminus_Tminus=0
        Psym=Tsym=PTsym=None   #symmetries will be +1, -1 or 0 (even, odd, none)
        gen_scalar_allowed=site_scalar_allowed=1
        tensor_allowed=0
        if abs(sum_phase_all)<tol:
            site_scalar_allowed=0
        if abs(sum_phase_gen)<tol:
            gen_scalar_allowed=0        
        
    
        sum_phases=[]   #sum of phases for each symmetry operator
        for sym_phase in sym_phases:
            sum_phases+=[sum(sym_phase[2])]
            sum_all_phases+=sum(sym_phase[2])                             #add all phases (if all zero then forbidden for scalar)
            #if not np.allclose(sum(sym_phase[1]), 0, atol=tol):
            if not np.allclose(sum(sym_phase[2]), 0, atol=tol): ########## fix bug - hangover from before T was added
                tensor_allowed=1
            if np.allclose(sym_phase[0], identity_mat, atol=tol) and abs(sym_phase[1]-1)<tol:
                sum_Pplus_Tplus+=sum(sym_phase[2])
            elif np.allclose(sym_phase[0], inv_mat, atol=tol) and abs(sym_phase[1]-1)<tol:
                sum_Pminus_Tplus+=sum(sym_phase[2])
            elif np.allclose(sym_phase[0], identity_mat, atol=tol) and abs(sym_phase[1]+1)<tol: #time odd
                sum_Pplus_Tminus+=sum(sym_phase[2])                           
            elif np.allclose(sym_phase[0], inv_mat, atol=tol) and abs(sym_phase[1]+1)<tol:
                sum_Pminus_Tminus+=sum(sym_phase[2])                            
    
    
    #    if tensor_allowed and abs(sum_Pplus_Tplus)>tol: #if there is no item with plus time and plus parity then there is no specific symmetry
        if tensor_allowed: #if there is no item with plus time and plus parity then there is no specific symmetry
            if sum_Pplus_Tplus-sum_Pminus_Tplus==0:
                Psym=+1
            if sum_Pplus_Tplus+sum_Pminus_Tplus==0:
                Psym=-1
            if sum_Pplus_Tplus-sum_Pplus_Tminus==0:
                Tsym=+1
            if sum_Pplus_Tplus+sum_Pplus_Tminus==0:
                Tsym=-1
            if sum_Pplus_Tplus-sum_Pminus_Tminus==0:
                PTsym=+1
            if sum_Pplus_Tplus+sum_Pminus_Tminus==0:
                PTsym=-1
    
        sum_phases=np.array(sum_phases)
        if np.allclose(sum_phases, sum_phases.real, atol=tol):
            sum_phases=np.real(sum_phases)
        else:
            print'=== Warning: sum of phases is compex. This was not np.expected (see below):\n',sum_phases
        if abs(sum_phases[0])>tol:
            sum_phases=np.array(sum_phases); sum_phases=sum_phases/sum_phases[0] #normalize to first (identity)
        else:
            print '=== Warning: the phase sum for first (identity) operator is close to zero. This was not np.expected'
    
                  
        txtyn=['Yes','Invalid value', 'No', 'Invalid value']; txtoe=['Even', 'Odd', 'Either', 'Either']; 
        outstr = \
            (self.fmt+'%s') % ('Site allowed', self.msg(site_scalar_allowed, txt=txtyn)) \
            +(self.fmt+'%s') % ('Spacegroup allowed', self.msg(gen_scalar_allowed, txt=txtyn)) \
            +(self.fmt+'%s') % ('Tensor allowed:', self.msg(tensor_allowed, txt=txtyn)) \
            +(self.fmt+'%s') % ('Parity', self.msg(Psym, txt=txtoe) ) \
            +(self.fmt+'%s') % ('Time', self.msg(Tsym, txt=txtoe)) \
            +(self.fmt+'%s') % ('PT', self.msg(PTsym, txt=txtoe)) \


    
        sym_sum_phases=deepcopy(sym_phases)
        for ii in range(len(sym_sum_phases)):
            sym_sum_phases[ii][2]=sum_phases[ii]
  
        self.sym_sum_phases, self.sum_phases, self.gen_scalar_allowed, self.site_scalar_allowed, self.tensor_allowed, self.Psym, self.Tsym, self.PTsym, self.sym_phases\
            = sym_sum_phases, sum_phases, gen_scalar_allowed, site_scalar_allowed, tensor_allowed, Psym, Tsym, PTsym, sym_phases
 
    
    
    #save as attributes
    #           return [sym_sum_phases, sum_phases, gen_scalar_allowed, site_scalar_allowed, tensor_allowed, Psym, Tsym, PTsym, sym_phases,]
    
        return outstr
  
    def msg(self, num, txt=['plus','minus','zero','other']):
        #return message text for +1,-1, 0, other (e.g. None)
        if num==1:
            str=txt[0]
        elif num==-1:
            str=txt[1]
        elif num==0:
            str=txt[2]
        else:
            str=txt[3]
        return str
        
        
        
        
        
#TimeEven=True
#en=8.0
#hkl=array([0,1,1]); Fe=array([0,0,0]); S=array([0.385,0.385,0.385]);hkln=array([0,2,0]); lam=12.4/en ; 
#sitevec=S; 
#mpol='E1E1';K=2; time=+1;parity=+1; d_an=d_PG004=3.355/2; pol_theta=arcsin(lam/2/d_an);
        
        
    def spacegroup_list_from_genpos_list(self, genposlist):
        sglist=[];
        for genpos in genposlist:
            sglist+=[self.genpos2matvec(genpos)+[1]] #add +1 to indicate time symmetry
        return sglist    

    def genpos2matvec(self,gen_pos_string):
        'convert general position string to vector/matrix form (floats) using lists as row vectors'
        #gp=gen_pos_string
        gp=gen_pos_string.lower();
        x=y=z=0.; vec=list(eval(gp.replace('/','./')))
        x=1.; y=z=0.; m0=list(eval(gp.replace('/','./'))); m0[0]=m0[0]-vec[0]; m0[1]=m0[1]-vec[1];m0[2]=m0[2]-vec[2];
        y=1.; x=z=0.; m1=list(eval(gp.replace('/','./'))); m1[0]=m1[0]-vec[0]; m1[1]=m1[1]-vec[1];m1[2]=m1[2]-vec[2];
        z=1.; x=y=0.; m2=list(eval(gp.replace('/','./'))); m2[0]=m2[0]-vec[0]; m2[1]=m2[1]-vec[1];m2[2]=m2[2]-vec[2];
        return [np.array([m0, m1, m2]).T, np.array(vec)]       

    def latt2b(self, lat, direct=False, BLstyle=False):
        #follow Busing&Levy, D.E.Sands
        #direct=False: normal recip space B matrix (B&L)
        #direct=True, BLstyle=True: Busing & Levy style applied to real space (i.e. x||a)
        #direct=True, BLstyle=False: Real space B matrix compatible with recip space B matrix
        a1=lat[0];    a2=lat[1];    a3=lat[2];
        alpha1=lat[3]*np.pi/180;    alpha2=lat[4]*np.pi/180;    alpha3=lat[5]*np.pi/180;
        v=a1*a2*a3*np.sqrt(1-np.cos(alpha1)**2-np.cos(alpha2)**2-np.cos(alpha3)**2+2*np.cos(alpha1)*np.cos(alpha2)*np.cos(alpha3))
        b1=a2*a3*np.sin(alpha1)/v;    b2=a3*a1*np.sin(alpha2)/v;    b3=a1*a2*np.sin(alpha3)/v
        beta1=np.arccos((np.cos(alpha2)*np.cos(alpha3)-np.cos(alpha1))/np.sin(alpha2)/np.sin(alpha3))
        beta2=np.arccos((np.cos(alpha1)*np.cos(alpha3)-np.cos(alpha2))/np.sin(alpha3)/np.sin(alpha1))
        beta3=np.arccos((np.cos(alpha1)*np.cos(alpha2)-np.cos(alpha3))/np.sin(alpha1)/np.sin(alpha2))
        #reciprocal space
        B=np.array([    [b1, b2*np.cos(beta3), b3*np.cos(beta2)],
        [0, b2*np.sin(beta3), -b3*np.sin(beta2)*np.cos(alpha1)],
        [0, 0, 1/a3], ])
        #real space: Busing & Levy style applied to real space (i.e. x||a)
        BD=np.array([    [a1, a2*np.cos(alpha3), a3*np.cos(alpha2)],
        [0, a2*np.sin(alpha3), -a3*np.sin(alpha2)*np.cos(beta1)],
        [0, 0, 1/b3], ])
        # Real space  B matrix consistent with recip space B matrix (useful of calculations involve real and reiprocal space)
        Bdd=inv(B.transpose())
    
        if not direct:
            return B  
        else:
            if BLstyle:
                return BD
            else:
                return Bdd 
            
            
            
    def site_sym(self,spacegroup_list, sitevec):
        symlist=[];
        tol=1e-6;   #coordinates treated as indintical if within tol
        sitevec=(sitevec+tol)%1-tol;      #map into range 0<=x<1 using tolerance tol
        for sg in spacegroup_list:
            newpos=np.dot(sg[0], sitevec)+sg[1]    #new coordinates after applying symmetry operator
            newpos=(newpos+tol)%1-tol;      #map into range 0<=x<1 using tolerance tol
            if np.allclose(newpos, sitevec, atol=.001):    #spacegroup operator presenves position so it is a point group operator
                symlist+=[[sg[0],sg[2]]]                                   #add matrix and time part of sg op to pg but...
                for sym in symlist[0:-1]:
                    if np.allclose(sym[0], sg[0], atol=.001) and abs(sym[1]-sg[2])<tol:     #... remove it again if already in list
                        symlist=symlist[0: -1]
                        break
        return symlist
       
        
    def equiv_sites(self, spacegroup_list, sitevec):
        '''
        equiv_sites(spacegroup_list, sitevec)
        returns symmetry-equivalent sites for selected site
        '''
        poslist=[sitevec];
        tol=1e-6;   #coordinates treated as indintical if within tol
        for sg in spacegroup_list:
            newpos=np.dot(sg[0], sitevec)+sg[1]    #new coordinates after applying symmetry operator
            newpos=(newpos+tol)%1-tol;      #map into range 0<=x<1 using tolerance tol
            poslist+=[newpos]            #add new position to list...
            for pos in poslist[0:-1]:
                if np.allclose(pos, newpos, atol=tol):    #...if position already in list
                    poslist=poslist[0:-1]                                   #...remove it
        return poslist

    def crystal_point_sym(self, spacegroup_list):
        symlist=[]
        tol=1e-6;   #coordinates treated as indintical if within tol
        for sg in spacegroup_list:
            symlist+=[[sg[0],sg[2]]]                         #add matrix  and time part of sg op to pg but...
            for sym in symlist[0:-1]:
                if np.allclose(sym[0], sg[0], atol=tol) and abs(sym[1]-sg[2])<tol:     #... remove it again if already in list
                    symlist=symlist[0: -1]
                    break
        return symlist

    def invert(self):
        '''
        self.invert()
        inverts current spacegroup operators and sites
        '''
        newsg=deepcopy(self.sglist)
        for sgop in newsg:
            sgop[1]=-sgop[1]
        self.sglist = newsg
        self.sitevec = self.firstCell(-self.sitevec)
        return

    def firstCell(self, V):
        #fold V back to first unit cell (0..1)
        return np.array([z-np.floor(z) for z in V])
    
    def isGroup(self, G):
        tol=0.0000001
        #group is a list of [mat, vec, timescalar]
        eye_index=-1
        for ind in range(len(G)):
            if np.all(abs(G[ind][0]-np.eye(3))<tol) and np.all(abs(G[ind][1]-np.zeros(3))<tol) and abs(G[ind][2]-1)<tol:
                eye_index=ind
        if eye_index !=0:
            print '=== Warning: Identity not first element'
        for S1 in G:
            for S2 in G:
                M3=np.dot(S1[0], S2[0])
                V3=self.firstCell(S1[1] + np.dot(S1[0],S2[1]));    #fold back to first unit cell
                T3=S1[2] * S2[2]
                n=0
                for S3 in G:
                    if np.all(abs(M3-S3[0])<tol) and np.all(abs(V3-self.firstCell(S3[1]))<tol) and abs(T3-S3[2])<tol:
                        n+=1
                if n!=1:
                    print '=== Warning: Not a group!'
                    print '=== There should be one occurence of the following symmetry operator but were %i' % n
                    print M3, V3, T3, '\n=== Derived from\n', S1, '\n=== and\n', S2
                    return False
        return True
        
    
    
    



