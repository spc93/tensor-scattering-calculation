import TensorScatteringClass as ten

t=ten.TensorScatteringClass(CIFfile='/home/spc93/spc_cifs/nchem.2848-s6.cif', Site='Cu1');
t.PlotIntensityInPolarizationChannels('E1E1', lam=1, hkl=np.array([0,0,1]), hkln=np.array([1,0,0]), K=2, Time=1, Parity=1, mk=None, nk=None, sk=None, sigmapi='sigma')
t.print_tensors()

