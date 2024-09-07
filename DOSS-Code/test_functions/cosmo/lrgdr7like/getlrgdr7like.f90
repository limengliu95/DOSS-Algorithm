!   Written by Beth Reid 2009; portions taken from CosmoMC
!     by Antony Lewis (http://cosmologist.info/) and Sarah Bridle.
!     See readme.html for documentation. 
!
!   This example driver assumes that the model matter power spectra 
!   generated with our CAMB  patch are located in the current directory in files
!   models/lrgdr7model_matterpowerzFAR.dat, models/lrgdr7model_matterpowerzMID.dat, 
!   models/lrgdr7model_matterpowerzNEAR.dat, models/lrgdr7model_matterpowerz0.dat 
!   the fiducial model dat files, located in models/lrgdr7fiducialmodel_matterpowerz*.dat
!   are provided with the code release and necessary to compute the likelihoods

program getlrgdr7like
  use precision
  use lrgconstants
  use lrgutils
  implicit none
  
  integer :: ios
  integer :: iopb,i
  real, dimension(num_matter_power) :: halopowerlrgtheory
  Type (mpkdataset) :: mset  !! holds all data, windows, covariance matrix
  real(dl) :: lrglnlike
  real(dl) :: omegak,omegav,w

  call fill_LRGTheory(halopowerlrgtheory,omegak,omegav,w)
  call ReadmpkDataset(mset)
  lrglnlike = LSS_LRG_mpklike(halopowerlrgtheory,mset,omegak,omegav,w)
  print *,lrglnlike
  end program getlrgdr7like
