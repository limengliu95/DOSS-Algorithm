!   Written by Beth Reid 2009; portions taken from CosmoMC
!     by Antony Lewis (http://cosmologist.info/) and Sarah Bridle.
!     See readme.html for documentation. 
!     Data files are provided with the code.  Assumed to be in current
!     directory named data/lrgdr7_kbands.txt, data/lrgdr7_ccmeasurements.txt,
!     data/lrgdr7_windows.txt, data/lrgdr7_zerowindowfxn.txt, 
!     data/lrgdr7_zerowindowfxnsubtractdat.txt, data/lrgdr7_invcov.txt
!     For your model to be tested, place the CAMB patch matterpower output 
!     files in the models/ subdirectory.

module Precision
      implicit none
      public

      integer, parameter :: dl = KIND(1.d0)
      integer, parameter :: sp = KIND(1.0)

end module Precision

module lrgconstants
  use precision
  implicit none
  public
  
  !! BR09_LRG hard-coded values in CosmoMC LRG version
  integer, parameter :: tmp_file_unit = 50
  
  !! parameters of kvec at which theory/fiducial power spectra are stored (produced from CAMB patch)
  integer, parameter ::  num_matter_power = 300
  real, parameter ::  matter_power_minkh =  0.999e-4  !minimum value of k/h to store
  real, parameter ::  matter_power_dlnkh = 0.03     !log spacing in k/h
  
  real(dl), parameter ::  aNEAR = 0.809717d0, aMID = 0.745156d0, aFAR = 0.70373d0
  real(dl), parameter ::  zNEAR = 0.235d0, zMID = 0.342d0, zFAR = 0.421d0  
  real(dl), parameter ::  sigma2BAONEAR = 86.9988, sigma2BAOMID = 85.1374, sigma2BAOFAR = 84.5958  !!BAO LRG damping
  real(dl), parameter :: wNEAR = 0.395d0, wMID = 0.355d0, wFAR = 0.250d0
  real(dl), parameter :: zeffDR7 = 0.312782  !! redshift at which a_scl is evaluated.

  integer, parameter :: mpk_d = kind(1.d0)
  integer, parameter :: wp = selected_real_kind(11,99)

  real(dl), parameter :: k1 = 0.1d0, k2 = 0.2d0, s1 = 0.04d0, s2 = 0.10d0, a1maxval = 1.1482d0
  integer, parameter :: nptsa1 = 41, nptsa2 = 41, nptstot = 325  !! but total number of points to evaluate is much smaller than 41**2 because lots of the space is not allowed by the s1,s2 constraints.

end module lrgconstants

module lrgutils
use precision
use lrgconstants
implicit none
public

! adapted from mpk.f90 in CosmoMC
Type mpkdataset
    integer :: num_mpk_points_use ! total number of points used (ie. max-min+1)
    integer :: num_mpk_kbands_use ! total number of kbands used (ie. max-min+1)
    character(LEN=20) :: name
    real, pointer, dimension(:,:) :: mpk_W, mpk_invcov
    real, pointer, dimension(:) :: mpk_P, mpk_sdev, mpk_k
    real, pointer, dimension(:) :: mpk_zerowindowfxn
    real, pointer, dimension(:) :: mpk_zerowindowfxnsubtractdat
    real :: mpk_zerowindowfxnsubtractdatnorm !!the 0th entry in windowfxnsubtract file
    logical :: use_scaling !as SDSS_lrgDR3
end Type mpkdataset

! only want to compute these once.
real(dl), dimension(nptstot) :: a1list, a2list

real(dl), dimension(3) :: zeval, zweight, sigma2BAO

real(dl), dimension(3) :: powerscaletoz0  !! this is to scale the amplitude of the redshift slices power spectra to the z=0 amplitude; this is the assumption of the model.

contains

!! adapted but uses mpk.f90
  subroutine ReadmpkDataset(mset)
    Type (mpkdataset), intent(out) :: mset

    integer :: num_mpk_points_use
    integer :: num_mpk_kbands_use
    integer i,iopb
    real keff,klo,khi,beff, dummyreal1, dummyreal2
    character(80) :: dummychar
    real, dimension(:,:), allocatable :: mpk_Wfull, mpk_covfull
    real, dimension(:), allocatable :: mpk_kfull

    real, dimension(:), allocatable :: mpk_zerowindowfxnsubtractdattemp

    !! hard code for LRGs
    num_mpk_points_use = 45
    num_mpk_kbands_use = 250
    mset%num_mpk_points_use = num_mpk_points_use
    mset%num_mpk_kbands_use = num_mpk_kbands_use
    mset%use_scaling = .true.

    allocate(mset%mpk_P(mset%num_mpk_points_use))
    allocate(mset%mpk_k(mset%num_mpk_kbands_use))
    allocate(mset%mpk_W(mset%num_mpk_points_use,mset%num_mpk_kbands_use))
    allocate(mset%mpk_zerowindowfxn(mset%num_mpk_kbands_use))
    allocate(mset%mpk_zerowindowfxnsubtractdat(mset%num_mpk_points_use))
    allocate(mpk_zerowindowfxnsubtractdattemp(num_mpk_points_use+1))  !!need to add 1 to get the normalization held in the first (really zeroth) entry
    allocate(mset%mpk_invcov(num_mpk_points_use,num_mpk_points_use))

    call ReadVectorBR('data/lrgdr7_kbands.txt',mset%mpk_k,num_mpk_kbands_use)
    open(unit=tmp_file_unit,file='data/lrgdr7_ccmeasurements.txt', form='formatted',err=600, iostat=iopb)
    mset%mpk_P=0.
    read (tmp_file_unit,*) dummychar
    read (tmp_file_unit,*) dummychar
    do i =1, mset%num_mpk_points_use
       read (tmp_file_unit,*, iostat=iopb) keff,klo,khi,mset%mpk_P(i),dummyreal1,dummyreal2
    end do
    close(tmp_file_unit)
600 if (iopb.ne.0) then
       stop 'Error reading mpk file'
    endif
    call ReadMatrixBR('data/lrgdr7_windows.txt',mset%mpk_W,num_mpk_points_use,num_mpk_kbands_use)
    call ReadVectorBR('data/lrgdr7_zerowindowfxn.txt',mset%mpk_zerowindowfxn,num_mpk_kbands_use)
    call ReadVectorBR('data/lrgdr7_zerowindowfxnsubtractdat.txt',mpk_zerowindowfxnsubtractdattemp,num_mpk_points_use+1)
    mset%mpk_zerowindowfxnsubtractdat(1:num_mpk_points_use) = mpk_zerowindowfxnsubtractdattemp(2:num_mpk_points_use+1)
    mset%mpk_zerowindowfxnsubtractdatnorm = mpk_zerowindowfxnsubtractdattemp(1)
    deallocate(mpk_zerowindowfxnsubtractdattemp)
    call ReadMatrixBR('data/lrgdr7_invcov.txt',mset%mpk_invcov,num_mpk_points_use,num_mpk_points_use)

    call LSS_LRG_mpklike_init()  !!sets up nuisance parameter integration.
  end subroutine ReadmpkDataset

subroutine LSS_LRG_mpklike_init()
   real(dl) :: a1val, a2val
   real(dl) :: da1, da2  ! spacing of numerical integral over nuisance params.
   integer :: countcheck = 0
   integer :: i, j

   da1 = a1maxval/(nptsa1/2)
   da2 = a2maxpos(-a1maxval)/(nptsa2/2)
   do i = -nptsa1/2, nptsa1/2
      do j = -nptsa2/2, nptsa2/2
         a1val = da1*i
         a2val = da2*j

         if ((a2val >= 0.0d0 .and. a2val <= a2maxpos(a1val) .and. a2val >= a2minfinalpos(a1val)) .or. &
     & (a2val <= 0.0d0 .and. a2val <= a2maxfinalneg(a1val) .and. a2val >= a2minneg(a1val))) then
            if(testa1a2(a1val,a2val) .eqv. .false.)  then
               print *,'Failed a1, a2: ',a1val,a2val
               if (a2val >= 0.0d0) print *,'pos', a2maxpos(a1val), a2minfinalpos(a1val)
               if (a2val <= 0.0d0) print *,'neg', a2maxfinalneg(a1val), a2minneg(a1val)
               stop
            end if
            countcheck = countcheck + 1
            if(countcheck > nptstot) then
               print *, 'countcheck > nptstot failure.'
               stop
            end if
            a1list(countcheck) = a1val
            a2list(countcheck) = a2val
         end if
      end do
   end do
   if(countcheck .ne. nptstot) then
     print *, 'countcheck issue', countcheck, nptstot
     stop
   end if

end subroutine LSS_LRG_mpklike_init

! HARD CODING OF POLYNOMIAL FITS TO NEAR, MID, FAR SUBSAMPLES.
subroutine LRGtoICsmooth(k,fidpolys)
  real(dl), intent(in) :: k
  real(dl) :: fidNEAR, fidMID, fidFAR
  real(dl), dimension(3), intent(out) :: fidpolys

  if(k < 0.194055d0) then !!this is where the two polynomials are equal
    fidNEAR = (1.0d0 - 0.680886d0*k + 6.48151d0*k**2)
  else
    fidNEAR = (1.0d0 - 2.13627d0*k + 21.0537d0*k**2 - 50.1167d0*k**3 + 36.8155d0*k**4)*1.04482d0
  end if

  if(k < 0.19431) then
    fidMID = (1.0d0 - 0.530799d0*k + 6.31822d0*k**2)
  else
    fidMID = (1.0d0 - 1.97873d0*k + 20.8551d0*k**2 - 50.0376d0*k**3 + 36.4056d0*k**4)*1.04384
  end if

  if(k < 0.19148) then
    fidFAR = (1.0d0 - 0.475028d0*k + 6.69004d0*k**2)
  else
    fidFAR = (1.0d0 - 1.84891d0*k + 21.3479d0*k**2 - 52.4846d0*k**3 + 38.9541d0*k**4)*1.03753
  end if
  fidpolys(1) = fidNEAR
  fidpolys(2) = fidMID
  fidpolys(3) = fidFAR
end subroutine LRGtoICsmooth

subroutine fill_LRGTheory(halopowerlrgtheory, omegak,omegav,w)

  real, dimension(num_matter_power), intent(out) :: halopowerlrgtheory
  real, dimension(num_matter_power) :: khvec
  real, dimension(num_matter_power, 3) :: outpower, outpowernw, outpowerrationwhalofit
  real, dimension(num_matter_power, 3) :: ratio_power_nw_nl_fid
  real(dl), intent(out) :: omegak,omegav,w
  real(dl) :: omegakdummy,omegavdummy,wdummy
  integer :: ios
  integer :: iopb,i,ik,iz

  real(dl), dimension(3) :: fidpolys, holdval
  real(dl) :: kval, plin, psmooth, rationwhalofit
  real(dl) :: getabstransferscaleNEAR, getabstransferscaleMID, getabstransferscaleFAR, getabstransferscalez0
  real(dl) :: getabstransferscalefiddummy

  real(dl) :: expval, psmear, nlrat  

  sigma2BAO(1) = sigma2BAONEAR
  sigma2BAO(2) = sigma2BAOMID
  sigma2BAO(3) = sigma2BAOFAR

  zeval(1) = zNEAR
  zeval(2) = zMID
  zeval(3) = zFAR

  zweight(1) = wNEAR
  zweight(2) = wMID
  zweight(3) = wFAR

  !! first read in everything needed from the CAMB output files.
  iopb = 0 !! check later if there was an error

  open(unit=tmp_file_unit,file='models/lrgdr7fiducialmodel_matterpowerzNEAR.dat',form='formatted',err=500, iostat=ios)
  read (tmp_file_unit,*,iostat=iopb) getabstransferscalefiddummy, omegakdummy,omegavdummy,wdummy
  do i = 1, num_matter_power
    read (tmp_file_unit,*,iostat=iopb) kval, plin, psmooth, rationwhalofit
    khvec(i) = kval
    ratio_power_nw_nl_fid(i,1) = rationwhalofit
  end do
  close(tmp_file_unit)

  open(unit=tmp_file_unit,file='models/lrgdr7fiducialmodel_matterpowerzMID.dat',form='formatted',err=500, iostat=ios)
  read (tmp_file_unit,*,iostat=iopb) getabstransferscalefiddummy,omegakdummy,omegavdummy,wdummy
  do i = 1, num_matter_power
    read (tmp_file_unit,*,iostat=iopb) kval, plin, psmooth, rationwhalofit
    if(abs(kval - khvec(i)) > 0.001) stop 'kvecs should be identical'
    ratio_power_nw_nl_fid(i,2) = rationwhalofit
  end do
  close(tmp_file_unit)

  open(unit=tmp_file_unit,file='models/lrgdr7fiducialmodel_matterpowerzFAR.dat',form='formatted',err=500,iostat=ios)
  read (tmp_file_unit,*,iostat=iopb) getabstransferscalefiddummy,omegakdummy,omegavdummy,wdummy
  do i = 1, num_matter_power
    read (tmp_file_unit,*,iostat=iopb) kval, plin, psmooth, rationwhalofit
    if(abs(kval - khvec(i)) > 0.001) stop 'kvecs should be identical'
    ratio_power_nw_nl_fid(i,3) = rationwhalofit
  end do
  close(tmp_file_unit)

  !! read in spectra of the model to be tested.
  open(unit=tmp_file_unit,file='models/lrgdr7model_matterpowerz0.dat',form='formatted',err=500,iostat=ios)
  read (tmp_file_unit,*,iostat=iopb) getabstransferscalez0, omegak,omegav,w
  close(tmp_file_unit)

  open(unit=tmp_file_unit,file='models/lrgdr7model_matterpowerzNEAR.dat',form='formatted',err=500,iostat=ios)
  read (tmp_file_unit,*,iostat=iopb) getabstransferscaleNEAR,omegakdummy,omegavdummy,wdummy
  if(abs(omegakdummy - omegak) > 0.001) stop 'omegak values should be identical'
  if(abs(omegavdummy - omegav) > 0.001) stop 'omegav values should be identical'
  if(abs(wdummy - w) > 0.001) stop 'w values should be identical'
  do i = 1, num_matter_power
    read (tmp_file_unit,*,iostat=iopb) kval, plin, psmooth, rationwhalofit
    if(abs(kval - khvec(i)) > 0.001) stop 'kvecs should be identical'
    outpower(i,1) = plin
    outpowernw(i,1) = psmooth
    outpowerrationwhalofit(i,1) = rationwhalofit
  end do
  close(tmp_file_unit)

  open(unit=tmp_file_unit,file='models/lrgdr7model_matterpowerzMID.dat',form='formatted',err=500,iostat=ios)
  read (tmp_file_unit,*,iostat=iopb) getabstransferscaleMID,omegakdummy,omegavdummy,wdummy
  if(abs(omegakdummy - omegak) > 0.001) stop 'omegak values should be identical'
  if(abs(omegavdummy - omegav) > 0.001) stop 'omegav values should be identical'
  if(abs(wdummy - w) > 0.001) stop 'w values should be identical'
  do i = 1, num_matter_power
    read (tmp_file_unit,*,iostat=iopb) kval, plin, psmooth, rationwhalofit
    if(abs(kval - khvec(i)) > 0.001) stop 'kvecs should be identical'
    outpower(i,2) = plin
    outpowernw(i,2) = psmooth
    outpowerrationwhalofit(i,2) = rationwhalofit
  end do
  close(tmp_file_unit)

  open(unit=tmp_file_unit,file='models/lrgdr7model_matterpowerzFAR.dat',form='formatted',err=500,iostat=ios)
  read (tmp_file_unit,*,iostat=iopb) getabstransferscaleFAR,omegakdummy,omegavdummy,wdummy
  if(abs(omegakdummy - omegak) > 0.001) stop 'omegak values should be identical'
  if(abs(omegavdummy - omegav) > 0.001) stop 'omegav values should be identical'
  if(abs(wdummy - w) > 0.001) stop 'w values should be identical'
  do i = 1, num_matter_power
    read (tmp_file_unit,*,iostat=iopb) kval, plin, psmooth, rationwhalofit
    if(abs(kval - khvec(i)) > 0.001) stop 'kvecs should be identical'
    outpower(i,3) = plin
    outpowernw(i,3) = psmooth
    outpowerrationwhalofit(i,3) = rationwhalofit
  end do
  close(tmp_file_unit)

500 if(ios .ne. 0) stop 'Unable to open file'
  if(iopb .ne. 0) stop 'Error reading model or fiducial theory files.'

  powerscaletoz0(1) = getabstransferscalez0**2.0d0/getabstransferscaleNEAR**2.0
  powerscaletoz0(2) = getabstransferscalez0**2.0d0/getabstransferscaleMID**2.0
  powerscaletoz0(3) = getabstransferscalez0**2.0d0/getabstransferscaleFAR**2.0

  do ik=1,num_matter_power
    kval = khvec(ik)
    halopowerlrgtheory(ik) = 0.
    do iz = 1, 3
      expval = exp(-kval**2*sigma2BAO(iz)*0.5)
      psmear = (outpower(ik,iz))*expval + (outpowernw(ik,iz))*(1.0-expval) 
      psmear = psmear*powerscaletoz0(iz)
      nlrat = outpowerrationwhalofit(ik,iz)/ratio_power_nw_nl_fid(ik,iz)
      call LRGtoICsmooth(kval,fidpolys)
      holdval(iz) = zweight(iz)*psmear*nlrat*fidpolys(iz)
      halopowerlrgtheory(ik) = halopowerlrgtheory(ik) + holdval(iz)
    end do 
  end do 
end subroutine fill_LRGTheory

   !! this function is just a copy of MatterPowerAt but with LRG theory put in instead of linear theory
   function LRGPowerAt(halopowerlrgtheory, kh)
     !get LRG matter power spectrum today at kh = k/h by interpolation from stored values
     real, intent(in) :: kh
     real, dimension(num_matter_power), intent(in) :: halopowerlrgtheory
     real LRGPowerAt
     real x, d
     integer i
  
     x = log(kh/matter_power_minkh) / matter_power_dlnkh
     if (x < 0 .or. x >= num_matter_power-1) then
        write (*,*) ' k/h out of bounds in MatterPowerAt (',kh,')'
        stop
     end if
     i = int(x)
     d = x - i
     LRGPowerAt = exp(log(halopowerlrgtheory(i+1))*(1-d) + log(halopowerlrgtheory(i+2))*d)
     !Just do linear interpolation in logs for now..
     !(since we already cublic-spline interpolated to get the stored values)
   end function

function LSS_LRG_mpklike(halopowerlrgtheory,mset,omegak,omegav,w) result(LnLike)
   real, dimension(num_matter_power), intent(in) :: halopowerlrgtheory
   Type (mpkdataset), intent(in) :: mset
   real(dl), intent(in) :: omegak,omegav,w
   real LnLike
   integer :: i
   real, dimension(:), allocatable :: mpk_raw, mpk_Pth, mpk_Pth_k, mpk_Pth_k2, k_scaled
   real, dimension(:), allocatable :: mpk_WPth, mpk_WPth_k, mpk_WPth_k2
   real :: covdat(mset%num_mpk_points_use), covth(mset%num_mpk_points_use), &
          & covth_k(mset%num_mpk_points_use), covth_k2(mset%num_mpk_points_use), &
          & covth_zerowin(mset%num_mpk_points_use)

   real, dimension(nptstot) :: chisq, chisqmarg  !! minus log likelihood list
   real :: minchisq,maxchisq,deltaL

   real(dl) :: a1val, a2val, zerowinsub
   real :: sumDD, sumDT, sumDT_k, sumDT_k2, sumTT,&
     &  sumTT_k, sumTT_k2, sumTT_k_k, sumTT_k_k2, sumTT_k2_k2, &
     &  sumDT_tot, sumTT_tot, &
     &  sumDT_zerowin, sumTT_zerowin, sumTT_k_zerowin, sumTT_k2_zerowin, sumTT_zerowin_zerowin

   real :: sumzerow_Pth, sumzerow_Pth_k, sumzerow_Pth_k2

   real :: a_scl      !LV_06 added for LRGDR4

   real(wp) :: temp1,temp2,temp3
   real :: temp4

   !! added for no marg
   integer :: myminchisqindx
   real :: currminchisq, currminchisqmarg, minchisqtheoryamp, chisqnonuis
   real :: minchisqtheoryampnonuis, minchisqtheoryampminnuis

   allocate(mpk_raw(mset%num_mpk_kbands_use) ,mpk_Pth(mset%num_mpk_kbands_use))
   allocate(mpk_Pth_k(mset%num_mpk_kbands_use) ,mpk_Pth_k2(mset%num_mpk_kbands_use))
   allocate(mpk_WPth(mset%num_mpk_points_use),mpk_WPth_k(mset%num_mpk_points_use),mpk_WPth_k2(mset%num_mpk_points_use))
   allocate(k_scaled(mset%num_mpk_kbands_use))!LV_06 added for LRGDR4
   chisq = 0

   call compute_scaling_factor(omegak,omegav,w,a_scl)
   !! applied in wrong direction in previous CosmoMC version!  simple fix:
   a_scl = 1.0d0/a_scl

   do i=1, mset%num_mpk_kbands_use
         k_scaled(i)=max(matter_power_minkh,a_scl*mset%mpk_k(i))
         mpk_raw(i)=LRGPowerAt(halopowerlrgtheory, k_scaled(i))/a_scl**3
   end do

   mpk_Pth = mpk_raw

   mpk_Pth_k = mpk_Pth*k_scaled
   mpk_Pth_k2 = mpk_Pth*k_scaled**2
   mpk_WPth = matmul(mset%mpk_W,mpk_Pth)
   mpk_WPth_k = matmul(mset%mpk_W,mpk_Pth_k)
   mpk_WPth_k2 = matmul(mset%mpk_W,mpk_Pth_k2)

   sumzerow_Pth = sum(mset%mpk_zerowindowfxn*mpk_Pth)/mset%mpk_zerowindowfxnsubtractdatnorm
   sumzerow_Pth_k = sum(mset%mpk_zerowindowfxn*mpk_Pth_k)/mset%mpk_zerowindowfxnsubtractdatnorm
   sumzerow_Pth_k2 = sum(mset%mpk_zerowindowfxn*mpk_Pth_k2)/mset%mpk_zerowindowfxnsubtractdatnorm

   covdat = matmul(mset%mpk_invcov,mset%mpk_P)
   covth = matmul(mset%mpk_invcov,mpk_WPth)
   covth_k = matmul(mset%mpk_invcov,mpk_WPth_k)
   covth_k2 = matmul(mset%mpk_invcov,mpk_WPth_k2)
   covth_zerowin = matmul(mset%mpk_invcov,mset%mpk_zerowindowfxnsubtractdat)

   sumDD = sum(mset%mpk_P*covdat)
   sumDT = sum(mset%mpk_P*covth)
   sumDT_k = sum(mset%mpk_P*covth_k)
   sumDT_k2 = sum(mset%mpk_P*covth_k2)
   sumDT_zerowin = sum(mset%mpk_P*covth_zerowin)

   sumTT = sum(mpk_WPth*covth)
   sumTT_k = sum(mpk_WPth*covth_k)
   sumTT_k2 = sum(mpk_WPth*covth_k2)
   sumTT_k_k = sum(mpk_WPth_k*covth_k)
   sumTT_k_k2 = sum(mpk_WPth_k*covth_k2)
   sumTT_k2_k2 = sum(mpk_WPth_k2*covth_k2)
   sumTT_zerowin = sum(mpk_WPth*covth_zerowin)
   sumTT_k_zerowin = sum(mpk_WPth_k*covth_zerowin)
   sumTT_k2_zerowin = sum(mpk_WPth_k2*covth_zerowin)
   sumTT_zerowin_zerowin = sum(mset%mpk_zerowindowfxnsubtractdat*covth_zerowin)

   currminchisq = 1000.0d0
   do i=1,nptstot
     a1val = a1list(i)
     a2val = a2list(i)
     zerowinsub = -(sumzerow_Pth + a1val*sumzerow_Pth_k + a2val*sumzerow_Pth_k2)

     sumDT_tot = sumDT + a1val*sumDT_k + a2val*sumDT_k2 + zerowinsub*sumDT_zerowin
     sumTT_tot = sumTT + a1val**2.0d0*sumTT_k_k + a2val**2.0d0*sumTT_k2_k2 + &
                 & zerowinsub**2.0d0*sumTT_zerowin_zerowin &
       & + 2.0d0*a1val*sumTT_k + 2.0d0*a2val*sumTT_k2 + 2.0d0*a1val*a2val*sumTT_k_k2 &
       & + 2.0d0*zerowinsub*sumTT_zerowin + 2.0d0*zerowinsub*a1val*sumTT_k_zerowin &
       & + 2.0d0*zerowinsub*a2val*sumTT_k2_zerowin
     minchisqtheoryamp = sumDT_tot/sumTT_tot
     chisq(i) = sumDD - 2.0d0*minchisqtheoryamp*sumDT_tot + minchisqtheoryamp**2.0d0*sumTT_tot
     chisqmarg(i) = sumDD - sumDT_tot**2.0d0/sumTT_tot &
         & + log(sumTT_tot) &
         & - 2.0*log(1.0d0 + erf(sumDT_tot/2.0d0/sqrt(sumTT_tot)))

     if(i == 1 .or. chisq(i) < currminchisq) then
        myminchisqindx = i
        currminchisq = chisq(i)
        currminchisqmarg = chisqmarg(i)
        minchisqtheoryampminnuis = minchisqtheoryamp
     end if
     if(i == int(nptstot/2)+1) then
        chisqnonuis = chisq(i)
        minchisqtheoryampnonuis = minchisqtheoryamp
        if(abs(a1val) > 0.001 .or. abs(a2val) > 0.001) then
           print *, 'ahhhh! violation!!', a1val, a2val
        end if
     end if

   end do

! numerically marginalize over a1,a2 now using values stored in chisq
   minchisq = minval(chisqmarg)
   maxchisq = maxval(chisqmarg)

   LnLike = sum(exp(-(chisqmarg-minchisq)/2.0d0)/(nptstot*1.0d0))
   if(LnLike == 0) then
     ! LnLike = LogZero
     stop 'LRG LnLike LogZero error.'
   else
     LnLike = -log(LnLike) + minchisq/2.0d0
   end if
 deltaL = (maxchisq - minchisq)*0.5

 !!interesting information about the fit, print if you're really curious.
 !print *, 'minnuis: ', a1list(myminchisqindx), a2list(myminchisqindx)
 !print *, 'minnuis chisqs: ', currminchisq, currminchisqmarg, currminchisq - currminchisqmarg
 !print *, 'margnuis: ', LnLike*2.0d0
 !print *, 'nonuis: ', chisqnonuis
 !print *, 'no/min theory amp: ',minchisqtheoryampnonuis,minchisqtheoryampminnuis
 !print *, 'a_scl = ',a_scl

   deallocate(mpk_raw, mpk_Pth)
   deallocate(mpk_Pth_k, mpk_Pth_k2)
   deallocate(mpk_WPth, mpk_WPth_k, mpk_WPth_k2)
   deallocate(k_scaled)

end function LSS_LRG_mpklike

! Read sybroutines copied from settings.f90
subroutine ReadVectorBR(aname,vec,n)
   character(LEN=*), intent(IN) :: aname
   integer, intent(in) :: n
   real, intent(out) :: vec(n)
   integer j, ios
   open(unit=tmp_file_unit,file=aname,form='formatted',err=200,iostat=ios)
   do j=1,n 
      read (tmp_file_unit,*, end = 200) vec(j)
   end do

   close(tmp_file_unit)
   return

200 write(*,*) 'vector file '//trim(aname)//' is the wrong size'
end subroutine ReadVectorBR

subroutine ReadMatrixBR(aname, mat, m,n)

   character(LEN=*), intent(IN) :: aname
   integer, intent(in) :: m,n
   real, intent(out) :: mat(m,n)
   integer j,k,ios
   real tmp

   open(unit=tmp_file_unit,file=aname,form='formatted',err=200,iostat=ios)

   do j=1,m
      read (tmp_file_unit,*, end = 200, err=100) mat(j,1:n)
   end do
   goto 120

100 rewind(tmp_file_unit)  !Try other possible format
   do j=1,m
    do k=1,n
      read (tmp_file_unit,*, end = 200) mat(j,k)
    end do
   end do

120 read (tmp_file_unit,*, err = 150, end =150) tmp
   goto 200

150 close(tmp_file_unit)
    return

 200 write (*,*) 'matrix file '//trim(aname)//' is the wrong size'
     stop

end subroutine ReadMatrixBR

!-----------------------------------------------------------------------------
!LV added to include lrg DR4

subroutine compute_scaling_factor(Ok,Ol,w,a)
  ! a = dV for z=0.35 relative to its value for flat Om=0.25 model.
  ! This is the factor by which the P(k) measurement would shift
  ! sideways relative to what we got for this fiducial flat model.
  ! * a = (a_angular**2 * a_radial)**(1/3)
  ! * a_angular = comoving distance to z=0.35 in Mpc/h relative to its value for flat Om=0.25 model
  !     dA = (c/H)*eta = (2997.92458 Mpc/h)*eta, so we care only about
  !     eta scaling, not h scaling.
  !     For flat w=-1 models, a ~ (Om/0.25)**(-0.065)
  !     For the LRG mean redshift z=0.35, the power law fit
  !    dA(z,Om= 0.3253 (Om/0.25)^{-0.065}c H_0^{-1} is quite good within
  !    our range of interest,
  !     accurate to within about 0.1% for 0.2<Om<0.3.
  ! * a_radial = 1/H(z) relative to its value for flat Om=0.25 model
  implicit none
  real(mpk_d) Or, Om, Ok, Ol, w, Ok0, Om0, Ol0, w0, z, eta, eta0, Hrelinv, Hrelinv0, tmp
  real(mpk_d) a_radial, a_angular
  real a
  !Or= 0.0000415996*(T_0/2.726)**4 / h**2
  Or= 0! Radiation density totally negligible at  z < 0.35
  Om= 1-Ok-Ol-Or
  !!!z  = 0.35  !!edited by Beth 21-11-08: change to zeff of Will's LRG sample.
  z = zeffDR7
  Hrelinv= 1/sqrt(Ol*(1+z)**(3*(1+w)) + Ok*(1+z)**2 + Om*(1+z)**3 + Or*(1+z)**4)
!  write(*,*) Ok,Ol,w
call compute_z_eta(Or,Ok,Ol,w,z,eta)
  tmp = sqrt(abs(Ok))
  if (Ok.lt.-1.d-6) eta = sin(tmp*eta)/tmp
  if (Ok.gt.1d-6)   eta = (exp(tmp*eta)-exp(-tmp*eta))/(2*tmp) ! sinh(tmp*eta)/tmp
  Ok0= 0
  Ol0= 0.75
  w0= -1
  Om0= 1-Ok0-Ol0-Or
  call compute_z_eta(Or,Ok0,Ol0,w0,z,eta0)
  Hrelinv0= 1/sqrt(Ol0*(1+z)**(3*(1+w0)) + Ok0*(1+z)**2 + Om0*(1+z)**3 + Or*(1+z)**4)
  !a_angular = (Om/0.25)**(-0.065) * (-w*Otot)**0.14 ! Approximation based on Taylor expansion
  a_angular = eta/eta0
  a_radial= Hrelinv/Hrelinv0
  a=  (a_angular**2 * a_radial)**(1/3.d0)
  !write(*,'(9f10.5)') Ok,Ol,w,a,a_radial,a_angular,(Om/0.25)**(-0.065) * (-w*(1-Ok))**0.14
  !write(*,'(9f10.5)') Ok,Ol,w,a,a_radial**(2/3.d0),a_angular**(4/3.d0),((Om/0.25)**(-0.065) * (-w*(1-Ok))**0.14)**(4/3.d0)
end subroutine compute_scaling_factor

subroutine eta_demo
  implicit none
  real(mpk_d) Or, Ok, Ol, w, h, z, eta
  h  = 0.7
  Ok = 0
  Ol = 0.7
  Or = 0.0000416/h**2
  w  = -1
  z  = 1090
  call compute_z_eta(Or,Ok,Ol,w,z,eta)
!  print *,'eta.............',eta
!  print *,'dlss in Gpc.....',(2.99792458/h)*eta
end subroutine eta_demo

!INTERFACE
logical function nobigbang2(Ok,Ol,w)
  ! Test if we're in the forbidden zone where the integrand blows up
  ! (where's H^2 < 0 for some a between 0 and 1).
  ! The function f(a) = Omega_m + Omega_k*a + Omega_l*a**(-3*w)
  ! can have at most one local minimum, at (Ok/(3*w*Ol))**(-1/(1+3*w)),
  ! so simply check if f(a)<0 there or at the endpoints a=0, a=1.
  ! g(0) = Omega_m - Omega_l*a**(-3*w) < 0 if w > 0 & Omega_k > 1
  !                                     or if w = 0 & Omega_l < 1
  ! g(1) = Omega_m + Omega_k + Omega_l = 1 > 0
  implicit none
  real(mpk_d) Ok, Ol, w, Om, tmp, a, epsilon
  integer failure
  failure = 0
  epsilon = 0
  !epsilon = 0.04  ! Numerical integration fails even before H^2 goes negative.
  Om = 1.d0 - Ok - Ol
  if (w*Ol.ne.0) then
     tmp = Ok/(3*w*Ol)
     if ((tmp.gt.0).and.(1+3*w.ne.0)) then ! f'(0)=0 for some a>0
        a = tmp**(-1/(1+3*w))
        if (a.lt.1) then
           if (Om + Ok*a + Ol*a**(-3*w).lt.epsilon) failure = 1
        end if
     end if
  end if
  if ((w.eq.0).and.(Ok.gt.1)) failure = 2
  if ((w.gt.0).and.(Ol.lt.0)) failure = 3
  nobigbang2 = (failure.gt.0)
  if (failure.gt.0) print *,'Big Bang failure mode ',failure
  return
end function nobigbang2
!END INTERFACE

real(mpk_d) function eta_integrand(a)
  implicit none
  real(mpk_d) Or, Ok, Ox, w
  common/eta/Or, Ok, Ox, w
  real(mpk_d) a, Om
  ! eta = int (H0/H)dz = int (H0/H)(1+z)dln(1+z) = int (H0/H)/a dlna = int (H0/H)/a^2 da =
  ! Integrand = (H0/H)/a^2
  ! (H/H0)**2 = Ox*a**(-3*(1+w)) + Ok/a**2 + Om/a**3 + Or/a**4
  if (a.eq.0.d0) then
     eta_integrand = 0.d0
  else
     Om = 1.d0 - Or - Ok - Ox
     eta_integrand = 1.d0/sqrt(Ox*a**(1-3*w) + Ok*a**2 + Om*a + Or)
  end if
  return
end function eta_integrand

subroutine eta_z_integral(Omega_r,Omega_k,Omega_x,w_eos,z,eta)
  ! Computes eta as a function
  ! of the curvature Omega_k, the dark energy density Omega_x
  ! and its equation of state w.
  implicit none
  real(mpk_d) Or, Ok, Ox, w
  common/eta/Or, Ok, Ox, w
  real(mpk_d) Omega_r, Omega_k,Omega_x,w_eos, z, eta, epsabs, epsrel, amin, amax!, eta_integrand
  Or = Omega_r
  Ok = Omega_k
  Ox = Omega_x
  w  = w_eos
  epsabs  = 0
  epsrel  = 1.d-10
  amin= 1/(1+z)
  amax= 1
  call qromb2(eta_integrand,amin,amax,epsabs,epsrel,eta)
  return
end subroutine eta_z_integral

subroutine compute_z_eta(Or,Ok,Ox,w,z,eta)
  ! Computes the conformal distance eta(z)
  implicit none
  real(mpk_d) Or, Ok, Ox, w, z, eta
!  logical nobigbang2
  if (nobigbang2(Ok,Ox,w)) then
     print *,'No big bang, so eta undefined if z>zmax.'
     eta = 99
  else
     call eta_z_integral(Or,Ok,Ox,w,z,eta)
     ! print *,'Or, Ok, Ox, w, z, H_0 t_0...',Or, Ok, Ox, w, eta
  end if
  return
end subroutine compute_z_eta

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!! num rec routines
!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE qromb2(func,a,b,epsabs,epsrel,ss)
! The numerical recipes routine, but modified so that is decides
! it's done when either the relative OR the absolute accuracy has been attained.
! The old version used relative errors only, so it always failed when
! when the integrand was near zero.
! epsabs = epsrel = 1e-6 are canonical choices.
  INTEGER JMAX,JMAXP,K,KM
  real(mpk_d) a,b,func,ss,epsabs,epsrel
  EXTERNAL func
  PARAMETER (JMAX=20, JMAXP=JMAX+1, K=5, KM=K-1)
                                !    USES polint,trapzd
  INTEGER j
  real(mpk_d) dss,h(JMAXP),s(JMAXP)
  h(1)=1.d0
  do j=1,JMAX
     call trapzd(func,a,b,s(j),j)
     if (j.ge.K) then
        call polint(h(j-KM),s(j-KM),K,0.d0,ss,dss)
        if (abs(dss).le.epsrel*abs(ss)) return
        if (abs(dss).le.epsabs) return
     endif
     s(j+1)=s(j)
     h(j+1)=0.25d0*h(j)
  ENDDO
  print *,'Too many steps in qromb'

  RETURN
END SUBROUTINE qromb2

SUBROUTINE polint(xa,ya,n,x,y,dy) ! From Numerical Recipes
  INTEGER n,NMAX
  real(mpk_d) dy,x,y,xa(n),ya(n)
  PARAMETER (NMAX=10)
  INTEGER i,m,ns
  real(mpk_d) den,dif,dift,ho,hp,w,c(NMAX),d(NMAX)
  ns=1
  dif=abs(x-xa(1))
  do  i=1,n
     dift=abs(x-xa(i))
     if (dift.lt.dif) then
        ns=i
        dif=dift
     endif
     c(i)=ya(i)
     d(i)=ya(i)
  enddo
  y=ya(ns)
  ns=ns-1
  do  m=1,n-1
     do  i=1,n-m
        ho=xa(i)-x
        hp=xa(i+m)-x
        w=c(i+1)-d(i)
        den=ho-hp
        if(den.eq.0.) then
           print*, 'failure in polint'
           stop
        endif
        den=w/den
        d(i)=hp*den
        c(i)=ho*den
     enddo
     if (2*ns.lt.n-m)then
        dy=c(ns+1)
     else
        dy=d(ns)
        ns=ns-1
     endif
     y=y+dy
  enddo
  return
END SUBROUTINE polint

SUBROUTINE trapzd(func,a,b,s,n) ! From Numerical Recipes
  INTEGER n
  real(mpk_d) a,b,s,func
  EXTERNAL func
  INTEGER it,j
  real(mpk_d) del,sum,tnm,x
  if (n.eq.1) then
     s=0.5*(b-a)*(func(a)+func(b))
  else
     it=2**(n-2)
     tnm=it
     del=(b-a)/tnm
     x=a+0.5*del
     sum=0.
     do  j=1,it
        sum=sum+func(x)
        x=x+del
     enddo
     s=0.5*(s+(b-a)*sum/tnm)
  endif
  return
END SUBROUTINE trapzd

!! functions added for nuisance parameter space checks.
  function a2maxpos(a1val) result(a2max)
    real(dl), intent(in) :: a1val
    real(dl) a2max
    a2max = -1.0d0
    if (a1val <= min(s1/k1,s2/k2)) then
      a2max = min(s1/k1**2 - a1val/k1, s2/k2**2 - a1val/k2)
    end if
  end function a2maxpos

  function a2min1pos(a1val) result(a2min1)
    real(dl), intent(in) :: a1val
    real(dl) a2min1
    a2min1 = 0.0d0
    if(a1val <= 0.0d0) then
      a2min1 = max(-s1/k1**2 - a1val/k1, -s2/k2**2 - a1val/k2, 0.0d0)
    end if
  end function a2min1pos

  function a2min2pos(a1val) result(a2min2)
    real(dl), intent(in) :: a1val
    real(dl) a2min2
    a2min2 = 0.0d0
    if(abs(a1val) >= 2.0d0*s1/k1 .and. a1val <= 0.0d0)  then
      a2min2 = a1val**2/s1*0.25d0
    end if
  end function a2min2pos

  function a2min3pos(a1val) result(a2min3)
    real(dl), intent(in) :: a1val
    real(dl) a2min3
    a2min3 = 0.0d0
    if(abs(a1val) >= 2.0d0*s2/k2 .and. a1val <= 0.0d0)  then
      a2min3 = a1val**2/s2*0.25d0
    end if
  end function a2min3pos

  function a2minfinalpos(a1val) result(a2minpos)
    real(dl), intent(in) :: a1val
    real(dl) a2minpos
    a2minpos = max(a2min1pos(a1val),a2min2pos(a1val),a2min3pos(a1val))
  end function a2minfinalpos

  function a2minneg(a1val) result(a2min)
    real(dl), intent(in) :: a1val
    real(dl) a2min
    if (a1val >= max(-s1/k1,-s2/k2)) then
      a2min = max(-s1/k1**2 - a1val/k1, -s2/k2**2 - a1val/k2)
    else
      a2min = 1.0d0
    end if
  end function a2minneg

  function a2max1neg(a1val) result(a2max1)
    real(dl), intent(in) :: a1val
    real(dl) a2max1
    if(a1val >= 0.0d0) then 
      a2max1 = min(s1/k1**2 - a1val/k1, s2/k2**2 - a1val/k2, 0.0d0)
    else
      a2max1 = 0.0d0
    end if
  end function a2max1neg
  
  function a2max2neg(a1val) result(a2max2)
    real(dl), intent(in) :: a1val
    real(dl) a2max2
    a2max2 = 0.0d0
    if(abs(a1val) >= 2.0d0*s1/k1 .and. a1val >= 0.0d0)  then
      a2max2 = -a1val**2/s1*0.25d0
    end if
  end function a2max2neg
  
  function a2max3neg(a1val) result(a2max3)
    real(dl), intent(in) :: a1val
    real(dl) a2max3
    a2max3 = 0.0d0
    if(abs(a1val) >= 2.0d0*s2/k2 .and. a1val >= 0.0d0)  then
      a2max3 = -a1val**2/s2*0.25d0
    end if
  end function a2max3neg
  
  function a2maxfinalneg(a1val) result(a2maxneg)
    real(dl), intent(in) :: a1val
    real(dl) a2maxneg
    a2maxneg = min(a2max1neg(a1val),a2max2neg(a1val),a2max3neg(a1val))
  end function a2maxfinalneg


function testa1a2(a1val, a2val) result(testresult)
    real(dl), intent(in) :: a1val,a2val
    logical :: testresult

    real(dl) :: kext, diffval
    testresult = .true.

    ! check if there's an extremum; either a1val or a2val has to be negative, not both
    kext = -a1val/2.0d0/a2val
    diffval = abs(a1val*kext + a2val*kext**2)
    if(kext > 0.0d0 .and. kext <= k1 .and. diffval > s1) testresult = .false.
    if(kext > 0.0d0 .and. kext <= k2 .and. diffval > s2) testresult = .false.

    if (abs(a1val*k1 + a2val*k1**2) > s1) testresult = .false.
    if (abs(a1val*k2 + a2val*k2**2) > s2) testresult = .false.

end function testa1a2

! code for erfc copied from http://nn-online.org/code/erfc/erfc11.f90

      function erf(x) result(erfval)
      use precision

      implicit none
      real(wp), intent(in)  :: x
      real(wp)              :: erfval

      erfval = 1.0_wp - erfc(x)
      end function erf

      function erfc11(x) result(erfc)

      use precision

      implicit none
      real(wp), intent(in)  :: x
      real(wp)              :: erfc
      real(wp)              :: ax,t,num,den
      integer               :: i
      real(wp), parameter   :: p(0:5) = (/ &
                                      0.0000000297886562639399288862e+08_wp, &
                                      0.0000007409740605964741794425e+07_wp, &
                                      0.0000061602098531096305440906e+06_wp, &
                                      0.0005019049726784267463450058e+04_wp, &
                                      0.1275366644729965952479585264e+01_wp, &
                                      0.5641895835477550741253201704e+00_wp  /)
      real(wp), parameter   :: q(0:6) = (/ &
                                      0.0000000033690752069827527677e+09_wp, &
                                      0.0000009608965327192787870698e+07_wp, &
                                      0.0001708144074746600431571095e+05_wp, &
                                      0.0120489519278551290360340491e+03_wp, &
                                      0.9396034016235054150430579648e+01_wp, &
                                      0.2260528520767326969591866945e+01_wp, &
                                      0.1000000000000000000000000000e+01_wp  /)
      real(wp), parameter   :: c1(20) = (/ &
                                   +0.106073416421769980345174155056e+01_wp, &
                                   -0.042582445804381043569204735291e+01_wp, &
                                   +0.004955262679620434040357683080e+01_wp, &
                                   +0.000449293488768382749558001242e+01_wp, &
                                   -0.000129194104658496953494224761e+01_wp, &
                                   -0.000001836389292149396270416979e+01_wp, &
                                   +0.000002211114704099526291538556e+01_wp, &
                                   -0.523337485234257134673693179020e-06_wp, &
                                   -0.278184788833537885382530989578e-06_wp, &
                                   +0.141158092748813114560316684249e-07_wp, &
                                   +0.272571296330561699984539141865e-08_wp, &
                                   -0.206343904872070629406401492476e-09_wp, &
                                   -0.214273991996785367924201401812e-10_wp, &
                                   +0.222990255539358204580285098119e-11_wp, &
                                   +0.136250074650698280575807934155e-12_wp, &
                                   -0.195144010922293091898995913038e-13_wp, &
                                   -0.685627169231704599442806370690e-15_wp, &
                                   +0.144506492869699938239521607493e-15_wp, &
                                   +0.245935306460536488037576200030e-17_wp, &
                                   -0.929599561220523396007359328540e-18_wp  /)
      real(wp), parameter   :: c2(25) = (/ &
                                   +0.044045832024338111077637466616e+01_wp, &
                                   -0.143958836762168335790826895326e+00_wp, &
                                   +0.044786499817939267247056666937e+00_wp, &
                                   -0.013343124200271211203618353102e+00_wp, &
                                   +0.003824682739750469767692372556e+00_wp, &
                                   -0.001058699227195126547306482530e+00_wp, &
                                   +0.000283859419210073742736310108e+00_wp, &
                                   -0.000073906170662206760483959432e+00_wp, &
                                   +0.000018725312521489179015872934e+00_wp, &
                                   -0.462530981164919445131297264430e-05_wp, &
                                   +0.111558657244432857487884006422e-05_wp, &
                                   -0.263098662650834130067808832725e-06_wp, &
                                   +0.607462122724551777372119408710e-07_wp, &
                                   -0.137460865539865444777251011793e-07_wp, &
                                   +0.305157051905475145520096717210e-08_wp, &
                                   -0.665174789720310713757307724790e-09_wp, &
                                   +0.142483346273207784489792999706e-09_wp, &
                                   -0.300141127395323902092018744545e-10_wp, &
                                   +0.622171792645348091472914001250e-11_wp, &
                                   -0.126994639225668496876152836555e-11_wp, &
                                   +0.255385883033257575402681845385e-12_wp, &
                                   -0.506258237507038698392265499770e-13_wp, &
                                   +0.989705409478327321641264227110e-14_wp, &
                                   -0.190685978789192181051961024995e-14_wp, &
                                   +0.350826648032737849245113757340e-15_wp  /)
      real(wp), parameter :: c3(20) = (/ &
                                   +0.111684990123545698684297865808e+01_wp, &
                                   +0.003736240359381998520654927536e+00_wp, &
                                   -0.000916623948045470238763619870e+00_wp, &
                                   +0.000199094325044940833965078819e+00_wp, &
                                   -0.000040276384918650072591781859e+00_wp, &
                                   +0.776515264697061049477127605790e-05_wp, &
                                   -0.144464794206689070402099225301e-05_wp, &
                                   +0.261311930343463958393485241947e-06_wp, &
                                   -0.461833026634844152345304095560e-07_wp, &
                                   +0.800253111512943601598732144340e-08_wp, &
                                   -0.136291114862793031395712122089e-08_wp, &
                                   +0.228570483090160869607683087722e-09_wp, &
                                   -0.378022521563251805044056974560e-10_wp, &
                                   +0.617253683874528285729910462130e-11_wp, &
                                   -0.996019290955316888445830597430e-12_wp, &
                                   +0.158953143706980770269506726000e-12_wp, &
                                   -0.251045971047162509999527428316e-13_wp, &
                                   +0.392607828989125810013581287560e-14_wp, &
                                   -0.607970619384160374392535453420e-15_wp, &
                                   +0.912600607264794717315507477670e-16_wp  /)


      ax = abs(x)
      if (ax == 0.0_wp) then
          stop 'GSL_ERFC: zero argument'
      elseif (ax <= 1.0_wp) then
          t = 2.0_wp*ax - 1.0_wp
          erfc = evalcs(t,c1)
      elseif (ax <= 5.0_wp) then
          t = 0.5_wp*(ax-3.0_wp)
          erfc = evalcs(t,c2) * exp(-x**2)
      elseif (ax < 10.0_wp) then
          t = (2.0_wp*x - 15.0_wp)/5.0_wp
          erfc = evalcs(t,c3) * exp(-x**2)/x
      else
          num = p(5)
          do i=4,0,-1
              num = x*num + p(i)
          enddo
          den = q(6)
          do i=5,0,-1
              den = x*den + q(i)
          enddo
          erfc = num/den * exp(-x**2)
      endif

      if (x < 0.0_wp) erfc = 2.0_wp - erfc

      contains

!*******************************************************************************

      function evalcs(x,cs)

      implicit none
      real(wp), intent(in)  :: x,cs(:)
      real(wp)              :: evalcs
      real(wp)              :: b0,b1,b2,twox
      integer               :: i


      b0 = 0.0_wp
      b1 = 0.0_wp
      twox = 2.0_wp*x
      do i=size(cs),1,-1
          b2 = b1
          b1 = b0
          b0 = twox*b1 - b2 + cs(i)
      enddo
      evalcs = (b0-b2)/2.0_wp

      end function evalcs

!*******************************************************************************

      end function erfc11


end module lrgutils
