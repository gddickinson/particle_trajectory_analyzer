# featuresCalcsNew (adapted from featureCalcs6)
#
# J. Alfredo Freites
# jfreites@uci.edu
# 
# a collection of R functions to compute geometric features of a 2-D trajectory
# the 2-D trajectory is expected to be implemented as a 2-column matrix with each spatial component in a column
# trajectories sampled at constant rate dt (time/frame) are expected to be fully specified in the matrix
# meaning that  a trjectory sampled N-times should be specified as an N-by-2 matrix
# missing points (coordinates) are expected to have both spatial coordinates indicated as NA
# sample averages are computed after removing NAs
#
# the features selection matches those reported in Ly et al 2024
#

#vector length: vector length
veclen<-function(v) {
  return(sqrt(sum(v^2)))
}

#getStesp: goes from a pointwise representation of N-point trajectory x(t)={x1,x2,...,xN} where xi is a 2-D position vector
#	   to a stepwise representation dx(t)={x2-x1,x3-x2,...,xN-xN-1}

getSteps<-function(xy){
  return(apply(xy,2,diff))
}

#getStepsNorm: compute steps and scale by the corresponding RMS
getStepsNorm<-function(xy){
  stepe<-apply(xy,2,diff)
  stepeMean<-apply(stepe,2,function(z){sqrt(mean(z^2,na.rm=T))})
  return(scale(stepe,center=F,scale=stepeMean))
}

#getRg: trajectory radius of gyration
getRg<-function(xy) {
  avg<-colMeans(xy,na.rm=T)
  avg2<-colMeans(xy^2,na.rm=T)
  rg<-sqrt(sum(avg2-avg^2))
  return(rg)
}

#getMeanStepLength:
getMeanStepLength<-function(xy,stepe=NULL){
  if(is.null(stepe))
      stepe<-apply(xy,2,diff)
  slen<-mean(apply(stepe,1,function(z){sqrt(sum(z^2,na.rm=T))}))
  return(slen)
}

#getStepLen: the lengths of each trajectory step {|x2-x1|,|x3-x2|,...,|xN-xN-1|}
getStepLen<-function(xy,stepe=NULL){
  if(is.null(stepe))
      stepe<-apply(xy,2,diff)
  slen<-apply(stepe,1,function(z){sum(z^2)})
  return(slen)
}

#getStepLenDelta compute step lengths but with a specified lag (e.g. different from 1)  betwen points
getStepLenDelta<-function(xy,delta) {
	if(nrow(xy)>delta-1) {
	stepe<-apply(xy,2,diff,lag=delta)
  slen<-apply(stepe,1,function(z){sqrt(sum(z^2))})
	}
  return(slen)
}

#getStepLenDeltaNorm same as above but scaled by the RMS
getStepLenDeltaNorm<-function(xy,delta) {
        if(nrow(xy)>delta-1) {
        stepe<-apply(xy,2,diff,lag=delta)
	stepe<-apply(stepe,2,function(z)z/sqrt(mean(z^2,na.rm=T)))
  slen<-apply(stepe,1,function(z){sqrt(sum(z^2))})
        }
  return(slen)
}

#compute Rg scaled as in Golan and Sherman Nat Comm 2017    
getsRg<-function(xy) {
        Rg<-getRg(xy)
        meanS<-getMeanStepLength(xy)
        sRg<-sqrt(pi/2)*Rg/meanS
        return(sRg)
}
