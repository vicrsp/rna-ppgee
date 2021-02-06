


require('mlbench')
treinaRBF<-function(xin,yin,p)
{

  N<-dim(xin)[1]
  n<-dim(xin)[2]
  xin<-as.matrix(xin)
  yin<-as.matrix(yin)

  pdfnvar<-function(x,m,K,n) ((1/(sqrt((2*pi)^n*(det(K)))))*exp(-0.5*(t(x-m) %*% (solve(K)) %*% (x-m))))

  xclust<-kmeans(xin,p)

  m<-as.matrix(xclust$centers)
  covlist<-list()


  for(i in 1:p)
  {
    ici<-which(xclust$cluster==i)
    xci<-xin[ici,]
    if(n==1)
      covi<-var(xci)
    else
      covi<-cov(xci)
    
    covlist[[i]]<-covi
  }

  H<-matrix(nrow=N,ncol=p)
  for(j in 1:N)
  {
    for(i in 1:p)
    {
      mi<-m[i,]
      covi<-covlist[i]
      
      covi
      matrix(unlist(covlist[i]),ncol=n,byrow=T)


      covi<-matrix(unlist(covlist[i]),ncol=n,byrow=T)+0.001*diag(n)
      H[j,i]<-pdfnvar(xin[j,],mi,covi,n)
    }
  }

  Haug<-cbind(1,H)
  W<-(solve(t(Haug)%*%Haug)%*%t(Haug))%*%yin

  return(list(m,covlist,W,H))
}

data = mlbench.2dnormals(200)
treinaRBF(data$x, data$classes,5)