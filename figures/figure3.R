args <- commandArgs(trailingOnly=TRUE)
dat <- read.delim(args[1],check.names=FALSE,stringsAsFactors=FALSE)
grDevices::cairo_pdf(args[2], width=170/25.4,height=174/25.4,family='Arial',pointsize=8,onefile=FALSE)
layout(matrix(1:3,ncol=1))
titles <- c('A  Complete-case top-1 accuracy','B  Evidence override','C  Failure-inclusive top-1 accuracy')
cols <- c('#B94A23','#4256B2','#157248')
for(i in 1:3){
  d <- dat[dat$panel==LETTERS[i],]
  par(mai=c(0.40,1.28,0.28,1.80),cex=1.0,ps=8,mgp=c(1.4,0.35,0),tcl=-0.25,xaxs='i',yaxs='i',las=1,cex.axis=0.95)
  plot(NA,xlim=c(-15,40),ylim=c(0.45,4.9),xlab='',ylab='',axes=FALSE)
  abline(v=seq(-10,40,10),col='#E1E1E1',lwd=0.55)
  abline(v=0,col='#555555',lwd=0.8,lty=2)
  y <- 4:1
  segments(d$lo,y,d$hi,y,col=cols[i],lwd=1.35)
  segments(d$lo,y-0.09,d$lo,y+0.09,col=cols[i],lwd=1.0)
  segments(d$hi,y-0.09,d$hi,y+0.09,col=cols[i],lwd=1.0)
  points(d$rd,y,pch=15,cex=0.85,col=cols[i])
  axis(1,at=seq(-10,40,10),lwd=0.7,lwd.ticks=0.7)
  axis(2,at=y,labels=d$strategy,tick=FALSE,hadj=1,cex.axis=1.0,line=0.25)
  # Estimate labels have their own right-hand column, outside the confidence intervals.
  labels <- sprintf('%+.2f (%+.2f to %+.2f)',d$rd,d$lo,d$hi)
  text(42.5,y,labels,adj=0,xpd=NA,cex=0.95)
  text(42.5,4.8,'RD, pp (95% CI)',adj=0,xpd=NA,font=2,cex=0.95)
  mtext(titles[i],side=3,line=0.6,adj=0,font=2,cex=1.12)
  mtext('Risk difference (percentage points)',side=1,line=1.35,cex=0.95)
}
dev.off()
cat('Created three-panel figure from 12 reported estimates\n')
