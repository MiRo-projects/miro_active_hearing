import numpy as np
from numpy.fft import rfft, irfft, fft, ifft
import matplotlib.pyplot as plt

def gcc(d0, d1):
	pad1 = np.zeros(len(d0))
	pad2 = np.zeros(len(d1))
	s1 = np.hstack([d0[-113:],pad1])
	s2 = np.hstack([pad2,d1[-113:]])
	f_s1 = fft(s1)
	f_s2 = fft(s2)
	f_s2c = np.conj(f_s2)
	f_s = f_s1 * f_s2c
	#denom = abs(f_s)
	#f_s = f_s/denom
	Xgcorr = np.abs(ifft(f_s,40))[:]
        
	return Xgcorr
	#return Xcorr


def gccphat(d0, d1):
	pad1 = np.zeros(len(d0))
	pad2 = np.zeros(len(d1))
	s1 = np.hstack([d0[-20:],pad1])
	s2 = np.hstack([pad2,d1[-20:]])
	f_s1 = fft(s1)
	f_s2 = fft(s2)
	f_s2c = np.conj(f_s2)
	f_s = f_s1 * f_s2c
	denom = abs(f_s)
	f_s = f_s/denom
	Xgcorr = np.abs(ifft(f_s))[:]
        
	return Xgcorr
	#return Xcorr

def main():
    x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    sig = [0,2,0,1,2,3,2,1,0,0,0,3,4,5,0,0,0,1,2,0]
    refsig = [0,1,2,3,2,1,0,0,0,3,4,5,0,0,0,1,2,0,0,2] # refsig has 2 lag delay with sig

    # three method to calculate cross-correlation
    xgcorr = gcc(sig,refsig)
    xgcorr2 = gccphat(sig,refsig)
    xco = np.correlate(sig,refsig, mode='same')
    # choose the sigal from the center and has the same length with original signal
    flatted = xgcorr[10:30] 

    # plot two signal
    plt.subplot(2,1,1)
    plt.plot(x,refsig)
    plt.title("signal 1")
    plt.subplot(2,1,2)
    plt.plot(x,sig)
    plt.title("signal 2")
    plt.show()

    #plot Generalized Cross Correlation vs Cross-correlation
    plt.subplot(2,1,1)
    plt.plot(np.linspace(0,19,20),flatted)
    plt.title("GCC")
    plt.subplot(2,1,2)
    plt.plot(x,xco)
    plt.title("CC")
    plt.show()

    #plot Generalized Cross Correlation vs GCC-PHAT
    plt.subplot(2,1,1)
    plt.plot(np.linspace(0,19,len(xgcorr)),xgcorr)
    plt.title("GCC")
    plt.subplot(2,1,2)
    plt.plot(np.linspace(0,19,len(xgcorr2)),xgcorr2)
    plt.title("GCC-PHAT")
    plt.show()



if __name__ == "__main__":
    main()
