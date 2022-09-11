#	@section COPYRIGHT
#	Copyright (C) 2021 Consequential Robotics Ltd
#	
#	@section AUTHOR
#	Consequential Robotics http://consequentialrobotics.com
#	
#	@section LICENSE
#	For a full copy of the license agreement, and a complete
#	definition of "The Software", see LICENSE in the MDK root
#	directory.
#	
#	Subject to the terms of this Agreement, Consequential
#	Robotics grants to you a limited, non-exclusive, non-
#	transferable license, without right to sub-license, to use
#	"The Software" in accordance with this Agreement and any
#	other written agreement with Consequential Robotics.
#	Consequential Robotics does not transfer the title of "The
#	Software" to you; the license granted to you is not a sale.
#	This agreement is a binding legal agreement between
#	Consequential Robotics and the purchasers or users of "The
#	Software".
#	
#	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
#	KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
#	WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
#	PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS
#	OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
#	OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#	OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
#	SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#	

import numpy as np
import miro2 as miro
from numpy.fft import rfft, irfft, fft, ifft

# create kinematic chain object with default (calibration) configuration
# of joints (and zeroed pose of FOOT in WORLD)
KC = miro.lib.kc_interf.kc_miro()

SAMP_PER_BLOCK=500
SAMP_BUFFERED=SAMP_PER_BLOCK*2
RAW_MAGNITUDE_THRESH = 0.01 # normalized; audio event processing skipped unless over thresh

SPEED_OF_SOUND = 343.0 # m/s
INTER_EAR_DISTANCE = 0.104 # metres
MIC_SAMPLE_RATE = 20000 # audio sample rate
INTER_EAR_LAG = INTER_EAR_DISTANCE / SPEED_OF_SOUND * MIC_SAMPLE_RATE
EAR_TAIL_DISTANCE = 0.121 # metres
EAR_TAIL_LAG = EAR_TAIL_DISTANCE / SPEED_OF_SOUND * MIC_SAMPLE_RATE

ASSUMED_SOUND_SOURCE_HEIGHT = 0.15 # metres
ASSUMED_SOUND_SOURCE_RANGE = 0.5 # metres


class DetectAudioEvent():

	def __init__(self, data):

		self.azim = data[0]
		self.elev = data[1]
		self.level = data[2]
		self.ang = data[3]

class DetectAudioEventHead():
	# Cartesian in the HEAD frame
	def __init__(self, data):

		self.x = data[0]
		self.y = data[1]
		self.z = data[2]


class DetectAudioEngine():

	def __init__(self):

		# state
		self.n = SAMP_PER_BLOCK
		self.buf = None
		self.buf_head = None
		self.buf_abs = None
		self.buf_tail = None
		self.buf_tail_abs = None
		self.buf_head_abs = None
		self.buf_abs_fast = np.zeros((2, SAMP_PER_BLOCK), 'float32')
		self.buf_abs_slow = np.zeros((2, SAMP_PER_BLOCK), 'float32')
		self.buf_diff = np.zeros((SAMP_PER_BLOCK), 'float32')

		# best high point
		self.hn = 0

		# queue of not-yet-processed high points
		self.queue = []

		# output
		self.azim = 0.0
		self.elev = 0.0
		self.level = 0.0
		self.ang = 0.0

		#roll buffer
		self.thresh = 0.00
	

	# dynamic threshold : threshold increase tin noisy environment
	def non_silence_thresh(self,x):
		# data range:samples for 0.5s around high point
		if self.hn is not None:
			noise = np.abs(x[-10000:])
			if x is not None:
				self.thresh = np.mean(noise) # apply a new threshold for non-silence state

		else:
			self.thresh = RAW_MAGNITUDE_THRESH
		#print(self.thresh)
		return self.thresh
        
	

	def filter(self, x, n):

		# create filter
		H = np.ones(n) / n

		# determine lead-in start point
		s = SAMP_PER_BLOCK - n

		# do filter
		y = np.vstack((
			np.convolve(H, x[0][s:], mode='full'),
			np.convolve(H, x[1][s:], mode='full')
		))

		# remove lead-in samples which are bunkum because
		# the filter was starting up, and lead-out samples
		# which we do not need
		y = y[:, n:-n+1]

		# ok
		return y

	# generalized cross correlation
	def gcc(self, d0, d1):
        # substract the means
        # (in order to get a normalized cross-correlation at the end)
		d0 -= d0.mean()
		d1 -= d1.mean()

		# Hann window to mitigate non-periodicity effects
		window = np.hanning(len(d0))

        # compute the cross-correlation
		D0 = rfft(d0 * window)
		D1 = rfft(d1 * window)
		D0r = D0.conjugate()
		G = D0r * D1 # frequency domain based cross correlation
		# W0 = 1. # frequency unweighted
		W1 = 1./np.abs(G) 
		Xgcorr = irfft(W1 * G) # generalized frequency domain based cross correlation with "PHAT"
		
		#absG = np.abs(G)
		#m = max(absG)
		#W = 1. / (1e-10 * m + absG)
		#Xcorr = irfft(W * G)

		#Xcorr = irfft(W * G)

		return Xgcorr
		#return Xcorr


	def high_point(self, hn):

		# FOR INFO, see dev/jeffress

		# measurement range must be large enough to spot correlations at
		# time lags of interest. the ears are separated by about 6 samples
		# at 20kHz (104mm / 343m * 20k) but longer samples are needed to
		# do a good job at spotting inter-ear correlations.
		
		L_max = 7 # 2 more sample for acceptable error in the distance of mics
		L = L_max * 8
		c = L # centre of xcorr equal to L in python 0-indexing (L+1 in matlab)

		# make sure the index is valid 
		hn = round(hn)

		# if out of range on left, discard, because this should only ever
		# happen during start up
		if (hn - L) < 0:
			return

		# if out of range on right, queue for next time
		if (hn + L) >= SAMP_BUFFERED:
			self.queue.append(hn)
			return

		# get height
		#print("best high point {}".format(hn))
		h = self.buf_diff[hn]

		# report
		#print hn-L, hn+L, h

		# extract section of original signal
		wav = self.buf[:, hn-L:hn+L+1]
		wav_tail = self.buf_tail[hn-L:hn+L+1]

		# xcorr
		xco = np.correlate(wav[0, :], wav[1, :], mode='same')
		#xco = self.gcc(wav[0, :], wav[1, :])

		# find best peak in xco (only in plausible range)
		i_peak = np.argmax(xco[c-L_max:c+L_max+1])
		i_peak += L - L_max


		# discard if not actually a peak
		if xco[i_peak-1] > xco[i_peak] or xco[i_peak+1] > xco[i_peak]:
			#print "discard (not a peak)", xco[i_peak-2:i_peak+3]
			return

		# determine lag and level
		lag = float(i_peak - c)
		level = xco[i_peak]

		# discard if level is too low
		if level <= 0:
			print("discard (level too low)", level)
			return

		# report
		#print "best peak", lag, level

		# adjust to sub-sample peak by fitting a parabola to three points
		y1 = xco[i_peak-1]
		y2 = xco[i_peak]
		y3 = xco[i_peak+1]
		dy1 = y2 - y1
		dy3 = y2 - y3
		den = dy1 + dy3
		#print y1, y2, y3, dy1, dy3

		# can only adjust if points are not nearly co-linear (if they are,
		# there would be no benefit anyway)
		if den > 1e-3:
			adj = 0.5 * (dy1 - dy3) / den
			lag = lag + adj

		# report
		#print "adjusted lag", lag

		# normalize and constrain
		lag *= (1.0 / INTER_EAR_LAG)
		lag = np.clip(lag, -1.0, 1.0)
		#print(lag)

		# report
		#print "normalized lag", lag

		# compute azimuth and RMS level
		azim = -np.arcsin(lag)
		level = np.sqrt(level / (2 * L + 1))

        # # the time lag of 45 degree is -61, 90 degree is -57, 135 degree is -54
		# if (azim>0):
		# 	xco_ear_tail = np.correlate(wav[0, :], wav_tail[:], mode='same')
		# 	#t_peak = xco_ear_tail[i_peak]
		# 	t_peak = np.argmax(xco_ear_tail[c-L_max:c+L_max+1])
		# 	t_lag = float(t_peak - c)
		# 	t_lag *= (1.0 / EAR_TAIL_LAG)
		# 	t_lag = np.clip(lag, -1.0, 1.0)
		# 	print(t_lag)
		# 	if(t_lag>=(-0.95)):
		# 		azim = np.pi-azim
			
		# else:
		# 	xco_ear_tail = np.correlate(wav[1, :], wav_tail[:], mode='same')
		# 	t_peak = np.argmax(xco_ear_tail[c-L_max:c+L_max+1])
		# 	t_lag = float(t_peak - c)
		# 	t_lag *= (1.0 / EAR_TAIL_LAG)
		# 	t_lag = np.clip(lag, -1.0, 1.0)
		# 	#print(t_lag)
		# 	if(t_lag>=(-54)):
		# 		azim = -np.pi-azim



		# report
		#print "azimuth and level", azim, level

		# store
		if level > self.level:
			self.azim = azim
			self.level = level

	def process_configuration(self):

		# if running under perftest, this won't be available
		if KC is None:
			print("Here here")
			return

		# get locations of ears in HEAD
		loc_ear_l_HEAD = miro.lib.get("LOC_EAR_L_HEAD")
		loc_ear_r_HEAD = miro.lib.get("LOC_EAR_R_HEAD")

		# transform into FOOT
		loc_ear_l_FOOT = KC.changeFrameAbs(miro.constants.LINK_HEAD, miro.constants.LINK_FOOT, loc_ear_l_HEAD)
		loc_ear_r_FOOT = KC.changeFrameAbs(miro.constants.LINK_HEAD, miro.constants.LINK_FOOT, loc_ear_r_HEAD)

		# get point between ears at assumed height of noise sources
		x = 0.5 * (loc_ear_l_FOOT[0] + loc_ear_r_FOOT[0])
		y = 0.5 * (loc_ear_l_FOOT[1] + loc_ear_r_FOOT[1])

		# get azimuth of "dead-ahead" from ear locations
		dx = loc_ear_r_FOOT[0] - loc_ear_l_FOOT[0]
		dy = loc_ear_r_FOOT[1] - loc_ear_l_FOOT[1]
		azim = np.arctan2(dy, dx) # azimuth from ear_l to ear_r
		azim += np.pi * 0.5 # azimuth of dead-ahead

		# add azimuth of sound source from Jeffress computation
		azim += self.azim

		# estimate sound source location
		dx = np.cos(azim) * ASSUMED_SOUND_SOURCE_RANGE
		dy = np.sin(azim) * ASSUMED_SOUND_SOURCE_RANGE
		x_src = x + dx
		y_src = y + dy
		z_src = ASSUMED_SOUND_SOURCE_HEIGHT

		# map that back into HEAD
		loc_src_FOOT = np.array([x_src, y_src, z_src])
		loc_src_HEAD = KC.changeFrameAbs(miro.constants.LINK_FOOT, miro.constants.LINK_HEAD, loc_src_FOOT)

		# recover from that the view line that we send as output
		#
		# (NB: this discards range and height information, which is
		# the right thing to do - they were useful only in estimation
		# of the view line, they are not useful to carry forward)
		x = loc_src_HEAD[0]
		y = loc_src_HEAD[1]
		z = loc_src_HEAD[2]
		r = np.sqrt(x*x + y*y)
		self.azim = np.arctan2(y, x)
		self.elev = np.arctan2(z, r)
		# azim in degree 
		self.ang = self.azim * 180/np.pi


		# NB2: Actually, let's not discard it just yet.
		self.loc_src_HEAD = loc_src_HEAD


	def process_data(self, data):

		# default
		# event = None
		# sound_level = []

		# clear any pending event (so we can send only one per block)
		self.level = 0.0

		# reshape
		data = np.asarray(data, 'float32') * (1.0 / 32768.0)
		#head_data = data[2:3][:]
		#data_all =  data.reshape((4, 20000))
		#print(data.shape)
		data = data.reshape((4, SAMP_PER_BLOCK))


		# compute level
		sound_level = []
		for i in range(4):
			x = np.mean(np.abs(data[i]))
			sound_level.append(x)
		#print(sound_level)

		# beyond sound level, only interested in left & right
		ear_data = data[0:2][:]
		head_data = data[2][:]
		tail_data = data[3][:]
	

		# fill buffer 0,1
		if self.buf is None:
			self.buf = ear_data
			self.buf_abs = np.abs(ear_data)
			# return (event, sound_level)
		if self.buf_head is None:
			self.buf_head = head_data
			self.buf_head_abs = np.abs(head_data)
		if self.buf_tail is None:
			self.buf_tail = tail_data
			self.buf_tail_abs = np.abs(tail_data)


		# roll buffers, same data is added
		self.buf = np.hstack((self.buf[:, -SAMP_PER_BLOCK:], ear_data))
		self.buf_abs = np.hstack((self.buf_abs[:, -SAMP_PER_BLOCK:], np.abs(ear_data)))
		# print(self.buf.shape) # output is (2,1000)
		self.buf_head = np.hstack((self.buf_head[-SAMP_PER_BLOCK:], head_data))
		self.buf_head_abs = np.hstack((self.buf_head_abs[-SAMP_PER_BLOCK:], np.abs(head_data)))
		self.buf_tail = np.hstack((self.buf_tail[-SAMP_PER_BLOCK:], tail_data))
		self.buf_tail_abs = np.hstack((self.buf_tail_abs[-SAMP_PER_BLOCK:], np.abs(tail_data)))

		# since it is a rolling buffer, we can filter it in a rolling
		# manner. however, I don't know if the convolve() function
		# supports storing filter state. since we will use FIR filters,
		# we can get around this by filtering a little way back from
		# the roll point and using only the later part of the result.

		# filter slow for background level
		b = self.filter(self.buf_abs, 500)
		self.buf_abs_slow = np.hstack((self.buf_abs_slow[:, -SAMP_PER_BLOCK:], b))

		# filter fast for immediate level
		nf = 50
		i = self.filter(self.buf_abs, nf)
		self.buf_abs_fast = np.hstack((self.buf_abs_fast[:, -SAMP_PER_BLOCK:], i))

		# diff those two to see events
		d = np.mean(i - b, axis=0)
		self.buf_diff = np.hstack((self.buf_diff[-SAMP_PER_BLOCK:], d))

		# process any queued high points
		for hn in self.queue:
			self.high_point(hn - SAMP_PER_BLOCK)
		self.queue = []

		# continue reading through looking for events
		N = SAMP_PER_BLOCK * 2
		d = self.buf_diff
		d_abs = np.where(d>0,d,0)
		d_mean = np.mean(d_abs)
		n = self.n - SAMP_PER_BLOCK
		hn = self.hn
		if hn > -1:
			if hn >= SAMP_PER_BLOCK:
				hn -= SAMP_PER_BLOCK
			else:
				# high point now forgotten
				hn = 0
		thresh = RAW_MAGNITUDE_THRESH
		#print(hn)
		# if hn!=0:
		# 	thresh = self.thresh
		# 	#thresh = RAW_MAGNITUDE_THRESH
		# else:
		# 	thresh = RAW_MAGNITUDE_THRESH
		# #print(thresh)
		
		#thresh = self.thresh
		#print(thresh)

		if hn >= 0:
			h = d[hn]
		else:
			h = thresh

		# loop through samples
		while n < N:

			# if waiting for reset
			if hn == -1:

				# if well below threshold
				if d[n] < (0.5 * RAW_MAGNITUDE_THRESH):

					# store new high point (below threshold)
					h = d[n]
					hn = n
					self.hn = hn

			# if not waiting for reset
			else:

				# look for high point
				if d[n] > h:

					# update stored high point
					h = d[n]
					hn = n
					self.hn = hn

				# look for end of high point
				if h > thresh and d[n] < (0.5 * h):

					# process high point
					self.high_point(self.hn - nf / 2)

					# clear
					h = thresh
					hn = -1

			# advance
			n += 1

		# restore
		self.n = n
		self.hn = hn

		# default
		event = None
		event_head = None

		# process any pending event
		if self.level:

			# adjust for configuration of robot right now
			self.process_configuration()

			# publish
			event = DetectAudioEvent([self.azim, self.elev, self.level,self.ang])
			event_head = DetectAudioEventHead(self.loc_src_HEAD)

		# return
		return (event, event_head, sound_level)



