import sys
import time


#Output example: [===  ] 75%
#width define bar width
#percent define current percentage

def progress(width,percent,t=1):
	if t==1:
		print "\r%s %d%%" %(('%%-%ds' % width ) %(width*percent/100*'='),percent),
	else:
		print "\r%-2s%%" %percent,
	sys.stdout.flush()

	if percent >= 100:
		print "\r",
		sys.stdout.flush()


if __name__ == "__main__":
	# this is for test!
	for i in xrange(100):
		progress(50,i+1,1)
		time.sleep(0.1)

