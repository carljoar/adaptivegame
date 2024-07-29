import numpy as np
from .utils import *

def run(num_iters,y0,eps,C):
	n = len(y0)
	y = y0[:]
	y_prev = y0[:]
	y_first = y0[:]
	L = Lmatrix(n)
	K = Kmatrix(n,C)
	df = np.matmul(K,y)

	y2 = np.zeros(num_iters)

	convergence = np.zeros(num_iters)
	e_history = np.zeros(num_iters)
	mca_history = np.zeros(num_iters)
	h2_history = np.zeros(num_iters)
	runTrue = True
	k=0
	y_first = y + eps*df

	# The following function defines an eps which takes y to MCA(y)=C.
	def new_eps(L,y):
		Ly = np.matmul(L,y)
		Cx = C-np.linspace(0,1,n)
		numer = np.sum(Cx*y)
		denom = np.sum(-Cx*Ly)
		return numer/denom

	while runTrue and MCA(y)<(C-0.45e-9) and k<num_iters:
		df = np.matmul(L,y)
		y_prev = y
		#df_prev = np.matmul(K,y_prev)
		thiseps = 10
		for i in range(n):
			if df[i] != 0.0:
				thiseps = -y[i]/df[i]
				#print(thiseps)
			if thiseps < eps and thiseps>=0 and y[i]>0.0:
				eps = thiseps*0.99
				print('eps update: ',eps)
			if new_eps(L,y)<eps:
				eps = new_eps(L,y)
				print('eps update: ',eps)
		for i in range(n):
			if df[i]<0.0 and y[i]<=0.0:
				runTrue = False
		# Newton forward integration
		y = y + eps * df
		if E(y,y_prev)<0 or MCA(y)>0.5:
			runTrue = False
		if eps < 1e-10:
			runTrue = False
		e_history[k] = E(y,y_prev)
		mca_history[k] = MCA(y)
		h2_history[k] = np.sum(y*y)
		convergence[k] = np.sum(y)
		y2[k] = y[1]
		k = k+1

	print('Iterations before projection onto MCA=C :',k)
	runTrue = True
	while runTrue and k<num_iters:
		df = np.matmul(K,y)
		y_prev = y
		#df_prev = np.matmul(K,y_prev)
		thiseps = 10
		for i in range(n):
			if df[i] != 0.0:
				thiseps = -y[i]/df[i]
				#print(thiseps)
			if thiseps < eps and thiseps>=0 and y[i]>0.0:
				eps = thiseps*0.99
				print('eps update: ',eps)
		for i in range(n):
			if df[i]<0.0 and y[i]<=0.0:
				runTrue = False
		# Newton forward integration
		y[0] = y[0] * (1 + 1e-5/num_iters)  # Avoid 'leakage' into MCA>0.5
		y = y + eps * df
		if E(y,y_prev)<0 or MCA(y)>0.5:
			runTrue = False
		if eps < 1e-10:
			runTrue = False
		e_history[k] = E(y,y_prev)
		mca_history[k] = MCA(y)
		h2_history[k] = np.sum(y*y)
		convergence[k] = np.sum(y)
		y2[k] = y[1]
		k = k+1


	# Format output
	outp = ' --- Terminate at a losing strategy'
	if E(y,y0)>0:
		outp = ' +++ The end result is a winning strategy'
	if E(y,y0)==0:
		outp = ' xxx It is a draw'

	if np.any(y<0):
		print(' ERROR: The result contains negative values')

	print('Finish at ',k,' iterations\n')
	print(outp)
	print("E[y,y0] =",E(y,y0))
	print("MCA(y) =",MCA(y))
	print("\nVerify that the first iteration provides a winning strategy:")
	print("E[y1,y0] =",E(y_first,y0))
	print("MCA(y1) =",MCA(y_first))

	return y