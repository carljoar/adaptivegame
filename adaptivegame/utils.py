import numpy as np

def MCA(f):
	"""
	Compute the mean competitive ability of a strategy.
	
	Parameters
	----------
	f : list
		A list of competitive abilities (floating point numbers).
	
	Returns
	-------
	float
	"""
	x = np.linspace(0,1,len(f))
	return np.sum(f*x) / np.sum(f)

def grad(x,f):
	"""
	Compute the unconstrained selection gradient at x. 
	This is equivalent to taking the x-th element of L*f, where L is defined in Lmatrix below.
	
	Parameters
	----------
	f : list
		A list of competitive abilities (floating point numbers).
	x : int
		Index of the evaluation point.
	
	Returns
	-------
	float
	"""
	# Compute the selection gradient.
	# Notice that slicing indexing ends at last element.
	# Second term ranges from x+1 to end.
	N = len(f)-1
	return (1./N) * ( np.sum(f[0:x])-np.sum(f[x+1:]) ) 

def E(f,g):
	"""
	Compute the expectation of strategy f in competition with strategy g.
	
	Parameters
	----------
	f : list
		A list of competitive abilities (floating point numbers).
	g : list
		A list of competitive abilities (floating point numbers).
	
	Returns
	-------
	float
	"""
	numpoints = len(f)
	if numpoints != len(g):
		return float("nan")
	g_integrand = np.zeros(numpoints)
	for i in range(numpoints):
		g_integrand[i] = grad(i,g)
	return (1./(numpoints-1)) * np.sum( f*g_integrand )

def ProjectionMatrix(n,C):
	"""
	Compute the projection onto the tangent space of MCA(f)=C.
	
	Parameters
	----------
	n : int
		The size (n x n) of the matrix. 
	C : float
		The value in MCA(f)=C, which defines the level set {f: MCA(f)=C}.
	
	Returns
	-------
	ndarray, shape=(n,n)
	"""
	nm1=n-1
	xvec = np.linspace(0,1,n)
	c_norm2 = np.sum( (xvec-C)*(xvec-C) )
	if nm1<=0:
		print("Number of points needs to be at least 2")
		return float("nan")
	# Projection matrix
	P = np.ones((n,n))
	for i in range(n):
		for j in range(n):
			P[i,j] = (1.0*i/nm1-C)*(1.0*j/nm1-C) /c_norm2 
	return P

def PositiveInd(L,y):
	"""
	Compute which indices that are in the risk of having negative competitive ability.
	
	Parameters
	----------
	L : ndarray, shape=(n,n)
		 The matrix that maps a strategy to the selection gradient.
	
	y : list
		A list of competitive abilities (floating point numbers).
	
	Returns
	-------
	list
	"""
	gfy = np.matmul(L,y)
	out = np.zeros(numpoints)
	#tmp = np.zeros(numpoints)
	#np.isposinf(gfy,tmp) # Store location of positive gradient in tmp
	for i in range(numpoints):
		if gfy[i]<0 and y[i]<=0:
			out[i] = 0.0
		else:
			out[i] = 1.0
	return out

def Lmatrix(n):
	"""
	Construct the matrix that maps a strategy to the selection gradient.
	
	Parameters
	----------
	n : int
		The size (n x n) of the matrix.
	
	Returns
	-------
	ndarray, shape=(n,n)
	"""
	if n<=1:
		print("Number of points needs to be at least 2")
		return float("nan")
	# Create unconstrained competition matrix
	L = np.ones((n,n))
	for i in range(n):
		L[i,i] = 0.0
		L[i,i+1:] = -1.0
	return L

def Kmatrix(n,C):
	"""
	Construct the matrix that maps a strategy to the selection gradient with projection onto MCA(f)=C.
	
	Parameters
	----------
	n : int
		The size (n x n) of the matrix.
	C : float
		The value in MCA(f)=C, which defines the level set {f: MCA(f)=C}.
	
	Returns
	-------
	ndarray, shape=(n,n)
	"""
	if n<=1:
		print("Number of points needs to be at least 2")
		return float("nan")
	L = Lmatrix(n)
	P = ProjectionMatrix(n,C) 
	# Return competition matrix with MCA(y)=C
	return L - np.matmul(P,L)