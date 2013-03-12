
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public
#  License as published by the Free Software Foundation; either
#  version 3.0 of the License, or (at your option) any later version.
#
#  The library is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#  General Public License for more details.
#
# (c) Sam Burden, UC Berkeley, 2013 

import numpy as np
np.set_printoptions(precision=4)

def monomials(M,N,n=0):
  """
  generate list of monomial powers in M variables with degree N
  """

  def _yieldParts(num,lt): 
      if not num: 
          yield [] 
      for i in range(min(num,lt),0,-1): 
          for parts in _yieldParts(num-i,i): 
              yield [i]+parts 

  def yieldParts(num): 
      for part in _yieldParts(num,num): 
          yield part 

  def permutations (orig_list):
      if not isinstance(orig_list, list):
          orig_list = list(orig_list)

      yield orig_list

      if len(orig_list) == 1:
          return

      for n in sorted(orig_list):
          new_list = orig_list[:]
          pos = new_list.index(n)
          del(new_list[pos])
          new_list.insert(0, n)
          for resto in permutations(new_list[1:]):
              if new_list[:1] + resto <> orig_list:
                  yield new_list[:1] + resto

  def pad(li,l,z=0):
    return [ll + [z for k in range(l - len(ll))] for ll in li if len(ll) <= l]

  p0 = [pad(list(yieldParts(nn)),M,0) for nn in range(n,N+1)]
  p1 = [list(permutations(ppp)) for pp in p0 for ppp in pp]
  p2 = [ppp for pp in p1 for ppp in pp]
  p3 = list(set([tuple(pp) for pp in p2]))

  return sorted(p3,key=lambda p : [sum(p)]+list(p))

def diff(p,a,k):
  """
  p,a = poly.diff  differentiate polynomial

  Inputs:
    p - Nm x Nx - exponents for Nm monomials in Nx variables
    a - Ny x Nm - Ny collections of Na monomial coefficients
    k - int - index to differentiate

  Outputs:
    dp - Nm x Nx - exponents for Nm monomials in Nx variables
    da - Ny x Nm - Ny collections of Na monomial coefficients
  """
  p = np.array(p); a = np.array(a)
  dp = monomials(p.shape[1],p.sum(axis=1).max()-1)
  da = np.zeros((a.shape[0],len(dp)))

  for aa,daa in zip(a,da):
    for pp,aaa in zip(p,aa):
      if pp[k] == 0:
        continue
      ppp = list(pp); ppp[k] -= 1; ppp = tuple(ppp)
      if ppp not in dp:
        raise LookupError, "FIXME:  monomial not found"
      daa[dp.index(ppp)] += (ppp[k]+1)*aaa

  return dp,da

def jac(p,a,k):
  """
  p,a = poly.jac  differentiate polynomial

  Inputs:
    p - Nm x Nx - exponents for Nm monomials in Nx variables
    a - Ny x Nm - Ny collections of Na monomial coefficients
    k - int - index to differentiate

  Outputs:
    dp - Nm x Nx - exponents for Nm monomials in Nx variables
    da - Ny x Nm - Ny collections of Na monomial coefficients
  """
  p = np.array(p); a = np.array(a)
  dp = monomials(p.shape[1],p.sum(axis=1).max()-1)
  da = np.zeros((a.shape[0],len(dp)))

  for aa,daa in zip(a,da):
    for pp,aaa in zip(p,aa):
      if pp[k] == 0:
        continue
      ppp = list(pp); ppp[k] -= 1; ppp = tuple(ppp)
      if ppp not in dp:
        raise LookupError, "FIXME:  monomial not found"
      daa[dp.index(ppp)] += (ppp[k]+1)*aaa

  return dp,da

def val(p,a,x):
  """
  y = poly.val  evaluate polynomial

  Inputs:
    p - Nm x Nx - exponents for Nm monomials in Nx variables
    a - Ny x Nm - Ny collections of Na monomial coefficients
    x - Nx x N - N samples of Nx dimensional input

  Outputs:
    y - Ny x N - N samples of Ny dimensional output
  """
  x = np.array(x)

  Nx,N = x.shape

  X = np.array([np.prod([np.power(xx,ppp) 
                         for xx,ppp in zip(x,pp)],axis=0) 
                         for pp in p])

  return np.dot(a,X)

def fit(x,y,deg=1,p=None):
  """
  p,a = poly.fit  fit polynomial to data using least-squares

  i.e. if y = a X, then a = y^T X (X^T X)^{-1}
       where cols of X contain monomials up to degree deg
       evaluated at values in x

  Inputs:
    x - Nx x N - N samples of Nx dimensional input
    y - Ny x N - N samples of Ny dimensional output
  (optional)
    deg - int - maximum degree of monomials (generates p)
    p - Nm x Nx - exponents for Nm monomials in Nx variables

  Outputs:
    p - Nm x Nx - exponents for Nm monomials in Nx variables
    a - Ny x Nm - Ny collections of Na monomial coefficients
  """
  x = np.array(x); y = np.array(y)

  Nx,N = x.shape

  if p is None:
    p = monomials(Nx,deg)

  a = []
  for yy in y:
    Y = yy.reshape((N,1))
    X = np.array([np.prod([np.power(xx,ppp) 
                           for xx,ppp in zip(x,pp)],axis=0) 
                           for pp in p]).T

    a += [np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,Y)).flatten()]

  return p,np.array(a)

if __name__ == "__main__":

  deg = 5; Nx = 3; Ny = 1; N = 100; s = 1e-2
  
  p = monomials(Nx,deg)
  a = np.random.randn(Ny,len(p))*np.round(np.random.rand(Ny,len(p)))
  x = np.random.randn(Nx,N)

  y = val(p,a,x) + s*np.random.randn(Ny,N)

  pp,aa = fit(x,y,deg)

  print ('%d order poly, %d vars, %d outputs, %d training samples' 
          % (deg,Nx,Ny,N))
  print (aa-a) 


  deg0 = 3; deg = 2; Nx = 2; Ny = 2; N = 100; s = 1e-3
  
  p = monomials(Nx,deg0)
  a = np.random.randn(Ny,len(p))#*np.round(np.random.rand(Ny,len(p)))
  x = np.random.randn(Nx,N)*1e-1

  y = val(p,a,x) + s*np.random.randn(Ny,N)

  pp,aa = fit(x,y,deg)

  print ('%d order poly, %d order fit, %d vars, %d outputs, %d training samples' 
          % (deg0,deg,Nx,Ny,N))
  print a[:,:6]
  print aa


  deg = 1; Nx = 2; Ny = 2; N = 100; s = 1e-3
  
  p = monomials(Nx,deg,n=0)
  a = np.random.randn(Ny,len(p))
  x = np.random.randn(Nx,N)

  y = val(p,a,x) + s*np.random.randn(Ny,N)

  pp,aa = fit(x,y,deg)

  print ('%d order poly, %d vars, %d outputs, %d training samples' 
          % (deg,Nx,Ny,N))
  print a
  print aa

