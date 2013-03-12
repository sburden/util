
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

def nanmean(a, axis=None):
  """
  Mean of array elements, ignoring NaNs

  Inputs
    a - np.array
    (optional)
    axis - int - axis to flatten; defaults to nanmean of flattened array

  Outputs
    b - nanmean along specified axis (scalar if axis is None)
  """
  n = np.sum( np.logical_not( np.isnan(a) ), axis=axis )
  z = (n == 0)
  b = np.nansum( a, axis=axis )
  if np.asarray(n).shape == ():
    if z:
      return np.nan
    else:
      return b / n
  else:
    b[z.nonzero()] = np.nan
    j = (1-z).nonzero()
    b[j] = b[j] / n[j]
    return b

def nanstd(a, axis=None, nm=0.):
  """
  Standard deviation of array elements, ignoring NaNs

  Inputs
    a - np.array
    (optional)
    axis - int - axis to flatten; defaults to nanstd of flattened array
    nm - correction for bias in STD of sample; subtracted from axis size

  Outputs
    b - nanstd along specified axis (scalar if axis is None)
  """
  n = np.sum( np.logical_not( np.isnan(a) ), axis=axis )
  z = (n == 0)
  m = nanmean( a, axis=axis )
  b = np.nansum( (a - m)**2, axis=axis )
  if np.asarray(n).shape == ():
    if z:
      return np.nan
    else:
      return np.sqrt( b / (n - nm) )
  else:
    b[z.nonzero()] = np.nan
    j = (1-z).nonzero()
    b[j] = np.sqrt( b[j] / (n[j] - nm) )
    return b

def nanstudentize(a, axis=None, nm=0.):
  """
  Studentize array, ignoring NaNs

  Inputs
    a - np.array
    (optional)
    axis - int - axis to flatten; defaults to nanstd of flattened array
    nm - correction for bias in STD of sample; subtracted from axis size

  Outputs
    b - nanstd along specified axis (scalar if axis is None)
  """
  return (a - nanmean(a,axis=axis)) / nanstd(a,axis=axis,nm=nm)

def interp0(s,t,xt,bd=None):
  """
  xs = interp0(s,t,xt)  zero-order hold

  Inputs:
    s - ms - output samples
    t - mt - input samples
    xt - mt x n - input values
    (optional)
    bd - scalar or [l,u] - boundary values; defaults to [xt[0],xt[-1]]

  Outputs:
    xs - ms x n - output values
  """
  from collections import Iterable
  assert len(s.shape) == 1
  assert len(t.shape) == 1
  assert np.all(np.diff(t) >= 0)
  assert xt.shape[0] == t.size
  m = s.size; mt = t.size
  if len(xt.shape) == 1:
    n = 1
    xs = np.nan*np.zeros(m)
  else:
    n = xt.shape[1]
    xs = np.nan*np.zeros((m,n))
  if bd is None:
    bd = [xt[0],xt[-1]]
  elif not isinstance(bd,Iterable):
    bd = [bd,bd]
  else:
    assert len(bd) == 2
  xs[s < t[0]]  = bd[0]
  xs[s >= t[-1]] = bd[1]
  js = ( (s >= t[0])*(s < t[-1]) ).nonzero()
  jt = (((s[js][:,np.newaxis] - t) >= 0) * range(t.size)).max(axis=1)
  xs[js] = xt[jt]

  return xs

def interp1(s,t,xt,bd=None):
  """
  xs = interp1(s,t,xt)  piecewise-linear

  Inputs:
    s - ms - output samples
    t - mt - input samples
    xt - mt x n - input values
    (optional)
    bd - scalar or [l,u] - boundary values; defaults to [xt[0],xt[-1]]

  Outputs:
    xs - ms x n - output values
  """
  from collections import Iterable
  assert len(s.shape) == 1
  assert len(t.shape) == 1
  assert np.all(np.diff(t) >= 0)
  assert xt.shape[0] == t.size
  m = s.size; mt = t.size
  if len(xt.shape) == 1:
    n = 1
    xs = np.nan*np.zeros(m)
  else:
    n = xt.shape[1]
    xs = np.nan*np.zeros((m,n))
  if bd is None:
    bd = [xt[0],xt[-1]]
  elif not isinstance(bd,Iterable):
    bd = [bd,bd]
  else:
    assert len(bd) == 2
  xs[s < t[0]]  = bd[0]
  xs[s >= t[-1]] = bd[1]
  js = ( (s >= t[0])*(s < t[-1]) ).nonzero()
  jt = (((s[js][:,np.newaxis] - t) >= 0) * range(t.size)).max(axis=1)
  sj = s[js]; tj = t[jt]; tjj = t[jt+1]; dt = tjj - tj
  s0 = ((sj - tj)/dt)[:,np.newaxis]
  s1 = ((tjj - sj)/dt)[:,np.newaxis]
  if len(xt.shape) == 1:
    s0 = s0.flatten()
    s1 = s1.flatten()
  xs[js] = xt[jt]*s0 + xt[jt+1]*s1
  xs[js] = xt[jt]

  return xs

def localmin( N, dat ):
  """
  mrk = localmin( N, dat )  find local minima within neighborhoods 

  INPUT:
    N - int - number of samples before and after putative minimum
    dat - D-dim array - data comparison along last axis

  OUTPUT:
    mrk - D-dim bool - true for samples that are local minima, 
      i.e. mrk[...,i] is true if dat[...,i] is the minimum of dat[...,i-N:i+N]

  By Shai Revzen, Berkeley 2006,2007
  Pythonified by Sam Burden, Berkeley 2012
  """
  #mrk = true(size(dat));
  mrk = np.ones_like(dat)
  #for k=1:N
  for k in range(N):
    #mrk( (k+1):end, : ) = mrk( (k+1):end, : ) & (dat((k+1):end,:)<=dat(1:(end-k),:));
    #mrk[k+1:,:] = mrk[k+1:,:] * ( dat[k+1:,:] <= dat[:-k,:] )
    mrk[...,k+1:] = mrk[...,k+1:] * ( dat[...,k+1:] <= dat[...,:-k-1] )
    #mrk( 1:(end-k), : ) = mrk( 1:(end-k), : ) & (dat(1:(end-k),:)<=dat((k+1):end,:));
    #mrk[:-k,:] = mrk[:-k,:] * ( dat[:-k,:] <= dat[k+1:,:] )
    mrk[...,:-k-1] = mrk[...,:-k-1] * ( dat[...,:-k-1] <= dat[...,k+1:] )
  #end
  #mrk(1:N,:)=false;
  #mrk[:N,:] = 0.;
  mrk[...,:N] = 0.;
  #mrk(end-N+1:end,:)=false;
  #mrk[-N+1:,:] = 0.;
  mrk[...,-N:] = 0.;
  #return
  return mrk

