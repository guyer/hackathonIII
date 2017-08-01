import os
import sys
import argparse
import time
import uuid 

import numpy as np

import datreant.core as dtr

import fipy as fp
from fipy.tools.numerix import cos, sin
from fipy.tools import parallelComm

parser = argparse.ArgumentParser()
parser.add_argument("--output", help="directory to store results in",
                    default=str(uuid.uuid4()))
parser.add_argument("--sweeps", help="number of nonlinear sweeps to take",
                    type=int, default=10)
parser.add_argument("--step", help="time step to attempt",
                    type=float, default=10.)
args, unknowns = parser.parse_known_args()
                    
if parallelComm.procID == 0:
    path = os.path.join("Data", args.output)
    print "storing results in {0}".format(path)
    data = dtr.Treant(path)
else:
    class dummyTreant(object):
        categories = dict()
        
    data = dummyTreant()
    
data.categories['args'] = " ".join(sys.argv)
data.categories['step'] = args.step
data.categories['sweeps'] = args.sweeps
data.categories['commit'] = os.popen('git log --pretty="%H" -1').read().strip()
data.categories['diff'] = os.popen('git diff').read()
    
mesh = fp.Grid2D(nx=100, ny=100)
volumes = fp.CellVariable(mesh=mesh, value=mesh.cellVolumes)

c = fp.CellVariable(mesh=mesh, name="$c$", hasOld=True)
psi = fp.CellVariable(mesh=mesh, name=r"$\psi$", hasOld=True)
Phi = fp.CellVariable(mesh=mesh, name=r"$\Phi$", hasOld=True)

calpha = 0.3
cbeta = 0.7
kappa = 2.
rho = 5.
M = 5.
k = 0.09
epsilon = 90.


ceq = fp.TransientTerm(var=c) == fp.DiffusionTerm(coeff=M, var=psi)

fchem = rho * (c - calpha)**2 * (cbeta - c)**2
felec = k * c * Phi / 2.
f = fchem + (kappa/2.) * c.grad.mag**2 + felec
dfchemdc = 2 * rho * (c - calpha) * (cbeta - c) * (calpha + cbeta - 2 * c)
d2fchemd2c = 2 * rho * ((calpha + cbeta - 2 * c)**2 - 2 * (c - calpha) * (cbeta - c))
psieq = (fp.ImplicitSourceTerm(coeff=1., var=psi) 
         == fp.ImplicitSourceTerm(coeff=d2fchemd2c, var=c) - d2fchemd2c * c + dfchemdc
         - fp.DiffusionTerm(coeff=kappa, var=c)
         + fp.ImplicitSourceTerm(coeff=k, var=Phi))
         
stats = []

Phieq = fp.DiffusionTerm(var=Phi) == fp.ImplicitSourceTerm(coeff=-k/epsilon, var=c)

eq = ceq & psieq & Phieq

x, y = mesh.cellCenters
X, Y = mesh.faceCenters

Phi.faceGrad.constrain(0., where=mesh.facesTop | mesh.facesBottom)
Phi.constrain(0., where=mesh.facesLeft)
Phi.constrain(sin(Y/7.), where=mesh.facesRight)

c0 = 0.5
c1 = 0.04
c.setValue(c0 + c1 * (cos(0.2*x) * cos(0.11*y) 
                      + (cos(0.13*x) * cos(0.087*y))**2 
                      + cos(0.025*x - 0.15*y) * (cos(0.07*x - 0.02*y))))
Phi.setValue(0.)

synchTimes = [0, 5, 10, 20, 50, 100, 200, 400, 1000]
synchTimes.reverse()

t = synchTimes.pop()

def saveData(t, dt, dt_synch):
    if dt_synch == dt:
        fp.tools.dump.write((c, Phi), 
                            filename=data["t={}.tar.gz".format(t)].make().abspath)
                            
    stats.append((t, (f.cellVolumeAverage * mesh.numberOfCells).value))
    fp.numerix.save(data['stats.npy'].make().abspath, 
                    fp.numerix.array(stats, 
                                     dtype=[('time', float), ('energy', float)]))

dt = args.step
saveData(t, dt, dt)

start = time.clock()

while True:
    c.updateOld()
    psi.updateOld()
    Phi.updateOld()

    try:
        synchTime = synchTimes.pop()
    except IndexError:
        break
    dt_synch = synchTime - t
    dt_save = dt
    if dt_synch < dt:
        dt = dt_synch
    elif dt_synch > dt:
        synchTimes.append(synchTime)
    t += dt
    
    for sweep in range(args.sweeps):
        res = eq.sweep(dt=dt) #, solver=fp.LinearGMRESSolver(precon=fp.JacobiPreconditioner()))

    saveData(t, dt, dt_synch)
    
    dt = dt_save
    
data.categories['elapsed'] = time.clock() - start
