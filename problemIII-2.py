import argparse
import time
import uuid 

import datreant.core as dtr

import fipy as fp
from fipy.tools.numerix import cos, sin
from fipy.tools import parallelComm

parser = argparse.ArgumentParser()
parser.add_argument("--output", help="directory to store results in",
                    default=str(uuid.uuid4()))
args, unknowns = parser.parse_known_args()
                    
if parallelComm.procID == 0:
    print "storing results in {0}".format(args.output)
    data = dtr.Treant(args.output)
else:
    class dummyTreant(object):
        categories = dict()
        
    data = dummyTreant()
    
mesh = fp.Grid2D(nx=100, ny=100)

c = fp.CellVariable(mesh=mesh, name="$c$", hasOld=True)
psi = fp.CellVariable(mesh=mesh, name=r"$\psi$", hasOld=True)
Phi = fp.CellVariable(mesh=mesh, name=r"$\Phi$", hasOld=True)

calpha = 0.3
cbeta = 0.7
kappa = 2.
rho = 5.
M = 5.
k = 0.09
epsilon = 9.


# Boundary condition is stipulated as $\nabla c \cdot \hat{n} = 0$, which is not no-flux. Constraining c.faceGrad to zero seems to do nothing, so we impose a flux constraint on the remainder of the flux (assuming that $\nabla^3 c \cdot \hat{n} = 0$). Thus,
# \begin{align*}
# \vec{J} &= M\nabla\left(\frac{\partial^2 f_{chem}}{\partial c^2}\nabla c - \kappa \nabla^2 c + k \Phi\right) 
# \\
# &\approx M\nabla\left(k \Phi\right) 
# \end{align*}
# on exterior faces.

c_BC = (M*k*Phi.faceGrad*mesh.exteriorFaces).divergence
ceq = fp.TransientTerm(var=c) == fp.DiffusionTerm(coeff=M, var=psi) + c_BC


dfchemdc = 2 * rho * (c - calpha) * (cbeta - c) * (calpha + cbeta - 2 * c)
d2fchemd2c = 2 * rho * ((calpha + cbeta - 2 * c)**2 - 2 * (c - calpha) * (cbeta - c))
psieq = (fp.ImplicitSourceTerm(coeff=1., var=psi) 
         == fp.ImplicitSourceTerm(coeff=d2fchemd2c, var=c) - d2fchemd2c * c + dfchemdc
         - fp.DiffusionTerm(coeff=kappa, var=c)
         + fp.ImplicitSourceTerm(coeff=k, var=Phi))

Phieq = fp.DiffusionTerm(var=Phi) == 0. # fp.ImplicitSourceTerm(coeff=-k/epsilon, var=c)

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

synchTimes = [5, 10, 20, 40, 100, 200, 400, 1000]
synchTimes.reverse()

t = 0.
dt = 1.

while True:
    c.updateOld()
    psi.updateOld()
    Phi.updateOld()

    synchTime = synchTimes.pop()
    dt_synch = synchTime - t
    dt_save = dt
    if dt_synch < dt:
        dt = dt_synch
    elif dt_synch > dt:
        synchTimes.append(synchTime)
    t += dt
    
    for sweep in range(1):
        res = eq.sweep(dt=dt) #, solver=fp.LinearGMRESSolver(precon=fp.JacobiPreconditioner()))

    if dt_synch == dt:
        fp.tools.dump.write((c, Phi), 
                            filename=data["t={}.tar.gz".format(t)].make().abspath)
                            
    dt = dt_save