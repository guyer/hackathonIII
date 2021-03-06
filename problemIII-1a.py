import os
import sys
import argparse
import time
import uuid 

from scipy.optimize import fsolve

import datreant.core as dtr

import fipy as fp
from fipy.tools.numerix import cos, sin
from fipy.tools import parallelComm

parser = argparse.ArgumentParser()
parser.add_argument("--output", help="directory to store results in",
                    default=os.path.join("Data", str(uuid.uuid4())))
parser.add_argument("--sweeps", help="number of nonlinear sweeps to take",
                    type=int, default=10)
parser.add_argument("--check", help="period of sweeps to checkpoint data",
                    type=int, default=1)
parser.add_argument("--dx", help="cell spacing in x direction",
                    type=float, default=0.2)
parser.add_argument("--dy", help="cell spacing in y direction",
                    type=float, default=0.2)
parser.add_argument("--compression", help="number of compressed cells within outlet 'cell'",
                    type=int, default=1)
args, unknowns = parser.parse_known_args()
                    
if parallelComm.procID == 0:
    print "storing results in {0}".format(args.output)
    data = dtr.Treant(args.output)
else:
    class dummyTreant(object):
        categories = dict()
        
    data = dummyTreant()
    
data.categories['problem'] = "III-1a"
data.categories['args'] = " ".join(sys.argv)
data.categories['sweeps'] = args.sweeps
data.categories['dx'] = args.dx
data.categories['dy'] = args.dy
data.categories['compression'] = args.compression
data.categories['commit'] = os.popen('git log --pretty="%H" -1').read().strip()
data.categories['diff'] = os.popen('git diff').read()
    
viscosity = 1
density = 100.
gravity = [0., -0.001]
pressureRelaxation = 0.8
velocityRelaxation = 0.5

Lx = 30.
Ly = 6.
dx = args.dx
dy = args.dy

def fn(f, N):
    '''Root solving kernel for compression factor
    
    Determine f(N), such that $\Delta x \sum_{i=0}^N f^i = 2 \Delta x$
    '''
    return (1 - f**N) / (1 - f) - 2.
    
N = 1 + args.compression
compression = fsolve(fn, x0=[.5], args=(N))[0]

Nx = int(Lx / dx)
Ny = int(Ly / dy)
dx_variable = [dx] * (Nx - 2) + [dx * compression**i for i in range(N+1)]

dy_variable = [dy] * Ny

mesh = fp.Grid2D(dx=dx_variable, dy=dy_variable)
volumes = fp.CellVariable(mesh=mesh, value=mesh.cellVolumes)

pressure = fp.CellVariable(mesh=mesh, name="$p$")
pressureCorrection = fp.CellVariable(mesh=mesh, name="$p'$")
xVelocity = fp.CellVariable(mesh=mesh, name="$u_x$")
yVelocity = fp.CellVariable(mesh=mesh, name="$u_y$")

velocity = fp.FaceVariable(mesh=mesh, name=r"$\vec{u}$", rank=1)

xVelocityEq = fp.DiffusionTerm(coeff=viscosity) - pressure.grad.dot([1.,0.]) + density * gravity[0]
yVelocityEq = fp.DiffusionTerm(coeff=viscosity) - pressure.grad.dot([0.,1.]) + density * gravity[1]

ap = fp.CellVariable(mesh=mesh, value=1.)
coeff = 1./ ap.arithmeticFaceValue*mesh._faceAreas * mesh._cellDistances
pressureCorrectionEq = fp.DiffusionTerm(coeff=coeff) - velocity.divergence

contrvolume = volumes.arithmeticFaceValue

x, y = mesh.cellCenters
X, Y = mesh.faceCenters

def inlet(yy):
    return -0.001 * (yy - 3)**2 + 0.009
    
xVelocity.constrain(inlet(Y), mesh.facesLeft)
xVelocity.constrain(0., mesh.facesTop | mesh.facesBottom)

yVelocity.constrain(0., mesh.facesTop | mesh.facesBottom | mesh.facesLeft)

pressureCorrection.constrain(0., mesh.facesRight & (Y > Ly - dy))
# pressureCorrection.constrain(0., mesh.facesRight)

with open(data['residuals.npy'].make().abspath, 'a') as f:
    f.write("{}\t{}\t{}\t{}\t{}\n".format("sweep", "x_residual", "y_residual", "p_residual", "continuity"))

fp.tools.dump.write((xVelocity, yVelocity, velocity, pressure), 
                    filename=data["sweep={}.tar.gz".format(0)].make().abspath)

start = time.clock()

for sweep in range(1, args.sweeps+1):
    ## solve the Stokes equations to get starred values
    xVelocityEq.cacheMatrix()
    xres = xVelocityEq.sweep(var=xVelocity,
                             underRelaxation=velocityRelaxation)
    xmat = xVelocityEq.matrix

    yres = yVelocityEq.sweep(var=yVelocity,
                             underRelaxation=velocityRelaxation)

    ## update the ap coefficient from the matrix diagonal
    ap[:] = -xmat.takeDiagonal()

    ## update the face velocities based on starred values with the
    ## Rhie-Chow correction.
    ## cell pressure gradient
    presgrad = pressure.grad
    ## face pressure gradient
    facepresgrad = pressure.faceGrad

    velocity[0] = xVelocity.arithmeticFaceValue \
         + contrvolume / ap.arithmeticFaceValue * \
           (presgrad[0].arithmeticFaceValue-facepresgrad[0])
    velocity[1] = yVelocity.arithmeticFaceValue \
         + contrvolume / ap.arithmeticFaceValue * \
           (presgrad[1].arithmeticFaceValue-facepresgrad[1])
    velocity[..., mesh.exteriorFaces.value] = 0.
    velocity[0, mesh.facesLeft.value] = inlet(Y)[mesh.facesLeft.value]
    velocity[0, mesh.facesRight.value] = xVelocity.faceValue[mesh.facesRight.value]
    velocity[1, mesh.facesRight.value] = yVelocity.faceValue[mesh.facesRight.value]

    ## solve the pressure correction equation
    pressureCorrectionEq.cacheRHSvector()
    ## right top point must remain at pressure 0, so no correction
    pres = pressureCorrectionEq.sweep(var=pressureCorrection)
    rhs = pressureCorrectionEq.RHSvector

    ## update the pressure using the corrected value
    pressure.setValue(pressure + pressureRelaxation * pressureCorrection)
    ## update the velocity using the corrected pressure
    xVelocity.setValue(xVelocity - pressureCorrection.grad[0] / \
                                               ap * mesh.cellVolumes)
    yVelocity.setValue(yVelocity - pressureCorrection.grad[1] / \
                                               ap * mesh.cellVolumes)

    if sweep % args.check == 0:
        fp.tools.dump.write((xVelocity, yVelocity, velocity, pressure), 
                            filename=data["sweep={}.tar.gz".format(sweep)].make().abspath)
                   
    with open(data['residuals.txt'].make().abspath, 'a') as f:
        f.write("{}\t{}\t{}\t{}\t{}\n".format(sweep, xres, yres, pres, max(abs(rhs))))
                            
data.categories['elapsed'] = time.clock() - start
