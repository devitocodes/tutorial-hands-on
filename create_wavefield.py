import numpy as np
from devito import Grid, Function, TimeFunction, Eq, Operator
from sympy import solve
from examples.seismic import RickerSource

# Define the time parameters of our simulation
t0 = 0.0  # Start time
tn = 400.  # Final time
dt = 4.2  # Timestep size in ms
nt = int(1 + (tn - t0) / dt)  # Number of timesteps
time_values = np.linspace(t0, tn, nt)  # Discretized time axis

grid = Grid(shape=(120, 120), extent=(1800., 1800.))

# Define source geometry (slightly above the center)
src = RickerSource(name='src', grid=grid, f0=0.01, time=time_values)
src.coordinates.data[0, :] = [900., 550.]

u = TimeFunction(name='u', grid=grid, space_order=2, time_order=2)
m = Function(name='m', grid=grid)
m.data[:] = 1. / 1.5**2

eqn = Eq(m * u.dt2 - u.laplace)
stencil = solve(eqn, u.forward)[0]
update = Eq(u.forward, stencil)

source = src.inject(field=u.forward, expr=src * dt**2 / m)
op = Operator([update] + source)

# Run for warm-up time; make sure it's % 3
op(t=51, dt=dt)

np.save('wavefield', u.data)
