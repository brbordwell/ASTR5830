import numpy as np
import time
import os
import sys
import equations_bio_N as equations
import pdb
import numpy.random as npr
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)

from dedalus2.public import *
from dedalus2.tools  import post
from dedalus2.extras import flow_tools
#from dedalus2.extras.checkpointing import Checkpoint

initial_time = time.time()

logger.info("Starting Dedalus script {:s}".format(sys.argv[0]))

# save data in directory named after script
data_dir = sys.argv[0].split('.py')[0]+'/'

k_diff = 3e-6#float(sys.argv[1])
r_birth = np.zeros(3)+1.#float(sys.argv[2]) # For now, just replicating figure 17
r_death = np.zeros(3)+1.#float(sys.argv[3])
L = 100#float(sys.argv[4])
d = 2.#float(sys.argv[5])
noise = lambda: np.random.normal()

nx = np.int(L*3/2)
ny = np.int(L*3/2)
# nz = np.int(L*3/2)

# Set domain
Lx = 1#nx
Ly = 1#ny
#Lz = 1

x_basis = Fourier(nx,   interval=[0., Lx], dealias=2/3)
y_basis = Fourier(ny,   interval=[0., Ly], dealias=2/3)
#z_basis = Fourier(nz,   interval=[0., Lz], dealias=2/3)

domain = Domain([x_basis, y_basis], grid_dtype=np.float64)
#domain = Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64)

if domain.distributor.rank == 0:
  if not os.path.exists('{:s}/'.format(data_dir)):
    os.mkdir('{:s}/'.format(data_dir))


TS = equations.cyclic_predation(domain)
pde = TS.set_problem(k_diff, r_birth, r_death, d, L, noise)

ts = timesteppers.RK443
cfl_safety_factor = 0.1*4

# Build solver
solver = solvers.IVP(pde, domain, ts)

x = domain.grid(0)
y = domain.grid(1)



# initial conditions
a = solver.state['a']
b = solver.state['b']
c = solver.state['c']
#u = solver.state['u']
#w = solver.state['w']


solver.evaluator.vars['Lx'] = Lx
solver.evaluator.vars['Ly'] = Ly
#solver.evaluator.vars['Lz'] = Lz

# Islands
def randrange(a,b,n):
  rng = np.linspace(a,b,n)
  return npr.choice(rng)

def island(nr=10, n=[nx,ny], blockout=None):

  island = np.zeros((2*nr,2*nr))
  for i in np.linspace(0,2*np.pi,200,endpoint=True):
    for j in range(nr):
      island[int(j*np.cos(i)+nr),int(j*np.sin(i)+nr)] = np.exp(-(j**2)/(nr/3.)**2)
  grid = np.zeros((n[0],n[1]))
  #grid = np.zeros((nx,ny,nz))

  if blockout != None:
    inds = np.array(np.where(grid[nr:-nr,nr:-nr]+1)).T+nr
    for i in range(5): npr.shuffle(inds)
    
    for i in inds:
      dist = ((blockout-i)**2).sum(1)**0.5
      if np.where(dist > nr)[0].shape == len(dist): break

    indx, indy = i
  else:
    indx = randrange(nr,n[0]-nr,n[0]-2*nr)
    indy = randrange(nr,n[1]-nr,n[1]-2*nr)

  grid[indx-nr:indx+nr,indy-nr:indy+nr] = island
  #grid[indx-nr:indx+nr-1,indy-nr:indy+nr-1,indz-nr:indz+nr-1]

  return grid


def rand_disp(n=[nx,ny],npop=3,one=None,block=None):
  grid = np.zeros((n[0],n[1]))
  if block != None:
    inds = np.array(np.where(grid+1-block)).T
  else:
    inds = np.array(np.where(grid+1)).T
  for i in range(5): npr.shuffle(inds)
  ni = len(inds)

  cuts = [randrange(0, 1, 1000) for i in range(npop)]
  grids=[]
  
  if one != None:
    i = one
    x = np.array(grid)
    for j in np.linspace(i/3.*ni,ni*(i+cuts[i])/3.,ni*(i+cuts[i])/3.-i/3.*ni):
      x[inds[j][0],inds[j][1]] += 1
    grids.append(x)
  else:
    for i in range(npop):
      x = np.array(grid)
      for j in np.linspace(i/3.*ni,ni*(i+cuts[i])/3.,ni*(i+cuts[i])/3.-i/3.*ni):
        x[inds[j][0],inds[j][1]] += 1
      grids.append(x)

  return grids


island_opt=False
if island_opt:
# Island method
  island_a = island()
  a['g'] = island_a
  island_b = island(blockout=np.array(np.where(island_a)).T)
  b['g'] = island_b
  island_c = island(blockout=np.array(np.where(island_a+island_b)).T)
  c['g'] = island_c
else:
# Random dispersion (unnatural, but who knows...more chemical?)
  a['g'] = rand_disp(one=0)
  b['g'] = rand_disp(one=1,block=a['g'])
  c['g'] = rand_disp(one=2,block=a['g']+b['g'])


#u['g'] = k_diff/(L/nx)
#w['g'] = k_diff/(L/ny)


# integrate parameters
bit=1
max_dt = k_diff/(L/nx)**2*bit
cfl_cadence = 1
#cfl = flow_tools.CFL_conv_2D(solver, max_dt, cfl_cadence=cfl_cadence)

report_cadence = 1
output_time_cadence = max_dt*10
solver.stop_sim_time = 20
solver.stop_iteration= np.inf
solver.stop_wall_time = 0.25*3600

logger.info("output cadence = {:g}".format(output_time_cadence))

analysis_slice = solver.evaluator.add_file_handler(data_dir+"slices", sim_dt=output_time_cadence, max_writes=20, parallel=False)

analysis_slice.add_task("a", name="a")
analysis_slice.add_task("dx(a)", name="a_x")
analysis_slice.add_task("b", name="b")
analysis_slice.add_task("c", name="c")
analysis_slice.add_task("p", name="p")
#analysis_slice.add_task("(dx(w) - dz(u))**2", name="enstrophy")


do_checkpointing=False
if do_checkpointing:
    checkpoint = Checkpoint(data_dir)
    checkpoint.set_checkpoint(solver, wall_dt=1800)

solver.dt = max_dt

start_time = time.time()
while solver.ok:

    # advance
    solver.step(solver.dt)
    
    if solver.iteration % cfl_cadence == 0 and solver.iteration>=2*cfl_cadence:
        domain.distributor.comm_world.Barrier()
        solver.dt = k_diff/(L/nx)**2*bit#cfl.compute_dt(cfl_safety_factor)

        #pdb.set_trace()
    # update lists
    if solver.iteration % report_cadence == 0:
        log_string = 'Iteration: {:5d}, Time: {:8.3e}, dt: {:8.3e}, a_avg: {:8.3e}'.format(solver.iteration, solver.sim_time, solver.dt,np.mean(a['g']))
        logger.info(log_string)
        
end_time = time.time()

# Print statistics
elapsed_time = end_time - start_time
elapsed_sim_time = solver.sim_time
N_iterations = solver.iteration 
logger.info('main loop time: {:e}'.format(elapsed_time))
logger.info('Iterations: {:d}'.format(N_iterations))
logger.info('iter/sec: {:g}'.format(N_iterations/(elapsed_time)))
logger.info('Average timestep: {:e}'.format(elapsed_sim_time / N_iterations))

logger.info('beginning join operation')
if do_checkpointing:
    logger.info(data_dir+'/checkpoint/')
    post.merge_analysis(data_dir+'/checkpoint/')
logger.info(analysis_slice.base_path)
post.merge_analysis(analysis_slice.base_path)

if (domain.distributor.rank==0):

    N_TOTAL_CPU = domain.distributor.comm_world.size
    
    # Print statistics
    print('-' * 40)
    total_time = end_time-initial_time
    main_loop_time = end_time - start_time
    startup_time = start_time-initial_time
    print('  startup time:', startup_time)
    print('main loop time:', main_loop_time)
    print('    total time:', total_time)
    print('Iterations:', solver.iteration)
    print('Average timestep:', solver.sim_time / solver.iteration)
    print('scaling:',
          ' {:d} {:d} {:d} {:d} {:d} {:d}'.format(N_TOTAL_CPU, 0, N_TOTAL_CPU,nx, 0, ny),
          ' {:8.3g} {:8.3g} {:8.3g} {:8.3g} {:8.3g}'.format(startup_time,
                                                            main_loop_time, 
                                                            main_loop_time/solver.iteration, 
                                                            main_loop_time/solver.iteration/(nx*ny), 
                                                            N_TOTAL_CPU*main_loop_time/solver.iteration/(nx*ny)))
    print('-' * 40)


