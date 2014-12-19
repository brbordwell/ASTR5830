import numpy as np
import pdb
from dedalus2.public import *

import logging
logger = logging.getLogger(__name__.split('.')[-1])

class cyclic_predation:
    def __init__(self, domain):
        self.domain = domain

    def set_problem(self, k_diff, r_birth, r_death, d, L, noise):
        pop_size = L**d
        Ea = noise()  ;  Eb = noise()  ; Ec = noise()

        logger.info("D = {:g}, ub = {:g}, s = {:g}, N = {:g}, Ea = {:g},Eb = {:g}, Ec = {:g}".format(k_diff,r_birth[0],r_death[0],pop_size,Ea,Eb,Ec))

        if d == 3: 
            problem=ParsedProblem(axis_names=['x', 'y', 'z'],
                            field_names=['a','a_x','a_y','a_z',
                                         'b','b_x','b_y','b_z',
                                         'c','c_x','c_y','c_z',
                                         'p'],
                                         #'u', 'w'],
                            param_names=['D', 
                                         'ua','ub','uc', 
                                         'sa','sb','sc', 
                                         'N', 
                                         'Ea','Eb','Ec'])

        elif d == 2:
            problem=ParsedProblem(axis_names=['x', 'y'],
                                  field_names=['a','a_x','a_y',
                                               'b','b_x','b_y',
                                               'c','c_x','c_y',
                                               'p'],
                                             #  'u','w'],
                                  param_names=['D', 
                                               'ua','ub','uc', 
                                               'sa','sb','sc', 
                                               'N', 
                                               'Ea','Eb','Ec',
                                               'Dx','Dy'])

        # First species
        if d == 3: 
            problem.add_equation("dt(a) - D*(dx(a_x)+dy(a_y)+dz(a_z)) = \
            a*(ua*(1-p)-sa*c) + \
            N**-0.5 * (a*(ua*(1-p)+sa*c))**0.5 * Ea")
        else:
            problem.add_equation("dt(a) - D*(dx(a_x)+dy(a_y)) = \
            a*(ua*(1-p)-sa*c) + \
            N**-0.5 * (((a*(ua*(1-p)+sa*c))**2)**0.5)**0.5 * Ea")

        # Second species
        if d == 3: 
            problem.add_equation("dt(b) - D*(dz(b_z)+dy(b_y)+dx(b_x)) = \
            b*(ub*(1-p)-sb*a) + \
            N**-0.5 * (((b*(ub*(1-p)+sb*a))**2)**0.5)**0.5 * Eb")
        else:
            problem.add_equation("dt(b) - D*(dx(b_x)+dy(b_y)) = \
            b*(ub*(1-p)-sb*a) + \
            N**-0.5 * (((b*(ub*(1-p)+sb*a))**2)**0.5)**0.5 * Eb")



        # Third species
        if d == 3: 
            problem.add_equation("dt(c) - D*(dx(c_x)+dy(c_y)+dz(c_z)) = \
            c*(uc*(1-p)-sc*b) + \
            N**-0.5 * (c*(uc*(1-p)+sc*b))**0.5 * Ec")
        else:
            problem.add_equation("dt(c) - D*(dx(c_x)+dy(c_y)) = \
            c*(uc*(1-p)-sc*b) + \
            N**-0.5 * (((c*(uc*(1-p)+sc*b))**2)**0.5)**0.5 * Ec")


        # Keeping overall species density within capacity
        problem.add_equation("p - a - b - c = 0")
        #problem.add_equation("Integrate(p) - 1 = 0")

        # Getting those spatial derivatives done
        problem.add_equation("dx(a) - a_x = 0")
        problem.add_equation("dy(a) - a_y = 0")
        if d == 3: problem.add_equation("dz(a) - a_z = 0")

        problem.add_equation("dx(b) - b_x = 0")
        problem.add_equation("dy(b) - b_y = 0")
        if d == 3: problem.add_equation("dz(b) - b_z = 0")

        problem.add_equation("dx(c) - c_x = 0")
        problem.add_equation("dy(c) - c_y = 0")
        if d == 3: problem.add_equation("dz(c) - c_z = 0")

        #problem.add_equation("u = D/Dx")
        #problem.add_equation("w = D/Dy")

        logger.info("Imposing periodic boundary conditions.")

        problem.parameters['D']   = k_diff
        problem.parameters['ua']  = r_birth[0]
        problem.parameters['ub']  = r_birth[1]
        problem.parameters['uc']  = r_birth[2]
        problem.parameters['sa']  = r_death[0]
        problem.parameters['sb']  = r_death[1]
        problem.parameters['sc']  = r_death[2]
        problem.parameters['N']   = pop_size
        problem.parameters['Ea']  = Ea
        problem.parameters['Eb']  = Eb
        problem.parameters['Ec']  = Ec

        problem.expand(self.domain, order=1)

        return problem
