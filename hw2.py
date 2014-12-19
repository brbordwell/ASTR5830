#!/User/bin/python

import numpy as np
import matplotlib.pyplot as plt
from ktransit import FitTransit
import ktransit
import pdb
import os
import sys
from glob import glob
import cPickle as pickle
from copy import deepcopy
from numpy.random import shuffle
#import numpy.random.normal as normal

def msmooth(arr, wid):
    smarr = deepcopy(arr)
    for i in xrange(wid,len(arr)-wid,1):
        smarr[i] = np.median(arr[i-wid:i+wid])

    return smarr

def plotme(name, time,flux,err,model,fit):
    fig, ax = plt.subplots(2, sharex=True)
    ax[0].errorbar(time,flux,yerr=err)    
    ax[0].plot(time,flux,'ko',label="Data")
    ax[0].plot(time,model,'r',label="Model",linewidth=2)
    ax[0].set_xlabel("Time (days)")
    ax[0].set_ylabel("Normalized Flux")
    ax[0].set_title("Fitted Transit")
    ax[0].set_ylim(flux.min()-err.max(),flux.max()+err.max())
    ax[0].set_xlim(time.min(),time.max())

    #ax[0].legend(loc='lower right')

    res = flux-model
    ax[1].errorbar(time,res,yerr=err)
    ax[1].plot(time,res,'ko',label="Residuals")
    ax[1].plot(time,fit,'r',label="Syst. Fit",linewidth=2)
    ax[1].set_xlabel("Time (days)")
    ax[1].set_ylabel("Normalized Flux")
    ax[1].set_title("Fitted Residuals")
    ax[1].set_ylim(res.min()-err.max(),res.max()+err.max())
    ax[1].set_xlim(time.min(),time.max())
    #ax[1].legend(loc='lower right')
    
    fig.savefig(name)
    fig.clf()


def fold(master_lst):
#give the info as an index that is the same, the data, and the error, in a list

    sizes = np.array([[i[0],len(i[1])-i[0]] for i in master_lst]).T.min(1)
    master = np.array([np.vstack((i[1][i[0]-sizes[0]:i[0]+sizes[1]],i[2][i[0]-sizes[0]:i[0]+sizes[1]])) for i in master_lst])
    fold_curve = (master[0][0]/master[0][1]+master[1][0]/master[1][1]+master[2][0]/master[2][1])/(1/master[0][1]+1/master[1][1]+1/master[2][1])

    return sizes, [fold_curve, (master[0][1]**2+master[1][1]**2+master[2][1]**2)**0.5]


# It became apparent that this was easier than my original formatting based on problem 1 
def fit_data(data,
             star=[1.4,0,0,0,0,0,1], 
             planet=[365,0.0,0.0,.009155, 0.0, 0.0, 0.0],
             star_vary=[''],
             planet_vary=[''], auto=True, fold=False):
    """solves problems in assignment 2
       star = stellar density, limb darkening parameters (4), dilution, zpt
       planet = T0, period, impact, rprs, ecosw, esinw, occ
    """
    

    # Estimating basic input parameters, unless we give them in the call
    if auto:
        box = 5; sm_flux = msmooth(data[1],box)
        
        star[-1] = np.median(sm_flux) # Establishing photometric zero-point
        if fold:
            star[-1] += np.std(data[1])

        ind = np.where(sm_flux < star[-1]-np.std(data[1]))[0]
        ind = ind[np.where(ind > box)]
        ind = ind[0]
        planet[0] = data[0][ind] # Estimating something close to a transit midpoint
        if fold: 
            if star[0] > 15: planet[0] = 0.8 
    
        planet[3] = (data[1].max()-data[1].min())**0.5 # Estimating the depth

    
    # Performing the actual fit
    fitT = FitTransit()
    fitT.add_guess_star(rho=star[0],
                        ld1=star[1], ld2=star[2],ld3=star[3],ld4=star[4],
                        dil=star[5], zpt=star[6])
    fitT.add_guess_planet(T0=planet[0], period=planet[1],impact=planet[2],
                          rprs=planet[3],ecosw=planet[4],esinw=planet[5],
                          occ=planet[6])
    fitT.add_data(time=data[0],flux=data[1],ferr=data[2])
    fitT.free_parameters(star_vary, planet_vary)

    fitT.do_fit()
    return fitT.__dict__['fitresultstellar'], \
        fitT.__dict__['fitresultplanets'], \
        fitT.transitmodel, star, planet




def derive(data, model, planet, star, fit_p, fit_s, M_star, Ks, prob2=False,
           prob2_parms = None):

#data, rprs, model, T0, b, sig_b, P, sig_P, rho, sig_rho, M_star, Ks):
    val_p = np.mean(fit_p, 0) ; std_p = np.std(fit_p, 0)
    val_s = np.mean(fit_s, 0) ; std_s = np.std(fit_s, 0)
    T0 = val_p[0] ; T0_est = std_p[0]
    rprs = val_p[1] ; sig_est = std_p[1]

    P = planet[1] ; sig_P = T0_est*2
    print("Period = {d}, d = {s}".format(d=P,s=sig_P) )
    if prob2:
        b = planet[2] ; sig_b = 0
        print("b = {d}, d = {s}".format(d=b,s=sig_b) )
        rho = star[0] ; sig_rho = 0
    else:
        b = val_p[2] ; sig_b = std_p[2]
        print("b = {d}, d = {s}".format(d=b,s=sig_b) )
    rho = val_s[0] ; sig_rho = std_s[0]
    print("rho_s = {d}, d = {s}".format(d=rho,s=sig_rho) )


    sig = 1./(1./data[2]**2).sum()**0.5
    depth = rprs**2 ; sig_depth = sig ; sig_est *= 2 * depth
    print("delta = {d}, d = {s}, d_est = {se}".format(d=depth,s=sig, se=sig_est) )

    dt = np.abs(data[0][1]-data[0][0]) 
    temp = model[np.where(model < model.max())]
    T = (len(temp)-1)*dt
    temp = temp[np.where(temp > model.min())]
    tau = (len(temp)-1)*dt/2.
    T -= tau

    # page 68
    sig_tau = sig/depth*T*(6*tau/T)**0.5
    print("tau = {d}, d = {s}".format(d=tau,s=sig_tau) )

    sig_T = sig/depth*T*(2*tau/T)**0.5
    print("T = {d}, d = {s}".format(d=T,s=sig_T) )

    Tc = T0 ; sig_Tc = sig/depth*T*(tau/(2*T))**0.5
    print("Tc = {d}, d = {s}, d_est = {se}".format(d=Tc,s=sig_Tc,se=T0_est) )
    # for depth << 1, weak limb darkening, underestimates in case of limb darkening

    # page 61, eqn 32 to find period P
    
    # other parameters
    sig_rprs = sig_depth/(2*rprs)
    print("rprs = {d}, d = {s}".format(d=rprs,s=sig_rprs) )

    x = ((1-rprs)**2-b**2)**0.5 / (b*np.sin(np.pi*T/P))
    if prob2:
        i = prob2_parms[0]
        sig_i = 0
    else:
        i = np.arctan(x)/np.pi*180.
        sig_i = x/(1+x**2) *(
            ((2*sig_rprs)**2+4*sig_b**2)/4/((1-rprs)**2-b**2)**2 + \
            (sig_b**2/b**2+(np.cos(np.pi*T/P)/np.sin(np.pi*T/P))**2*\
             (sig_T**2/T**2+sig_P**2/P**2))) ** 0.5 \
            /np.pi*180.
    print("i [deg] = {d}, d = {s}".format(d=i,s=sig_i) )

    if prob2:
        rsa = prob2_parms[1]
    else:
        rsa = np.cos(i*np.pi/180.)/b 
    sig_rsa = rsa * ((sig_i*np.pi/180.)**2*np.tan(i/180.*np.pi)**2+sig_b**2/b**2)**0.5
    print("R/a = {d}, d = {s}".format(d=rsa,s=sig_rsa) )

    rs = (M_star/rho*3/(4*np.pi))**(1/3.)/6.958e10 
    sig_rs = rs*sig_rho/rho/3.
    print("Rs [Rsun] = {d}, d = {s}".format(d=rs,s=sig_rs) )


    G = 6.67e-8 # cgs
    gs = (M_star*G)/(rs**2*(6.958e10)**2)/(27.94*980.)
    sig_gs = gs*2*sig_rs/rs
    print("g_s [cgs] = {d}, d = {s}".format(d=gs,s=sig_gs) )

    rp = rprs*rs*6.958e10/6.9911e9
    sig_rp = rp* (sig_rprs**2/rprs**2 + sig_rs**2/rs**2)**0.5
    print("rp [Rj] = {d}, d = {s}".format(d=rp,s=sig_rp) )

    a = rs/rsa*6.958e10/1.49598e13 ; sig_a = a*(sig_rsa**2/rsa**2+sig_rs**2/rs**2)**0.5
    print("a [AU]= {d}, d = {s}".format(d=a, s=sig_a) )    

    P *= 24*3600.  ;  sig_P *= 24*3600
    #rho_p = (1/rprs)**3 * (3*np.pi/(G*P**2)*(1/rsa)**3-rho)
    #sig_rhop = rho_p * (9*sig_rprs**2/rprs**2 + 
    #                    ((rprs**3/rho_p)+rho)*(4*sig_P**2/P**2+9*sig_rsa**2/rsa**2)
    #                    /(rprs**3/rho_p)**2 )**0.5
    #print("rho_p [g/cm^3] = {d}, d = {s}".format(d=rho_p,s=sig_rhop) )


    g_p = 2*np.pi/P * Ks / (np.sin(i*np.pi/180.)*(rprs*rsa)**2)/2479.
    sig_gp = g_p * (sig_P**2/P**2 + (np.cos(i*np.pi/180.) \
            /np.sin(i*np.pi/180.))**2 *sig_i**2 + 4*(sig_rprs**2/rprs**2+\
            sig_rsa**2/rsa**2))**0.5 

    print("g_p [cm/s^2]= {d}, d = {s}".format(d=g_p,s=sig_gp) )

    m_p = M_star**(2/3.)*Ks/np.sin(i*np.pi/180.)*(P/(2*np.pi*G))**(1/3.)/1.898e30
    sig_mp = m_p*((sig_i*np.pi/180.*np.cos(i*np.pi/180.))**2/\
              np.sin(i*np.pi/180.)**2 +sig_P**2/P**2)**0.5
    print("m_p [Mj]= {d}, d = {s}".format(d=m_p,s=sig_mp) )
    
    rho_p = m_p*1.898e30/(4*np.pi*(rp*6.9911e9)**3/3.)/1.33
    sig_rhop = rho_p*(sig_mp**2/m_p**2+9*sig_rp**2/rp**2)**0.5
    print("rho_p [g/cm^3] = {d}, d = {s}".format(d=rho_p,s=sig_rhop) )


def detrend(time, residuals, N):
    # detrending
    x = np.vstack(tuple([time**i for i in xrange(N)]))
    coeff = np.dot(np.dot(residuals,x.T),np.linalg.inv(np.dot(x,x.T)))
    trend_fit = np.dot(x.T,coeff)
    print("The coefficients of the detrending are {cof}".format(cof=coeff))
    return coeff, trend_fit

def pull(planet, star, planet_vary, star_vary):
    
    planet = planet['pnum0']
    p = []
    for a in planet_vary:
        p.append(planet[a])
    p= np.array(p)

    s = []
    for a in star_vary:
        s.append(star[a])
    s= np.array(s)

    return p, s

def refit(data, model, star, planet, star_vary, planet_vary):
    res = data[1]-model 
    for i in xrange(10): shuffle(res)
    pseudo = deepcopy(data); pseudo[1] = model+res
    
    star, planet, model,gs,gp = fit_data(pseudo, star, planet, star_vary, planet_vary, auto=False)
    p, s = pull(planet, star, planet_vary, star_vary)
    
    return p,s



def solve_prob(filename, star=[1.4,0,0,0,0,0,1], 
               planet=[365,0.0,0.0,.009155, 0.0, 0.0, 0.0],
               star_vary=[''],
               planet_vary=[''],fld=False, no_vary=False):

    
    f = glob('*'+str(filename)+'*.txt') #locating the files for the problem
    
    # reading in the data (folding it if we're that cool)
    if fld:
        if no_vary:
            master=[] ; file = f[0]
            with open(file, 'rb') as ofile:
                data = np.genfromtxt(ofile,skip_header=3, delimiter='     ').T
            T0= planet[0]  ;  P = planet[1]
            n = int(np.floor((data[0][-1]-T0)/P))

            for i in xrange(1,n+1):
                T = T0+(i-1)*P
                ind1 = np.where(np.abs(data[0]-T) == 
                               np.abs(data[0]-T).min())[0][0]
                ind2 = np.where(np.abs(data[0]-T+P/2.) == 
                                np.abs(data[0]-T+P/2.).min())[0][0]
                ind3 = np.where(np.abs(data[0]-T-P/2.) == 
                                np.abs(data[0]-T-P/2.).min())[0][0]


                master.append([ind1-ind2, data[1][ind2:ind3+1],data[2][ind2:ind3+1]])
                
            s, folded =fold(master)
            time = data[0][ind1-s[0]:ind1+s[1]]
            time -= time.min()
            data = np.array([time,folded[0],folded[1]])
            f = f[0].replace('2','_fold')
        else:
            master=[]
            sr = star  ;  pt = planet
            for file in f:
                with open(file, 'rb') as ofile:
                    data = np.genfromtxt(ofile,skip_header=3, delimiter='     ').T

                data[2] = data[2]/np.median(data[1])
                data[1] = data[1]/np.median(data[1])
                star, planet, model, guess_s, guess_p = fit_data(data, sr, pt, star_vary, planet_vary)
                coeff, trend = detrend(data[0],data[1]-model,2)
                data[1]-= trend
                p,s = pull(planet, star, planet_vary, star_vary)
                T0 = p[0]
                ind = np.where(np.abs(data[0]-T0) == 
                               np.abs(data[0]-T0).min())[0][0]
                master.append([ind,data[1],data[2]])
        
            s, folded =fold(master)
            time = data[0][ind-s[0]:ind+s[1]]
            time -= time.min()
            data = np.array([time,folded[0],folded[1]])

            f = f[0].replace('1','_fold')
            star = guess_s  ; planet = guess_p

    else:
        f = f[0]
        with open(f, 'rb') as ofile:
            data = np.genfromtxt(ofile,skip_header=3, delimiter='     ').T
    # norm the flux values
    data[2] = data[2]/np.median(data[1])
    data[1] = data[1]/np.median(data[1])

    # Fitting the model and saving the results
    if fld: 
        fald = True
    else:
        fald=False
    star, planet, model, guess_s, guess_p = fit_data(data, star, planet, star_vary, planet_vary,fold=fald)

    print("")
    print("Initial guesses:")
    print("Star: {s}".format(s=guess_s))
    print("Planet: {s}".format(s=guess_p))
    print("")


    coeff, trend = detrend(data[0],data[1]-model,2)
    plotme(f.replace('data','plot1').replace('txt','png'),
           data[0],data[1],data[2],model,trend)

    # Detrending and refitting
    data[1] -= trend
    star, planet, model, gs,gp = fit_data(data, guess_s, guess_p, star_vary, planet_vary, auto=False,fold=fald)
    all_p, all_s = pull(planet, star, planet_vary, star_vary)
    guess_p[0] = all_p[0] ;     
    if not no_vary:
        guess_p[2] = all_p[1]
        guess_s[0] = all_s[0] ;        guess_s[-1] = all_s[1]
    else:
        guess_s[0] = all_s[0]; guess_s[-1] = all_s[1]

    if not no_vary:
        planet_vary.append('impact')#; planet_vary.append('period') ;      
    else:
        star_vary.append('ld1') ; star_vary.append('ld2')#; star_vary.append('rho')
    star, planet, model, gs, gp = fit_data(data, guess_s, guess_p, star_vary, planet_vary, auto=False, fold=fald)
    all_p, all_s = pull(planet, star, planet_vary, star_vary)

    # Refitting for fixed parameters based on seeking convergence
    p, s = refit(data, model, gs, gp, star_vary, planet_vary)
    all_p = np.vstack((all_p, p)) ; all_s = np.vstack((all_s, s))

    while ((np.abs(all_p.mean(0)-all_p[:-1].mean(0)))/
           np.abs(all_p[:-1].mean(0))).mean() > .01:
        p, s = refit(data, model, gs, gp, star_vary, planet_vary)
        all_p = np.vstack((all_p, p)) ; all_s = np.vstack((all_s, s))

    return data, model, gp, gs, all_p, all_s








if __name__=="__main__":
    
    rho_sun = 1.4 # g/cm^3

    # Problem 1
    Ks = 141.24e2 # cm/s
    Mstar = 0.9 ; Rstar = 1.06*Mstar**0.945 # Demircan Kahraman 1990 
    rho_star = rho_sun*Mstar/Rstar**3
    Mstar *= 1.99e33
    lst = ['prob1_data1','prob1_data2','prob1_data3']
    for i in lst:
        print("For {file}, the fitted values were achieved:".format(file=i))
        print("")
        data, model, planet, star, all_p, all_s = solve_prob(
            i, star=[rho_star, 0.0,0.0,0.0,0.0, 0.0, 1.0],
            planet=[0, 3.673, 0.01, .1, 0.0, 0.0, 0.0],
            star_vary = ['rho','zpt'],
            planet_vary = ['T0', 'rprs'])

        derive(data, model, planet, star, all_p, all_s, Mstar, Ks)
        print("")    ;    print("")    ;    print("")


    print("For {file}, the fitted values were achieved:".format(file="the folded data"))
    print("")

    data, model, planet, star, all_p, all_s = solve_prob(
        'prob1_data', star=[rho_star, 0.0,0.0,0.0,0.0, 0.0, 1.0],
        planet=[0, 3.673, 0.01, .1, 0.0, 0.0, 0.0],
        star_vary = ['rho','zpt'],
        planet_vary = ['T0', 'rprs'],fld=True)

    derive(data, model, planet, star, all_p, all_s, Mstar, Ks)
    print("")    ;    print("")    ;    print("")



    # Problem 2
    # Spectral Type M5-6 http://iopscience.iop.org/1538-4357/461/1/L51/fulltext/
    # Teff 2700-2900K  http://arxiv.org/pdf/1304.4072v1.pdf
    # quadratic law approximated with D Sing 2010 for 3500 K
    Ks = 0.98e2 # cm/s
    Mstar = 0.15 ; Rstar = 1.06*Mstar**0.945 ; rho_star = rho_sun*Mstar/Rstar**3
    rho_star = (58.523)**3*3*np.pi/(6.67e-8*(12.1644*24.*3600)**2)
    Rstar = (3*Mstar*1.99e33/rho_star/(4*np.pi))**(1/3.)/6.958e10
    b = 58.523*np.cos(89.193*np.pi/180.)

    #First running it like problem 1 because I can
    print("For {file}, the fitted values were achieved:".format(file="the problem 2 data"))
    data, model, planet, star, all_p, all_s = solve_prob(
        'prob2_data', star=[rho_star, 0.5254, 0.1117,0.0,0.0, 0.0, 1.0],
        planet=[0, 12.164450, b, .001, 0.0, 0.0, 0.0],
        star_vary = ['rho','zpt'],
        planet_vary = ['T0', 'rprs'], no_vary=True)
    print("")
    Mstar *= 1.99e33
    derive(data, model, planet, star, all_p, all_s, Mstar, Ks,prob2=True,
           prob2_parms=[89.193, 1/58.523])
    print("")    ;    print("")    ;    print("")
            

    print("For {file}, the fitted values were achieved:".format(file="the folded problem 2 data"))

    data, model, planet, star, all_p, all_s = solve_prob(
        'prob2_data', star=[rho_star, 0.0,0.0,0.0,0.0, 0.0, 1.0],
        planet=[all_p[:,0].mean(), 12.164450, b, 0.001, 0.0, 0.0, 0.0],
        star_vary = ['rho','zpt'],
        planet_vary = ['T0', 'rprs'],fld=True,no_vary=True)
    derive(data, model, planet, star, all_p, all_s, Mstar, Ks,prob2=True,
           prob2_parms=[89.193, 1/58.523])

    #Then doing what I'm actually supposed to
    file = glob('*prob2*data.txt')[0]
    with open(file, 'rb') as ofile:
        data = np.genfromtxt(ofile, skip_header=3,delimiter='     ').T
    data[2] = data[2]/np.median(data[1])
    data[1] = data[1]/np.median(data[1])

    f = glob('*model*')  ;  chi2 = [] ; rprs = [] ; err = []
    for file in f:
        with open(file, 'rb') as ofile:
            rprs.append(float(ofile.readline().replace('\n','').replace('Rp/Rs = ','')))
            model = np.genfromtxt(ofile,skip_header=1).T
        chi2.append(((data[1][1:]-model)**2/data[2][1:]**2).sum())
        err.append(((data[1][1:]-model)/data[2][1:]).sum())
        #How does error come into this?

    pdb.set_trace()
