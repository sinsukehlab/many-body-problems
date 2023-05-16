# coding: utf-8

# # Exercise 2 : Finite size scaling for Monte Calro simulation of the 2d Ising model
# 2019, May, Tsuyoshi Okubo  
# 2021, April Tsuyoshi Okubo  
# 2022, May Tsuyoshi Okubo (changed the initialization of random seeds, added connected susceptibility)
# 2023, May Tsuyoshi Okubo (added output of observables as text data, added improved estimator for Binder ratio) 
# 
# This code simulate 2d Ising model on the square lattice, whose Hamiltonian is given by
# $$ \mathcal{H} = -J \sum_{\langle i,j\rangle} S_i S_j - h \sum_i S_i ,$$
# where $S_i = \pm 1$.
# 
# You can select three simulation algorithms explained in the lecture:
# * metropolis
# * heatbath
# * cluster (Swendsen-Wang)
# 
# The main outputs are:
# * Energy: $\langle E\rangle = \langle \mathcal{H}\rangle/N$.
# * Squared magnetization: $\langle M^2\rangle = \langle (\sum_i S_i)^2\rangle/N^2$.
# * Specific heat: $N(\langle E^2\rangle - \langle E\rangle^2)/T$
# * Magnetic susceptibility: $N(\langle M^2\rangle)/T$ (It is correct only for $h=0$)
# * Another Magnetic susceptibility: $N(\langle M^2\rangle - \langle |M|\rangle^2)/T$ 
# * Binder ratio: $\langle M^4\rangle/\langle M^2\rangle^2$
# 
# 
# The code will make graphs of 
# * Energy
# * Specific heat
# * Squared magnetization 
# * Binder ratio
# * Magnetic susceptibility (calculated by the second definition)
# 
# as functions of the temperature for various system sizes $L$.
# 
# #### Parameters for MC simulations
# * themalizatoin: MC steps for thermalization, which is not used for calculating expectation values.
# * observation: MC steps for observation, which is used for calculating expectation values.
# * random_seed: Seed for random numbser generator. For the same random_seed, you always obtain the same result.
# 
# #### Parameters for the finite size scaling
# * Tc_est
# * nu
# * eta
# 
# These parameters are set after the simulations.
# When you repeat finite size scaling using the same simulation data, please run only after the simulation part.
# Instead, you may load simulation data, by setting "Read_simulation_data" is True.
# 
# #### Important notice
# * This code works on python3 with numpy and numba modules. In addition, for saving the file, we need pickle module.

# For usage of this code, please run it with -h option:
#
# python Ising-Ex2.py -h
#
# Note that the command line arguments only support
# limited conditions. If you want to simulate more complicated
# situation, please use jupyter notebook version or modify the code.

import numpy as np
import pickle
from Ising_lib import *
from matplotlib import pyplot
import argparse


def parse_args():
    Tc = 2.0/np.log(1.0+np.sqrt(2.0)) 
    parser = argparse.ArgumentParser(description='Monte Carlo simulation of the square lattice Ising model')
    parser.add_argument('--L_list',metavar='L_list',dest='L_list', type=int, nargs='+', default=[8,16,32,64],
                        help='the list of sizes of square lattice. (default: L_list = [8, 16, 32, 64])')
    parser.add_argument('-t', '--thermalization',metavar='thermalization',dest='thermalization', type=int,default=10000,
                        help='MC steps for thermalization. (default: 10000)')
    parser.add_argument('-o', '--observation',metavar='observation',dest='observation', type= int,default=50000,
                        help='MC steps for observation. (default: 50000)')
    parser.add_argument('--T_max',metavar='T_max',dest='T_max', type=float,default=2.35,
                        help='Maximum Temperature for simulations. (default: T_max = 2.35)')
    parser.add_argument('--T_min',metavar='T_min',dest='T_min', type=float,default=2.2,
                        help='Minimum Temperature for simulations. (default: T_min = 2.2)') 
    parser.add_argument('--T_step',metavar='T_step',dest='T_step', type=float,default=0.005,
                        help='Temperature step for simulations. (default: T_step = 0.005)') 
    parser.add_argument( '-a','--algorithm', metavar='algorithm',dest='algorithm',default="metropolis",
                         help='Algorithms for MC simulation. You can use "metropolis", "heatbath" or "cluster"(Swendsen-Wang) (default: metropolis)')
    parser.add_argument('-s', '--seed',metavar='seed',dest='seed', type=int,default=11,
                        help='seed for random number generator. (default: seed= 11)')
    parser.add_argument('--Tc',metavar='Tc_est',dest='Tc_est', type=float,default=Tc,
                        help=' Estimate of Tc for the finite size scaling. (default: Tc_est = Tc (exact))')
    parser.add_argument('--nu',metavar='nu',dest='nu', type=float, default=1.0,
                        help=' Estimate of nu for the finite size scaling. (default: nu = 1.0 (exact))')
    parser.add_argument('--eta',metavar='eta',dest='eta', type=float, default=0.25,
                        help=' Estimate of eta for the finite size scaling. (default: eta = 0.25 (exact))')

    parser.add_argument('--read_data',metavar='read_data',dest='read_data', action='store_const',
                        const=True, default=False, 
                        help='Set flag to read existing simulation data without performing simulation. List of Ls and temperatures are also read, while the algorithm is not read. (default: read_data = False)')

    parser.add_argument('--inout_file',metavar='inout_file',dest='inout_file', default="mcdata.dat",
                        help='file name for saving/loading the simulation data. When read_file flag is True, simulation date is loaded from inout_file, otherwise, simulation data is saved onto inout_file (default: inout_file = "mcdata.dat")')
    
    return parser.parse_args()


Tc = 2.0/np.log(1.0+np.sqrt(2.0)) ## The critical temperature of the Ising model 

args = parse_args()
L_list = args.L_list
T_list=np.arange(args.T_min,args.T_max,args.T_step)
T_list_all = []
for i in range(len(L_list)):
    T_list_all.append(T_list)
h = 0.0

algorithm = args.algorithm
random_seed = args.seed
np.random.seed(random_seed)
random_seed_list_all = []
for i in range(len(L_list)):
    random_seed_list_all.append(np.random.randint(2**31,size=len(T_list_all[i])))

thermalization = args.thermalization
observation = args.observation

thermalization_list = []
observation_list = []
for L in L_list:
    thermalization_list.append(thermalization)
    observation_list.append(observation)

Read_simulation_data = args.read_data
data_file = args.inout_file

print("## Simulation conditions:")
if algorithm == "heatbath":
    print("## Algorithm = Heat bath")
elif algorithm == "cluster":
    print("## Algorithm = Swendsen-Wang")
else:
    print("## Algorithm = Metropolis")
print("## L_list = "+repr(L_list))
print("## T_list = "+repr(T_list))
print("## h = "+repr(h))
print("## random seed = "+repr(random_seed))
print("## thermalization steps = "+repr(thermalization))
print("## observation steps = "+repr(observation))
print("## Read_simulation_data = "+repr(Read_simulation_data))
print("## inout file name = "+repr(data_file))


def Calc_averages(mag, mag2, mag2_imp, mag4, mag4_imp, mag_abs, ene,ene2, L,T,alogorithm):

    def variance(e,e2):
        return e2 -e**2
    def binder(m2,m4):
        return m4 / m2 **2

    observation = len(ene)
    N = L**2

    e, e_err = Jackknife(ene,bin_size=max(100,observation//100))
    e2,e2_err = Jackknife(ene2,bin_size=max(100,observation//100))
    m,m_err = Jackknife(mag,bin_size=max(100,observation//100))
    m2,m2_err = Jackknife(mag2,bin_size=max(100,observation//100))
    m4,m4_err = Jackknife(mag4,bin_size=max(100,observation//100))
    c, c_err = Jackknife(ene,bin_size=max(100,observation//100),func=variance, data2=ene2)
    c *= N/T**2
    c_err *= N/T**2
    b, b_err = Jackknife(mag2,bin_size=max(100,observation//100),func=binder, data2=mag4)
    chi, chi_err = Jackknife(mag_abs,bin_size=max(100,observation//100),func=variance, data2=mag2)
    
    chi *= N/T
    chi_err *= N/T
    
    if algorithm == "cluster":
        m2_imp, m2_imp_err = Jackknife(mag2_imp,bin_size=max(100,observation//100))
        b_imp, b_imp_err = Jackknife(mag2_imp,bin_size=max(100,observation//100),func=binder, data2=mag4_imp)
    else:
        m2_imp = 0.0
        m2_imp_err = 0.0
        b_imp = 0.0
        b_imp_err = 0.0
    return e, e_err, m2, m2_err, c, c_err, b,b_err,chi, chi_err,m2_imp, m2_imp_err,b_imp, b_imp_err


if Read_simulation_data:
    f = open(data_file,"rb")
    obs_list_all = pickle.load(f)
    f.close

    Ene_all = obs_list_all[0]
    Ene_err_all = obs_list_all[1]
    Mag2_all = obs_list_all[2]
    Mag2_err_all = obs_list_all[3]
    Mag2_imp_all = obs_list_all[4]
    Mag2_imp_err_all = obs_list_all[5]

    Binder_all = obs_list_all[6]
    Binder_err_all = obs_list_all[7]
    Binder_imp_all = obs_list_all[8]
    Binder_imp_err_all = obs_list_all[9]

    
    C_all = obs_list_all[10]
    C_err_all = obs_list_all[11]
    Chi_all = obs_list_all[12]
    Chi_err_all = obs_list_all[13]
    
    L_list = obs_list_all[14]
    T_list_all = obs_list_all[15]
else:
    ## run simulation

    Ene_all = []
    Ene_err_all = []
    Mag2_all = []
    Mag2_err_all = []
    Mag2_imp_all = []
    Mag2_imp_err_all = []


    Binder_all = []
    Binder_err_all = []
    Binder_imp_all = []
    Binder_imp_err_all = []

    C_all = []
    C_err_all = []
    Chi_all = []
    Chi_err_all = []
    
    for i in range(len(L_list)):
        L = L_list[i]
        Ene = []
        Ene_err = []
        Mag2 = []
        Mag2_err = []
        Binder = []
        Binder_err = []
        C = []
        C_err = []
        Mag2_imp = []
        Mag2_imp_err = []
        Binder_imp = []
        Binder_imp_err = []
        Chi = []
        Chi_err = []

        for j in range(len(T_list_all[i])):
            T = T_list_all[i][j]
            mag, mag2, mag2_imp, mag4, mag4_imp,mag_abs, ene,ene2 = MC(L,T,h,thermalization_list[i],observation_list[i],random_seed_list_all[i][j],algorithm)
            e, e_err, m2, m2_err, c, c_err, b,b_err,chi, chi_err,m2_imp, m2_imp_err,b_imp, b_imp_err = Calc_averages(mag, mag2, mag2_imp, mag4, mag4_imp, mag_abs, ene,ene2, L,T,algorithm)
            
            Ene.append(e)
            Ene_err.append(e_err)

            Mag2.append(m2)
            Mag2_err.append(m2_err)

            C.append(c)
            C_err.append(c_err)

            Binder.append(b)
            Binder_err.append(b_err)
            
            Chi.append(chi)
            Chi_err.append(chi_err)

            Mag2_imp.append(m2_imp)
            Mag2_imp_err.append(m2_imp_err)

            Binder_imp.append(b_imp)
            Binder_imp_err.append(b_imp_err)

        Ene_all.append(Ene)
        Ene_err_all.append(Ene_err)
        Mag2_all.append(Mag2)
        Mag2_err_all.append(Mag2_err)
        Mag2_imp_all.append(Mag2_imp)
        Mag2_imp_err_all.append(Mag2_imp_err)


        Binder_all.append(Binder)
        Binder_err_all.append(Binder_err)
        Binder_imp_all.append(Binder_imp)
        Binder_imp_err_all.append(Binder_imp_err)

        C_all.append(C)
        C_err_all.append(C_err)
        Chi_all.append(Chi)
        Chi_err_all.append(Chi_err)

    ## save to data_file

    f = open(data_file,"wb")
    obs_list_all = [Ene_all,Ene_err_all,Mag2_all,Mag2_err_all,Mag2_imp_all,Mag2_imp_err_all,Binder_all,Binder_err_all,Binder_imp_all,Binder_imp_err_all,C_all,C_err_all,Chi_all,Chi_err_all,L_list,T_list_all]
    pickle.dump(obs_list_all,f)
    f.close




## output observables
print("## output observables as text")
print("## L, T, E, E_err, C, C_err, M2, M2_err, Binder, Binder_err, Susceptibility, Susceptibility_err, (For cluste, M2_imp, M2_imp_err, Binder_imp, Binder_imp_err)")
for i in range(len(L_list)):
    for j in range(len(T_list_all[i])):
        if algorithm == "cluster":
            print(L_list[i], T_list_all[i][j],Ene_all[i][j], Ene_err_all[i][j], C_all[i][j], C_err_all[i][j], Mag2_all[i][j], Mag2_err_all[i][j], Binder_all[i][j], Binder_err_all[i][j], Chi_all[i][j], Chi_err_all[i][j],Mag2_imp_all[i][j], Mag2_imp_err_all[i][j], Binder_imp_all[i][j], Binder_imp_err_all[i][j])
        else:
            print(L_list[i], T_list_all[i][j],Ene_all[i][j], Ene_err_all[i][j], C_all[i][j], C_err_all[i][j], Mag2_all[i][j], Mag2_err_all[i][j], Binder_all[i][j], Binder_err_all[i][j], Chi_all[i][j], Chi_err_all[i][j])
    print("")

    

print("## Simulation end. Start analysis.")
    
# In the following, we perform the finite size scaling of simulation data:
# * Energy
# * Specific heat
# * Squared Magnetization
# * Binder ratio
# * Magnetic susceptibility
# 
# Fisrt, we plot raw date witout scaling.

## plot observables
pyplot.figure()
pyplot.title("Energy")
pyplot.xlabel("$T$")
pyplot.ylabel("$E$")
for i in range(len(L_list)):
    pyplot.errorbar(T_list_all[i],Ene_all[i],yerr=np.array(Ene_err_all[i]),capsize=3,fmt="o",label = "L="+repr(L_list[i]))
pyplot.legend()

pyplot.figure()
pyplot.title("Specific heat")
pyplot.xlabel("$T$")
pyplot.ylabel("$C$")
for i in range(len(L_list)):
    pyplot.errorbar(T_list_all[i],C_all[i],yerr=np.array(C_err_all[i]),capsize=3,fmt="o",label = "L="+repr(L_list[i]))
pyplot.legend()

pyplot.figure()
pyplot.title("Squared magnetization")
pyplot.xlabel("$T$")
pyplot.ylabel("$M^2$")
for i in range(len(L_list)):
    pyplot.errorbar(T_list_all[i],Mag2_all[i],yerr=np.array(Mag2_err_all[i]),capsize=3,fmt="o",label = "L="+repr(L_list[i]))
pyplot.legend()

if algorithm == "cluster":
    pyplot.figure()
    pyplot.title("Squared magnetization:improved")
    pyplot.xlabel("$T$")
    pyplot.ylabel("$M^2$")
    for i in range(len(L_list)):
        pyplot.errorbar(T_list_all[i],Mag2_all[i],yerr=np.array(Mag2_imp_err_all[i]),capsize=3,fmt="o",label = "L="+repr(L_list[i]))
    pyplot.legend()

pyplot.figure()
pyplot.title("Binder ratio")
pyplot.xlabel("$T$")
pyplot.ylabel("b")
for i in range(len(L_list)):
    pyplot.errorbar(T_list_all[i],Binder_all[i],yerr=np.array(Binder_err_all[i]),capsize=3,fmt="o",label = "L="+repr(L_list[i]))
pyplot.legend()

if algorithm == "cluster":
    pyplot.figure()
    pyplot.title("Binder ratio:improved")
    pyplot.xlabel("$T$")
    pyplot.ylabel("b")
    for i in range(len(L_list)):
        pyplot.errorbar(T_list_all[i],Binder_imp_all[i],yerr=np.array(Binder_imp_err_all[i]),capsize=3,fmt="o",label = "L="+repr(L_list[i]))
    pyplot.legend()


pyplot.figure()
pyplot.title("Magnetic susceptibility")
pyplot.xlabel("$T$")
pyplot.ylabel("$\chi$")
for i in range(len(L_list)):
    pyplot.errorbar(T_list_all[i],Chi_all[i],yerr=np.array(Chi_err_all[i]),capsize=3,fmt="o",label = "L="+repr(L_list[i]))
pyplot.legend()

pyplot.show()


# Next, we perform the finite size scaling.
# Note that each quantities obey the following scaling forms:
# 
# $$ E \sim L^{1/\nu-d}f_E((T-T_c)L^{1/\nu}),$$
# $$ C \sim L^{2/\nu-d}f_C((T-T_c)L^{1/\nu}),$$
# $$ M^2 \sim L^{-\eta+2-d}f_{M^2}((T-T_c)L^{1/\nu}),$$
# $$ b \sim f_{b}((T-T_c)L^{1/\nu}),$$
# $$ \chi \sim L^{-\eta+2}f_{\chi}((T-T_c)L^{1/\nu}),$$
# 
# where, $d$ is the spacial dimension. (In our case, $d=2$).
# Thus, when we properly set, $\nu$, $\eta$ and $T_c$, we obtain single curve independent on the system size in the "scaling plot" as $(x = (T-T_c)L^{1/\nu}, y = OL^{x})$ as explained in the lecture.
# 
# Note that the above scaling forms are correct only for the singular part of the observables. Among the above four quantities, the Energy, $E$, containss huge contribution form the regular part (non-sigular part). Thus, the standard finite scalig is expected to fail. On the contrary, the other three quantities basically represent "fluctuation" of the systems, and contributions from the sigular part of the free energy become dominant around the critical tempearture. It means, we expect scaling plot works well for $M^2$, $C$, $b$, and $\chi$.
# 
# In adittion, in two-dimensional Ising model, it is known that the critical exponent $\alpha = 2-\nu d = 0$ and there is a logrithmic correction for the specfic heat as 
# $$ C \sim (\log{L})^xf_C((T-T_c)L^{1/\nu}).$$
# Thus, in our case, the finite size scaling for the specific heat is also expected to fail.
# 
# In conculusion, we will see that when we set proper critical exponents ($\eta$ and $\nu$) and the critical temperature $T_c$ the data are on the single curve for the case of $M^2$, $b$, and $\chi$.
# 
# 
# ### Exercise
# * Let's see what happens if you change $T_c$ as 1% by setting Tc_est = 1.01 Tc
#     * You may see how the finite scaling is sensitive to the critical temperature.
# * How about the critical exponents?
# * Check scattering of data by changing the algorithm.
#     * You may see the cluster algorithm gives us much better result than local updates.

## parameter set
Tc_est = args.Tc_est
eta = args.eta
nu = args.nu

print("## Critical exponents:")
print("## Tc_est = "+repr(Tc_est))
print("## nu = "+repr(nu))
print("## eta = "+repr(eta))

## scaling plot
pyplot.figure()
pyplot.title("Energy: Finite size scaling")
pyplot.xlabel("$(T-T_c)L^{1/\\nu}$")
pyplot.ylabel("$E L^{2-1/\\nu}$")
for i in range(len(L_list)):
    pyplot.errorbar(np.array(T_list_all[i]-Tc_est)*L_list[i]**(1/nu),np.array(Ene_all[i])*L_list[i]**(2.0-1.0/nu),yerr=np.array(Ene_err_all[i])*L_list[i]**(2.0-1.0/nu),capsize=3,fmt="o",label = "L="+repr(L_list[i]))
pyplot.legend()

pyplot.figure()
pyplot.title("Specific heat: Finite size scaling")
pyplot.xlabel("$(T-T_c)L^{1/\\nu}$")
pyplot.ylabel("$C L^{2-2/\\nu}$")
for i in range(len(L_list)):
    pyplot.errorbar(np.array(T_list_all[i]-Tc_est)*L_list[i]**(1/nu),np.array(C_all[i])*L_list[i]**(2.0-2.0/nu),yerr=np.array(C_err_all[i])*L_list[i]**(2.0-2.0/nu),capsize=3,fmt="o",label = "L="+repr(L_list[i]))
pyplot.legend()

pyplot.figure()
pyplot.title("Squared magnetization: Finite size scaling")
pyplot.xlabel("$(T-T_c)L^{1/\\nu}$")
pyplot.ylabel("$M^2L^{\eta}$")
for i in range(len(L_list)):
    pyplot.errorbar(np.array(T_list_all[i]-Tc_est)*L_list[i]**(1/nu),np.array(Mag2_all[i])*L_list[i]**(eta),yerr=np.array(Mag2_err_all[i])*L_list[i]**(eta),capsize=3,fmt="o",label = "L="+repr(L_list[i]))
pyplot.legend()

if algorithm == "cluster":
    pyplot.figure()
    pyplot.title("Squared magnetization:improved: Finite size scaling")
    pyplot.xlabel("$(T-T_c)L^{1/\\nu}$")
    pyplot.ylabel("$M^2L^{\eta}$")
    for i in range(len(L_list)):
        pyplot.errorbar(np.array(T_list_all[i]-Tc_est)*L_list[i]**(1/nu),np.array(Mag2_imp_all[i])*L_list[i]**(eta),yerr=np.array(Mag2_imp_err_all[i])*L_list[i]**(eta),capsize=3,fmt="o",label = "L="+repr(L_list[i]))
    pyplot.legend()

pyplot.figure()
pyplot.title("Binder ratio: Finite size scaling")
pyplot.xlabel("$(T-T_c)L^{1/\\nu}$")
pyplot.ylabel("b")
for i in range(len(L_list)):
    pyplot.errorbar(np.array(T_list_all[i]-Tc_est)*L_list[i]**(1/nu),np.array(Binder_all[i]),yerr=np.array(Binder_err_all[i]),capsize=3,fmt="o",label = "L="+repr(L_list[i]))
pyplot.legend()

if algorithm == "cluster":
    pyplot.figure()
    pyplot.title("Binder ratio:improved: Finite size scaling")
    pyplot.xlabel("$(T-T_c)L^{1/\\nu}$")
    pyplot.ylabel("b")
    for i in range(len(L_list)):
        pyplot.errorbar(np.array(T_list_all[i]-Tc_est)*L_list[i]**(1/nu),np.array(Binder_imp_all[i]),yerr=np.array(Binder_imp_err_all[i]),capsize=3,fmt="o",label = "L="+repr(L_list[i]))
    pyplot.legend()


pyplot.figure()
pyplot.title("Magnetic susceptibility: Finite size scaling")
pyplot.xlabel("$(T-T_c)L^{1/\\nu}$")
pyplot.ylabel("$\chi L^{\eta - 2}$")
for i in range(len(L_list)):
    pyplot.errorbar(np.array(T_list_all[i]-Tc_est)*L_list[i]**(1/nu),np.array(Chi_all[i])*L_list[i]**(eta - 2),yerr=np.array(Chi_err_all[i])*L_list[i]**(eta - 2),capsize=3,fmt="o",label = "L="+repr(L_list[i]))
pyplot.legend()

pyplot.show()






