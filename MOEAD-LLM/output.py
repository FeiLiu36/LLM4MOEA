from pymoo.visualization.scatter import Scatter
from pymoo.util import plotting
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd import IGD
import numpy as np
import os

# output results
def output (res,problemname,dimension,outputfile):
    X, F = res.opt.get("X", "F")

    hist = res.history

    n_evals = []             # corresponding number of function evaluations\
    hist_F = []              # the objective space values in each generation

    for algo in hist:

        # store the number of function evaluations
        n_evals.append(algo.evaluator.n_eval)

        # retrieve the optimum from the algorithm
        opt = algo.opt

        # filter out only the feasible and append and objective space values
        hist_F.append(opt.get("F"))

    # hv convergence
    approx_ideal = F.min(axis=0)
    approx_nadir = F.max(axis=0)
    if len(approx_nadir)==3:
        ref_point = np.array([1.1, 1.1, 1.1])
    else:
        ref_point = np.array([1.1, 1.1])
    metric = Hypervolume(ref_point= ref_point,
                        norm_ref_point=False,
                        zero_to_one=True,
                        ideal=approx_ideal,
                        nadir=approx_nadir)
    hv = [metric.do(_F) for _F in hist_F]
    filename=outputfile+problemname+"_d"+str(dimension)+"_hv.dat"
    file_hv = open(filename,"w")
    for i in range(len(n_evals)):
        file_hv.write("{} {:.4f} \n".format(n_evals[i],hv[i]))
    file_hv.close()

    # igd convergence
    # pf = problem.pareto_front(use_cache=False)
    # metric = IGD(pf, zero_to_one=True)
    # igd = [metric.do(_F) for _F in hist_F]

    # filename=outputfile+problemname+"_d"+str(dimension)+"_igd.dat"
    # file_igd = open(filename,"w")
    # for i in range(len(n_evals)):
    #     file_igd.write("{} {:.4f} \n".format(n_evals[i],igd[i]))
    # file_igd.close()

    #PF_data
    filename=outputfile+problemname+"_d"+str(dimension)+"_PF_opt.dat"
    file_pf = open(filename,"w")
    pf_list = res.F
    for i in range(len(pf_list)):
        for j in range(len(pf_list[i])):
            file_pf.write("{:.4f} ".format(pf_list[i][j]))
        file_pf.write("\n")    
    file_pf.close()

    # PS_data
    filename=outputfile+problemname+"_d"+str(dimension)+"_PS_opt.dat"
    file_ps = open(filename,"w")
    ps_list = res.X
    for i in range(len(ps_list)):
        for j in range(len(ps_list[i])):
            file_ps.write("{:.4f} ".format(ps_list[i][j]))
        file_ps.write("\n")    
    file_ps.close()

    #PF plot
    Scatter().add(res.F).save(outputfile+problemname+"_d"+str(dimension))