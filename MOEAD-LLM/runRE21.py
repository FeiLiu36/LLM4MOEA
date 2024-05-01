from pymoo.algorithms.moo.moead_LLM import MOEAD_LLM
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from problem_interface import RE

from pymoo.util.ref_dirs import get_reference_directions
from output import output

def get_problem(problemname):
    if problemname =="UF1":
        problem = UF1(D=dimension)
    elif problemname =="UF2":
        problem = UF2(D=dimension)
    elif problemname =="UF3":
        problem = UF3(D=dimension)
    elif problemname =="UF5":
        problem = UF5(D=dimension)
    elif problemname =="UF7":
        problem = UF7(D=dimension)
    elif problemname =="UF8":
        problem = UF8(D=dimension)
    elif problemname == "RE21": #  RE24, RE31, RE32, RE33, RE34, RE37
        problem = RE(0)
    elif problemname == "RE22":
        problem = RE(1)
    elif problemname == "RE23":
        problem = RE(2)
    elif problemname == "RE24":
        problem = RE(3)
    elif problemname == "RE25":
        problem = RE(4)
    elif problemname == "RE31":
        problem = RE(5)
    elif problemname == "RE32":
        problem = RE(6)
    else:
        problem = get_problem(problemname,n_var=dimension)
    return problem

def get_algorithm(algorithmname):
    if algorithmname == "MOEAD_LLM":
        algorithm = MOEAD_LLM(
            ref_dirs,
            n_neighbors=neighbor_size,
            prob_neighbor_mating=0.7,
            debug_mode = debug_mode,
            model_LLM = model_LLM,
            endpoint = endpoint,
            key = key,
            out_file = out_filename_gpt,
        )
    elif algorithmname == "MOEAD":
        algorithm = MOEAD(
            ref_dirs,
            n_neighbors=neighbor_size,
            prob_neighbor_mating=0.7,
        )
    elif algorithmname == "NSGAII":
        algorithm = NSGA2(pop_size= pop_size)

    return algorithm


if __name__ == '__main__':

    algorithmname = "MOEAD_LLM" # NSGA-II, MOEAD, MOEAD_LLM
    record_gpt_solution = False # record the input and oupt of each run of gpt to learn a general linear operator
    model_LLM = "gpt-3.5-turbo" #"model": "gpt-3.5-turbo",
                            #"model": "gpt-4-0613",
				#gemini-pro
    endpoint = "api.chatanywhere.tech"
    key = "sk-BlawLfF8RIJCxpVZXK8Tt9El7uNbcmHmDosImqsFhqpgzmWM" # your key
    debug_mode = False

    pop_size = 50
    neighbor_size = 10
    n_gen = 20
    n_partition = 10 # for three objective only


    problems = ['RE21']

    n_repeat = 3
    for prob in problems:
        for n in n_repeat:

            problemname = prob
            dimension = 4

            outputfile = problemname+"/results"+str(n)+"/"

            if record_gpt_solution:
                out_filename_gpt=outputfile+problemname+"_d"+str(dimension)+"_gpt_sample.dat"
                file = open (out_filename_gpt,"w")
                file.close()
            else:
                out_filename_gpt= None

            if problemname in ["RE31","RE32","RE33","RE34","RE37"]:
                ref_dirs = get_reference_directions("uniform", 3, n_partitions=n_partition)   
            else: 
                ref_dirs = get_reference_directions("uniform", 2, n_partitions=pop_size)

            problem = get_problem(problemname)

            algorithm = get_algorithm(algorithmname)

            
            res = minimize(problem,
                        algorithm,
                        ('n_gen', n_gen),
                        seed=2023*n,
                        save_history=True,
                        verbose=True)
            
            output(res,problemname,dimension,outputfile)

