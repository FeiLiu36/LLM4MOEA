""" interface for integrating problems from other package"""
import numpy as np
from pymoo.core.problem import Problem, ElementwiseProblem
from numpy import  ones
from reproblem import RE21, RE24, RE31, RE32, RE33, RE34, RE37
from pymoo.problems.many.dtlz import generic_sphere


class B3OB(ElementwiseProblem):
    def __init__(self, idx, dim=30, ins=1, **kwargs):
        self.elementwise_evaluation = True
        self.p = biobj_suite.get_problem_by_function_dimension_instance(idx, dim, ins)
        self.nadir_point = self.p.largest_fvalues_of_interest
        self.D, self.M, self.xl, self.xu = dim, 2, -5 * ones(dim, float), 5. * ones(dim, float)
        self.zl, self.zu = (-1., -1.), (1., 1.)
        super().__init__(n_var=dim, n_obj=2, xl=self.xl, xu=self.xu, n_ieq_constr=0)

    def _evaluate(self, x, out, *args, **kwargs):
        # out["F"] = (self.p(x) - self.ideal_point) / (self.nadir_point - self.ideal_point)
        out["F"] = self.p(x) / self.nadir_point

    def name(self):
        return 'B3OB_' + str(self.p.id_function) + '_' + str(self.p.dimension)


mop_rw = (RE21, RE24, RE31, RE32, RE33, RE34, RE37)

class RE(ElementwiseProblem):
    def __init__(self, idx):
        self.p = mop_rw[idx]()
        ideal_path = 'ideal_nadir_points/ideal_point_' + self.p.problem_name + '.dat'
        nadir_path = 'ideal_nadir_points/nadir_point_' + self.p.problem_name + '.dat'
        self._nadir = np.fromfile(nadir_path, sep=' ', dtype=float)
        self._ideal = np.fromfile(ideal_path, sep=' ', dtype=float)
        self.zl = np.zeros(self.M, float)
        self.zu = np.ones(self.M, float) + .1  # for hypervolume calculation
        super().__init__(n_var=self.D, n_obj=self.M, xl=self.p.lbound, xu=self.p.ubound, n_ieq_constr=0)

    def name(self):
        return self.p.problem_name

    @property
    def D(self):
        return self.p.n_variables

    @property
    def M(self):
        return self.p.n_objectives

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = (self.p.evaluate(x) - self._ideal) / (self._nadir - self._ideal)
        # f = np.vstack([self.p.evaluate(x[i]) for i in range(len(x))])
        # out["F"] = (f - self._ideal) / (self._nadir - self._ideal)

