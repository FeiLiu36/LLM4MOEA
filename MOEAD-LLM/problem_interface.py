""" interface for integrating problems from other package"""
import numpy as np
from pymoo.core.problem import Problem, ElementwiseProblem
from numpy import abs, sum, prod, log10, exp, power, sqrt, linspace, vstack, where, zeros, ones, sin, cos, arange, pi, \
    e, arccos, average
from reproblem import RE21,RE22,RE23, RE24,RE25, RE31, RE32, RE33, RE34, RE37
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


mop_rw = (RE21,RE22,RE23, RE24,RE25, RE31, RE32, RE33, RE34, RE37)

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



class UF1(Problem):
    def __init__(self, D=3, use_built_in=False):
        self.D, self.M, self.xl, self.xu, self.zl, self.zu = D, 2, -ones(D, float), ones(D, float), (0, 0), (1, 1)
        self.xl[0] = 0
        self.use_built_in = use_built_in
        #self.p = problem(cec2009(1, dim=self.D))
        super().__init__(n_var=self.D, n_obj=self.M, n_ieq_constr=0, xl=self.xl, xu=self.xu)

    def _evaluate(self, x, out, *args, **kwargs):
        if self.use_built_in:
            out["F"] = np.array([self.p.fitness(x[i]) for i in range(x.shape[0])])
            return
        x1 = x[:, 0]
        x1T = np.transpose([x1])
        #print(arange(self.D))
        j = where(1 - arange(self.D) % 2)[0][1:]
        xj = x[:, j]
        j = np.repeat([j + 1], len(x), axis=0)
        f1 = x1 + 2 / len(j[0]) * sum((xj - sin(6 * pi * x1T + j * pi / self.D)) ** 2, axis=1)
        j = where(arange(self.D) % 2)[0]
        xj = x[:, j]
        j = np.repeat([j + 1], len(x), axis=0)
        f2 = 1 - sqrt(x1) + 2 / len(j[0]) * sum((xj - sin(6 * pi * x1T + j * pi / self.D)) ** 2, axis=1)
        out["F"] = vstack(([f1], [f2])).T

    def _calc_pareto_front(self, n_pareto_points=100):
        f0 = linspace(0., 1., n_pareto_points, True)
        f1 = 1 - sqrt(f0)
        return vstack((f0, f1)).T


class UF2(Problem):
    def __init__(self, D=3, use_built_in=False):
        self.D, self.M, self.xl, self.xu, self.zl, self.zu = D, 2, -ones(D, float), ones(D, float), (0, 0), (1, 1)
        self.xl[0] = 0
        self.use_built_in = use_built_in
        #self.p = problem(cec2009(2, dim=self.D))
        super().__init__(n_var=self.D, n_obj=self.M, n_ieq_constr=0, xl=self.xl, xu=self.xu)

    def _evaluate(self, x, out, *args, **kwargs):
        if self.use_built_in:
            out["F"] = np.array([self.p.fitness(x[i]) for i in range(x.shape[0])])
            return
        D = self.D
        x1 = x[:, 0]
        x1T = np.transpose([x1])
        j = where(1 - arange(self.D) % 2)[0][1:]
        xj = x[:, j]
        j = np.repeat([j + 1], len(x), axis=0)
        yj = xj - (.3 * x1T ** 2 * cos(24 * pi * x1T + 4 * pi * j / D) + .6 * x1T) * cos(6 * pi * x1T + j * pi / D)
        f1 = x1 + 2 / len(j[0]) * sum(yj ** 2, axis=1)
        j = where(arange(self.D) % 2)[0]
        xj = x[:, j]
        j = np.repeat([j + 1], len(x), axis=0)
        yj = xj - (.3 * x1T ** 2 * cos(24 * pi * x1T + 4 * pi * j / D) + .6 * x1T) * sin(6 * pi * x1T + j * pi / D)
        f2 = 1 - sqrt(x1) + 2 / len(j[0]) * sum(yj ** 2, axis=1)
        out["F"] = vstack(([f1], [f2])).T

    def _calc_pareto_front(self, n_pareto_points=100):
        f0 = linspace(0., 1., n_pareto_points, True)
        f1 = 1. - sqrt(f0)
        return vstack((f0, f1)).T


class UF3(Problem):
    def __init__(self, D=3, use_built_in=False):
        self.D, self.M, self.xl, self.xu, self.zl, self.zu = D, 2, zeros(D, float), ones(D, float), (0, 0), (1, 1)
        self.use_built_in = use_built_in
        #self.p = problem(cec2009(3, dim=self.D))
        super().__init__(n_var=self.D, n_obj=self.M, n_ieq_constr=0, xl=self.xl, xu=self.xu)

    def _evaluate(self, x, out, *args, **kwargs):
        if self.use_built_in:
            out["F"] = np.array([self.p.fitness(x[i]) for i in range(x.shape[0])])
            return
        x1 = x[:, 0]
        x1T = np.transpose([x1])
        j = where(1 - arange(self.D) % 2)[0][1:]
        xj = x[:, j]
        j = np.repeat([j + 1], len(x), axis=0)
        yj = xj - power(x1T, .5 * (1 + 3 * (j - 2) / (self.D - 2)))
        f1 = x1 + 4 / len(j[0]) * (2 * sum(yj ** 2, axis=1) - prod(cos(20 * yj * pi / sqrt(j)), axis=1) + 1)
        j = where(arange(self.D) % 2)[0]
        xj = x[:, j]
        j = np.repeat([j + 1], len(x), axis=0)
        yj = xj - power(x1T, .5 * (1 + 3 * (j - 2) / (self.D - 2)))
        f2 = 1 - sqrt(x1) + 4 / len(j[0]) * (2 * sum(yj ** 2, axis=1) - prod(cos(20 * yj * pi / sqrt(j)), axis=1) + 1)
        out["F"] = vstack(([f1], [f2])).T

    def _calc_pareto_front(self, n_pareto_points=100):
        f0 = linspace(0., 1., n_pareto_points, True)
        f1 = 1 - sqrt(f0)
        return vstack((f0, f1)).T


class UF4(Problem):
    # TODO has bug to fix
    def __init__(self, D=3):
        self.D, self.M, self.xl, self.xu, self.zl, self.zu = D, 2, -2 * ones(D, float), 2 * ones(D, float), (0, 0), (1, 1)
        self.xl[0], self.xu[0] = 0, 1
        #self.p = problem(cec2009(4, dim=self.D))
        super().__init__(n_var=self.D, n_obj=self.M, n_ieq_constr=0, xl=self.xl, xu=self.xu)

    def _evaluate(self, x, out, *args, **kwargs):
        '''x1 = x[:, 0]
        x1T = np.transpose([x1])
        j = where(1 - arange(self.D) % 2)[0][1:]
        xj = x[:, j]
        j = np.repeat([j + 1], len(x), axis=0)
        yj = xj - sin(6 * pi * x1T + j * pi / self.D)
        f1 = x1 + 2 / len(j[0]) * sum(self.h(yj), axis=1)
        j = where(arange(self.D) % 2)[0]
        xj = x[:, j]
        j = np.repeat([j + 1], len(x), axis=0)
        yj = xj - sin(6 * pi * x1T + j * pi / self.D)
        f2 = 1 - x1 ** 2 + 2 / len(j[0]) * sum(self.h(yj), axis=1)
        out["F"] = vstack(([f1], [f2])).T'''
        out["F"] = np.array([self.p.fitness(x[i]) for i in range(x.shape[0])])

    def h(self, t):
        return abs(t) / (1 + exp(2 * abs(t)))

    def _calc_pareto_front(self, n_pareto_points=100):
        f0 = linspace(0., 1., n_pareto_points, True)
        f1 = 1 - f0 ** 2
        return vstack((f0, f1)).T


class UF5(Problem):
    def __init__(self, D=3, N=10, eps=.1, use_built_in=False):
        self.D, self.M, self.xl, self.xu, self.zl, self.zu = D, 2, -ones(D, float), ones(D, float), (0, 0), (1, 1)
        self.xl[0] = 0
        self.N = N
        self.eps = eps
        self.use_built_in = use_built_in
        #self.p = problem(cec2009(5, dim=self.D))
        super().__init__(n_var=self.D, n_obj=self.M, n_ieq_constr=0, xl=self.xl, xu=self.xu)

    def _evaluate(self, x, out, *args, **kwargs):
        if self.use_built_in:
            out["F"] = np.array([self.p.fitness(x[i]) for i in range(x.shape[0])])
            return
        x1 = x[:, 0]
        x1T = np.transpose([x1])
        j = where(1 - arange(self.D) % 2)[0][1:]
        xj = x[:, j]
        j = np.repeat([j + 1], len(x), axis=0)
        yj = xj - sin(6 * pi * x1T + j * pi / self.D)
        f1 = x1 + (.5 / self.N + self.eps) * abs(sin(2 * self.N * pi * x1)) + 2 * average(self.h(yj), axis=1)
        j = where(arange(self.D) % 2)[0]
        xj = x[:, j]
        j = np.repeat([j + 1], len(x), axis=0)
        yj = xj - sin(6 * pi * x1T + j * pi / self.D)
        f2 = 1 - x1 + (.5 / self.N + self.eps) * abs(sin(2 * self.N * pi * x1)) + 2 * average(self.h(yj), axis=1)
        out["F"] = vstack(([f1], [f2])).T

    def h(self, t):
        return 2 * (t ** 2) - cos(4 * pi * t) + 1

    def _calc_pareto_front(self, n_pareto_points=100):
        f0 = arange(2 * self.N + 1) / 2 / self.N
        f1 = 1 - f0
        return vstack((f0, f1)).T


class UF6(Problem):
    # TODO has bug to fix
    def __init__(self, D=3):
        self.D, self.M, self.xl, self.xu, self.zl, self.zu = D, 2, -ones(D, float), ones(D, float), (0, 0), (1, 1)
        self.xl[0], self.xu[0] = 0., 1.
        #self.p = problem(cec2009(6, dim=self.D))
        super().__init__(n_var=self.D, n_obj=self.M, n_ieq_constr=0, xl=self.xl, xu=self.xu)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.array([self.p.fitness(x[i]) for i in range(x.shape[0])])

    def _calc_pareto_front(self, n_pareto_points=100):
        f0 = linspace(0., 1., n_pareto_points, True)
        f1 = 1 - f0 ** 2
        return vstack((f0, f1)).T


class UF7(Problem):
    def __init__(self, D=3):
        self.D, self.M, self.xl, self.xu, self.zl, self.zu = D, 2, -ones(D, float), ones(D, float), (0, 0), (1, 1)
        self.xl[0] = 0
        super().__init__(n_var=self.D, n_obj=self.M, n_ieq_constr=0, xl=self.xl, xu=self.xu)

    def _evaluate(self, x, out, *args, **kwargs):
        x1 = x[:, 0]
        x1T = np.transpose([x1])
        j = where(1 - arange(self.D) % 2)[0][1:]
        xj = x[:, j]
        j = np.repeat([j + 1], len(x), axis=0)
        yj = xj - sin(6 * pi * x1T + j * pi / self.D)
        f1 = np.power(x1, 1 / 5) + 2 / len(j[0]) * sum(self.h(yj), axis=1)
        j = where(arange(self.D) % 2)[0]
        xj = x[:, j]
        j = np.repeat([j + 1], len(x), axis=0)
        yj = xj - sin(6 * pi * x1T + j * pi / self.D)
        f2 = 1 - np.power(x1, 1 / 5) + 2 / len(j[0]) * sum(self.h(yj), axis=1)
        out["F"] = vstack(([f1], [f2])).T

    def h(self, t):
        return abs(t) / (1 + exp(2 * abs(t)))

    def _calc_pareto_front(self, n_pareto_points=100):
        f0 = linspace(0., 1., n_pareto_points, True)
        f1 = 1 - f0
        return vstack((f0, f1)).T


class UF8(Problem):
    def __init__(self, D=3, use_built_in=False):
        self.D, self.M, self.xl, self.xu, self.zl, self.zu = D, 3, -2 * ones(D, float), 2 * ones(D, float), (0, 0, 0), (1, 1, 1)
        self.xl[0], self.xl[1], self.xu[0], self.xu[1] = 0, 0, 1, 1
        self.use_built_in = use_built_in
        #self.p = problem(cec2009(8, dim=self.D))
        super().__init__(n_var=self.D, n_obj=self.M, n_ieq_constr=0, xl=self.xl, xu=self.xu)

    def _evaluate(self, x, out, *args, **kwargs):
        if self.use_built_in:
            out["F"] = np.array([self.p.fitness(x[i]) for i in range(x.shape[0])])
            return
        x1 = x[:, 0]
        x2 = x[:, 1]
        x1T = np.transpose([x1])
        x2T = np.transpose([x2])
        j = (1 + arange(self.D) - 1) % 3 == 0
        j[:2] = False
        j = np.where(j)[0]
        xj = x[:, j]
        j = np.repeat([j + 1], len(x), axis=0)
        f1 = cos(.5 * x1 * pi) * cos(.5 * x2 * pi) + 2 / len(j[0]) * sum((xj - 2 * x2T * sin(2 * pi * x1T + j * pi / self.D)) ** 2, axis=1)
        j = (1 + arange(self.D) - 2) % 3 == 0
        j[:2] = False
        j = np.where(j)[0]
        xj = x[:, j]
        j = np.repeat([j + 1], len(x), axis=0)
        f2 = cos(.5 * x1 * pi) * sin(.5 * x2 * pi) + 2 / len(j[0]) * sum((xj - 2 * x2T * sin(2 * pi * x1T + j * pi / self.D)) ** 2, axis=1)
        j = (1 + arange(self.D)) % 3 == 0
        j[:2] = False
        j = np.where(j)[0]
        xj = x[:, j]
        j = np.repeat([j + 1], len(x), axis=0)
        f3 = sin(.5 * x1 * pi) + 2 / len(j[0]) * sum((xj - 2 * x2T * sin(2 * pi * x1T + j * pi / self.D)) ** 2, axis=1)
        out["F"] = vstack(([f1], [f2], [f3])).T

    def _calc_pareto_front(self, ref_dirs):
        return generic_sphere(ref_dirs)


class MMF1(Problem):  # CEC20
    def __init__(self):
        self.D, self.M, self.xl, self.xu, self.zl, self.zu = 2, 2, (1, -1), (3, 1), (0, 0), (1, 1)
        super().__init__(n_var=self.D, n_obj=self.M, n_ieq_constr=0, xl=self.xl, xu=self.xu)

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = abs(x[:, 0] - 2)
        f2 = 1 - sqrt(abs(x[:, 0] - 2)) + 2 * power(x[:, 1] - sin(6*pi*abs(x[:, 0] - 2) + pi), 2)
        out["F"] = vstack(([f1], [f2])).T

    def _calc_pareto_front(self, n_pareto_points=100):
        f0 = linspace(0., 1., n_pareto_points, True)
        f1 = 1 - sqrt(f0)
        return vstack((f0, f1)).T


class MMF2(Problem):  # CEC20  the paper is wrong, refer to the code
    def __init__(self):
        self.D, self.M, self.xl, self.xu, self.zl, self.zu = 2, 2, (0, 0), (1, 2), (0, 0), (1, 1)
        super().__init__(n_var=self.D, n_obj=self.M, n_ieq_constr=0, xl=self.xl, xu=self.xu)

    def _evaluate(self, x, out, *args, **kwargs):
        f1, f2, x1, x2 = x[:, 0], zeros(len(x), float), x[:, 0], x[:, 1]
        lq1, gt1, srx1, sr2 = x1 <= 1, x1 > 1, sqrt(x1), sqrt(2)
        x11, x12, x21, x22, srx11, srx12 = x1[lq1], x1[gt1], x2[lq1], x2[gt1], srx1[lq1], srx1[gt1]
        f2[lq1] = 1-srx11+4*(2*power(x21-srx11, 2)-cos(10*sr2*pi*(x21-srx11))+1)
        f2[gt1] = 1-srx12+4*(2*power(x22-1-srx12, 2)-cos(10*sr2*pi*(x22-1-srx12))+1)
        out["F"] = vstack(([f1], [f2])).T

    def _calc_pareto_front(self, n_pareto_points=100):
        f0 = linspace(0., 1., n_pareto_points, True)
        f1 = 1 - sqrt(f0)
        return vstack((f0, f1)).T


class MMF1E(Problem):
    def __init__(self):
        self.a = e
        self.D, self.M, self.xl, self.xu, self.zl, self.zu = 2, 2, (1, -pow(self.a, 3)), (3, pow(self.a, 3)), (0, 0), (1, 1)
        super().__init__(n_var=self.D, n_obj=self.M, n_ieq_constr=0, xl=self.xl, xu=self.xu)

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = abs(x[:, 0] - 2)
        f2 = zeros(len(x), float)
        pik = where(x[:, 0] < 2)[0]
        x1, x2 = x[pik, 0], x[pik, 1]
        f2[pik] = 1-sqrt(abs(x1 - 2))+2*power(x2-sin(pi*(6*abs(x1-2)+1)), 2)
        pik = where(x[:, 0] >= 2)[0]
        x1, x2 = x[pik, 0], x[pik, 1]
        f2[pik] = 1-sqrt(abs(x1 - 2))+2*power(x2-power(self.a, x1)*sin(pi*(6*abs(x1-2)+1)), 2)
        out["F"] = vstack(([f1], [f2])).T

    def _calc_pareto_front(self, n_pareto_points=100):
        f0 = linspace(0., 1., n_pareto_points, True)
        f1 = 1 - sqrt(f0)
        return vstack((f0, f1)).T


class MMF4(Problem):
    def __init__(self):
        self.D, self.M = 2, 2
        self.xl, self.xu = (-1, 0), (1, 2)
        self.zl, self.zu = (0, 0), (1, 1)
        super().__init__(n_var=self.D, n_obj=self.M, n_ieq_constr=0, xl=self.xl, xu=self.xu)

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = abs(x[:, 0])
        f2 = zeros(len(x), float)
        pik = where(x[:, 1] < 1)[0]
        x1, x2 = x[pik, 0], x[pik, 1]
        f2[pik] = 1 - x1 ** 2 + 2 * (x2 - sin(pi * abs(x1))) ** 2
        pik = where(x[:, 1] >= 1)[0]
        x1, x2 = x[pik, 0], x[pik, 1]
        f2[pik] = 1 - x1 ** 2 + 2 * (x2 - 1 - sin(pi * abs(x1))) ** 2
        out["F"] = vstack((f1, f2)).T

    def _calc_pareto_front(self, n_pareto_points=100):
        f0 = linspace(0., 1., n_pareto_points, True)
        f1 = 1 - f0 ** 2
        return vstack((f0, f1)).T


class MMF5(Problem):
    def __init__(self):
        self.D, self.M = 2, 2
        self.xl, self.xu = (1, -1), (3, 3)
        self.zl, self.zu = (0, 0), (1, 1)
        super().__init__(n_var=self.D, n_obj=self.M, n_ieq_constr=0, xl=self.xl, xu=self.xu)

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = abs(x[:, 0] - 2)
        f2 = zeros(len(x), float)
        pik = where(x[:, 1] <= 1)[0]
        x1, x2 = x[pik, 0], x[pik, 1]
        f2[pik] = 1 - sqrt(abs(x1 - 2)) + 2 * (x2 - sin(6 * pi * abs(x1 - 2) + pi)) ** 2
        pik = where(x[:, 1] > 1)[0]
        x1, x2 = x[pik, 0], x[pik, 1]
        f2[pik] = 1 - sqrt(abs(x1 - 2)) + 2 * (x2 - 2 - sin(6 * pi * abs(x1 - 2) + pi)) ** 2
        out["F"] = vstack((f1, f2)).T

    def _calc_pareto_front(self, n_pareto_points=100):
        f0 = linspace(0., 1., n_pareto_points, True)
        f1 = 1 - sqrt(f0)
        return vstack((f0, f1)).T


class MMF6(Problem):
    def __init__(self):
        self.D, self.M = 2, 2
        self.xl, self.xu = (1, -1), (3, 2)
        self.zl, self.zu = (0, 0), (1, 1)
        super().__init__(n_var=self.D, n_obj=self.M, n_ieq_constr=0, xl=self.xl, xu=self.xu)

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = abs(x[:, 0] - 2)
        f2 = zeros(len(x), float)
        pik = where(x[:, 1] <= .5)[0]
        x1, x2 = x[pik, 0], x[pik, 1]
        f2[pik] = 1 - sqrt(abs(x1 - 2)) + 2 * (x2 - sin(6 * pi * abs(x1 - 2) + pi)) ** 2
        pik = where(x[:, 1] > .5)[0]
        x1, x2 = x[pik, 0], x[pik, 1]
        f2[pik] = 1 - sqrt(abs(x1 - 2)) + 2 * (x2 - 1 - sin(6 * pi * abs(x1 - 2) + pi)) ** 2
        out["F"] = vstack((f1, f2)).T

    def _calc_pareto_front(self, n_pareto_points=100):
        f0 = linspace(0., 1., n_pareto_points, True)
        f1 = 1 - sqrt(f0)
        return vstack((f0, f1)).T


class MMF7(Problem):
    def __init__(self):
        self.D, self.M = 2, 2
        self.xl, self.xu = (1, -1), (3, 1)
        self.zl, self.zu = (0, 0), (1, 1)
        super().__init__(n_var=self.D, n_obj=self.M, n_ieq_constr=0, xl=self.xl, xu=self.xu)

    def _evaluate(self, x, out, *args, **kwargs):
        x1, x2 = x[:, 0], x[:, 1]
        x1_2_abs = abs(x1 - 2)
        f1 = abs(x1 - 2)
        f2 = 1 - sqrt(x1_2_abs) + (
                x2 - (0.3 * power(x1_2_abs, 2) * cos(24 * pi * x1_2_abs + 4 * pi) +
                      0.6 * x1_2_abs) * sin(6 * pi * x1_2_abs + pi)) ** 2
        out["F"] = vstack((f1, f2)).T

    def _calc_pareto_front(self, n_pareto_points=100):
        f0 = linspace(0., 1., n_pareto_points, True)
        f1 = 1 - sqrt(f0)
        return vstack((f0, f1)).T


class MMF8(Problem):
    def __init__(self):
        self.D, self.M = 2, 2
        self.xl, self.xu = (-pi, 0), (pi, 9)
        self.zl, self.zu = (0, 0), (1, 1)
        super().__init__(n_var=self.D, n_obj=self.M, n_ieq_constr=0, xl=self.xl, xu=self.xu)

    def _evaluate(self, x, out, *args, **kwargs):
        x1_abs = abs(x[:, 0])
        f1 = sin(x1_abs)
        f2 = zeros(len(x), float)
        pik = where(x[:, 1] <= 4)[0]
        x1, x2 = x1_abs[pik], x[pik, 1]
        f2[pik] = sqrt(1 - sin(x1) ** 2) + 2 * (x2 - sin(x1) - x1) ** 2
        pik = where(x[:, 1] > 4)[0]
        x1, x2 = x1_abs[pik], x[pik, 1]
        f2[pik] = sqrt(1 - sin(x1) ** 2) + 2 * (x2 - 4 - sin(x1) - x1) ** 2
        out["F"] = vstack((f1, f2)).T

    def _calc_pareto_front(self, n_pareto_points=100):
        f0 = linspace(0., 1., n_pareto_points, True)
        f1 = 1 - sqrt(f0)
        return vstack((f0, f1)).T


class MMF13(Problem):
    def __init__(self, n_pf=2):
        self.np = n_pf  # n_global_and_local_PF
        self.D, self.M = 3, 2
        self.xl, self.xu = (0.1, 0.1, 0.1), (1.1, 1.1, 1.1)
        self.zl, self.zu = (0., 0.), (1., 1.)
        extremes = self._calc_pareto_front_raw(2)
        self.nadir = np.max(extremes, axis=0)
        self.ideal = np.min(extremes, axis=0)
        super().__init__(n_var=self.D, n_obj=self.M, n_ieq_constr=0, xl=self.xl, xu=self.xu)

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[:, 0]
        t = x[:, 1] + sqrt(x[:, 2])
        f2 = 2 - exp(-2 * log10(2) * ((t - .1) / .8) ** 2) * sin(self.np * pi * t) ** 6
        f2 /= x[:, 0]
        res = vstack((f1, f2)).T  # scale has a great influence on MOES/D
        out["F"] = self._normalize(res)

    def _normalize(self, res):
        return (res - self.ideal) / (self.nadir - self.ideal)

    def _calc_pareto_front_raw(self, n_pareto_points=100):
        f0 = linspace(0.1, 1.1, n_pareto_points, True)
        f1 = (2 - exp(-2 * log10(2) * ((1 / 2 / self.np - 0.1) / .8) ** 2) * sin(pi / 2)) / f0
        return vstack((f0, f1)).T

    def _calc_pareto_front(self, n_pareto_points=100):
        res = self._calc_pareto_front_raw(n_pareto_points)
        return self._normalize(res)


class MMF14(Problem):
    def __init__(self, D=3, M=3, n_p=2):
        self.n_p = n_p
        self.D, self.M = D, M
        self.xl, self.xu = zeros(D, float), ones(D, float)
        self.zl, self.zu = zeros(M, float), ones(M, float) * 2
        super().__init__(n_var=D, n_obj=M, n_ieq_constr=0, xl=self.xl, xu=self.xu)

    def _evaluate(self, x, out, *args, **kwargs):
        cos_ = cos(pi / 2 * x[:, :self.M-1])
        sin_ = sin(pi / 2 * x[:, :self.M-1])
        g1 = 1 + self._g(x)
        f = zeros((len(x), self.M), float)
        for i in range(self.M):
            if not i:
                f[:, i] = prod(cos_[:, :self.M-1], axis=1) * g1
            elif i == self.M - 1:
                f[:, i] = sin_[:, 0] * g1
            else:
                f[:, i] = prod(cos_[:, :self.M-1-i], axis=1) * sin_[:, self.M-1-i] * g1
        out["F"] = f

    def _g(self, x):
        return 2 - sin(self.n_p * pi * (x[:, -1] - .5 * sin(pi * x[:, -2]) + .5 / self.n_p)) ** 2

    def _calc_pareto_front(self, ref_dirs):
        return generic_sphere(ref_dirs) * 2.


class MMF15(Problem):
    def __init__(self, D=3, M=3, n_p=2):
        self.n_p = n_p
        self.D, self.M = D, M
        self.xl, self.xu = zeros(D, float), ones(D, float)
        self.zl, self.zu = zeros(M, float), ones(M, float) * 2.5
        super().__init__(n_var=D, n_obj=M, n_ieq_constr=0, xl=self.xl, xu=self.xu)

    def _evaluate(self, x, out, *args, **kwargs):
        cos_ = cos(pi / 2 * x[:, :self.M-1])
        sin_ = sin(pi / 2 * x[:, :self.M-1])
        g1 = 1 + self._g(x)
        f = zeros((len(x), self.M), float)
        for i in range(self.M):
            if not i:
                f[:, i] = prod(cos_[:, :self.M-1], axis=1) * g1
            elif i == self.M - 1:
                f[:, i] = sin_[:, 0] * g1
            else:
                f[:, i] = prod(cos_[:, :self.M-1-i], axis=1) * sin_[:, self.M-1-i] * g1
        out["F"] = f

    def _g(self, x):
        t = x[:, -1] - .5 * sin(pi * x[:, -2]) + .5 / self.n_p
        return 2 - exp(-2 * log10(2) * ((t - .1) / .8) ** 2) * sin(self.n_p * pi * t) ** 2

    def _calc_pareto_front(self, ref_dirs):
        return generic_sphere(ref_dirs) * 2.


def plot_sphere(N):
    r = power(linspace(0, 1, N), 1. / 3)
    theta = linspace(0, .5, N) * pi
    phi = arccos(linspace(0, 1, N))
    x = r * cos(theta) * sin(phi)
    y = r * sin(theta) * cos(phi)
    z = r * cos(phi)
    return vstack((x, y, z)).T
