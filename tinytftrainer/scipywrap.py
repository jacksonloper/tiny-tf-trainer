import dataclasses
import collections
import contextlib

import scipy.optimize
import scipy as sp
import numpy as np

import tensorflow as tf

class DummyIterator:
    def __init__(self,lst):
        self.lst=lst

    def __iter__(self):
        for l in self.lst:
            yield l

    def set_description(self,s):
        pass

    def update(self,t):
        pass

@contextlib.contextmanager
def MaybeTqdmContextManager(use_tqdm_notebook=False):
    if use_tqdm_notebook:
        import tqdm.notebook
        with tqdm.notebook.tqdm() as tq:
            yield tq
    else:
        yield DummyIterator([])

@dataclasses.dataclass
class Var:
    shape: tuple = ()
    lb: float = None
    ub: float = None
    init: float = None

    def __post_init__(self):
        if isinstance(self.shape,int):
            self.shape=(self.shape,)

        self.lb=[self.lb for i in range(self.sz)]
        self.ub=[self.ub for i in range(self.sz)]
        self.bounds=list(zip(self.lb,self.ub))

        if self.init is not None:
            self.init=np.require(self.init)
            if self.init.shape!=self.shape:
                raise ValueError("proposed initial value doesnt match shape")
            self.init_as_param=tf.convert_to_tensor(self.init)
        elif (self.lb is None) and (self.ub is None):
            self.init_as_param=tf.zeros(self.shape)
        else:
            raise ValueError("If lb or ub are provided, initial conditions must also be provided")

    def _set_init_as_param(self,init):
        init=np.require(init)


    @property
    def init_as_raw(self):
        return self.as_raw(self.init_as_param)

    def as_param(self,x):
        return tf.reshape(x,self.shape)

    def as_raw(self,x):
        return tf.reshape(x,(self.sz,))

    @property
    def sz(self):
        if self.shape==():
            return 1
        else:
            return int(np.prod(self.shape))

@dataclasses.dataclass
class OptimizationResult:
    result: dict
    scipyoptinfo: dict
    losses: np.ndarray
    attempts_raw: list
    final_loss: float


class OptimizationProblem:
    '''
    At each iteration, we evaluate

        lossfunc(*args,**attempt)

    where args is a tuple/list of fixed values, and attempt is a dictionary mapping
    variable names to quantities of interest.
    '''

    def __init__(self,lossfunc,variables,args=None,dtype=tf.float64):
        self.lossfunc=lossfunc
        self.variables=variables
        self.canonical_order=list(variables.keys())
        self.dtype=dtype

        self.init_as_param={nm:variables[nm].init_as_param for nm in variables}
        self.set_init()

        if args is None:
            self.args=()
        else:
            self.args=[
                (None if (x is None) else tf.convert_to_tensor(x))
                for x in args]

        self._recollect_bounds()

    def _recollect_bounds(self):
        self.bounds=[]
        for nm in self.canonical_order:
            self.bounds.extend(self.variables[nm].bounds)

    def set_args(self,*args):
        self.args=tuple([
            (None if (x is None) else tf.convert_to_tensor(x))
            for x in args])

    def set_init(self,**dct):
        for nm in dct:
            self.init_as_param[nm]=dct[nm]
        self.init=[np.require(self.variables[x].as_raw(self.init_as_param[x]),dtype=np.float64) for x in self.canonical_order]
        self.init=np.concatenate(self.init,axis=0)

    def as_paramdict(self,theta):
        i=0
        dct={}
        for nm in self.canonical_order:
            v=self.variables[nm]
            dct[nm]=v.as_param(theta[i:i+v.sz])
            i=i+v.sz
        return dct

    def as_raw(self,**params):
        ths=[]
        for i,nm in enumerate(self.canonical_order):
            ths.append(self.variables[nm].as_raw(params[nm]))
        return tf.concat(ths,axis=0)

    def _loss_and_grad_from_theta_compiled(self,theta,*args):
        theta=tf.convert_to_tensor(theta)
        with tf.GradientTape() as t:
            t.watch(theta)
            attempt=self.as_paramdict(theta)
            loss=self.lossfunc(*args,**attempt)
        grad=t.gradient(loss,theta)

        return loss,grad

    @tf.function
    def _loss_and_grad_from_theta(self,theta,*args):
        theta=tf.convert_to_tensor(theta)
        with tf.GradientTape() as t:
            t.watch(theta)
            attempt=self.as_paramdict(theta)
            loss=self.lossfunc(*args,**attempt)
        grad=t.gradient(loss,theta)

        return loss,grad

    def optimize(self,maxiter=100,solver='L-BFGS-B',use_tqdm_notebook=False,debug=False):
        options=dict(maxiter=maxiter)

        lgfunc = (self._loss_and_grad_from_theta if debug else self._loss_and_grad_from_theta_compiled)

        if solver!='L-BFGS-B':
            raise ValueError("only L-BFGS-B supported")

        # prep scipy function
        losses=[]
        attempts_raw=[]
        def scipyfunc(theta):
            assert not np.isnan(theta).any()
            theta=tf.convert_to_tensor(theta,dtype=self.dtype)
            loss,grad=lgfunc(theta,*self.args)
            attempts_raw.append(theta)
            loss=loss.numpy().astype(float)
            grad=grad.numpy().astype(float)
            losses.append(loss)
            assert not np.isnan(loss),'nan loss'
            assert not np.isnan(grad).any(),'nan grad'
            assert not np.isinf(loss),'inf loss'
            assert not np.isinf(grad).any(),'inf grad'

            return loss,grad

        # perform optimization
        with MaybeTqdmContextManager(use_tqdm_notebook=use_tqdm_notebook) as tq:
            def callback(*args,**kwargs):
                tq.update(1)
                tq.set_description(f"loss={losses[-1]}")
            optinfo=sp.optimize.minimize(
                scipyfunc,
                self.init,
                method=solver,
                options=options,
                bounds=self.bounds,
                callback=callback,
                jac=True
            )

        theta=tf.convert_to_tensor(optinfo['x'],dtype=tf.float64)
        result_tf=self.as_paramdict(theta)
        result={x:result_tf[x].numpy() for x in result_tf}
        final_loss=optinfo['fun']

        return OptimizationResult(
            result,optinfo,losses,attempts_raw,final_loss
        )