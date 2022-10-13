import tensorflow as tf
import tqdm
import time
import numpy as np
import threading
rng=np.random.default_rng()

import contextlib

from tinytftrainer import layers

class JointDataset:
    def __init__(self,datasets):
        self.datasets=datasets
    @contextlib.contextmanager
    def load(self):
        with contextlib.ExitStack() as stack:
            loaded_datasets = [stack.enter_context(x.load()) for x in self.datasets]

            def reap():
                return tuple([x() for x in loaded_datasets])

            yield reap

class DatasetNull:
    @contextlib.contextmanager
    def load(self):
        yield None


class DatasetSingleton:
    def __init__(self,args):
        self.args=args

    @contextlib.contextmanager
    def load(self):
        def reap():
            return self.args
        yield reap

class LoadedPrefetchDataset:
    def __init__(self,args,permitted_idxs,prefetch,batchsize):
        self.args=args
        self.permitted_idxs=permitted_idxs
        self.prefetch=prefetch
        self.batchsize=batchsize

        self.idxs=tf.Variable(tf.random.shuffle(self.permitted_idxs))
        self.i=tf.Variable(tf.zeros((),dtype=tf.int32))

        self.n=self.idxs.shape[0]

        self.q=tf.queue.FIFOQueue(
            self.prefetch,
            [x.dtype for x in self.args]+[tf.int32],
            shapes=[((self.batchsize,)+x.shape[1:]) for x in self.args] + [()]
        )
        self._th=threading.Thread(target=self._sow_loop)
        self._th.start()

    def close(self):
        self.q.close(cancel_pending_enqueues=True)
        self._th.join()

    def __call__(self):
        rez=self.q.dequeue()
        return tuple(rez[:-1])

    @tf.function
    def _reshuffle_and_sow(self):
        self.i.assign(0)
        self.idxs.assign(tf.random.shuffle(self.permitted_idxs))
        return self._sow()

    @tf.function(autograph=False)
    def _sow(self):
        subidxs=self.idxs[self.i:self.i+self.batchsize]
        self.i.assign(self.i+self.batchsize)
        return [tf.gather(x,subidxs) for x in self.args]+[0]

    def _sow_loop(self):
        while True:
            try:
                if self.i.numpy()+self.batchsize>self.n:
                    new_data=self._reshuffle_and_sow()
                else:
                    new_data=self._sow()
                self.q.enqueue(new_data)
            except tf.errors.CancelledError:
                return
            except:
                self.q.close(cancel_pending_enqueues=True)
                raise

class Dataset:
    def __init__(self,batchsize,args,permitted_idxs=None,prefetch=5):
        self.args=args
        self.batchsize=int(batchsize)
        if self.batchsize<1:
            raise ValueError("batchsize must be at least 1")

        self.n=self.args[0].shape[0]

        if permitted_idxs is None:
            self.permitted_idxs = tf.range(self.n)
        else:
            self.permitted_idxs = tf.convert_to_tensor(
                permitted_idxs,dtype=tf.int32)

        if len(self.permitted_idxs)<batchsize:
            raise ValueError("batchsize must be less than or equal to permitted idx length")

        self.prefetch=int(prefetch)
        if prefetch<1:
            raise ValueError("prefetch must be at least 1")

    @contextlib.contextmanager
    def load(self):
        lds=LoadedPrefetchDataset(
            self.args,
            self.permitted_idxs,
            self.prefetch,
            self.batchsize,
        )
        yield lds
        lds.close()

class LoadedRaggedDataset:
    def __init__(self,args):
        self.args=args

        self.i=tf.Variable(0,dtype=tf.int32)

        self.n=self.args[0].shape[0]
        self.nargs=len(self.args)

        self.permitted_idxs=tf.range(self.n)
        self.idxs=tf.Variable(tf.random.shuffle(self.permitted_idxs))

    def _steadyon(self):
        return self.i+1,self.idxs

    def _reset(self):
        return tf.zeros((),dtype=tf.int32),tf.random.shuffle(self.permitted_idxs)

    def __call__(self):
        result = [x[self.idxs[self.i]] for x in self.args]

        newi,newp=tf.cond(self.i==self.n-1,self._reset,self._steadyon)
        self.i.assign(newi)
        self.idxs.assign(newp)

        return result

class RaggedBatchedDataset:
    def __init__(self,args):
        self.args=args

    @contextlib.contextmanager
    def load(self):
        lds=LoadedRaggedDataset(
            self.args,
        )
        yield lds


def _linterp(st,en,alpha):
    return st*(1-alpha)+en*alpha

def snapshot(model):
    return [x.numpy() for x in model.variables]

def onecycle_scheduler(i,nsteps,base_learning_rate):
    alpha=i/nsteps

    if alpha<.4:
        local_alpha=alpha/.4
        return _linterp(base_learning_rate,10*base_learning_rate,local_alpha)
    elif alpha<.8:
        local_alpha=(alpha-.4)/.4
        return _linterp(10*base_learning_rate,base_learning_rate,local_alpha)
    else:
        local_alpha=(alpha-.8)/.2
        return _linterp(base_learning_rate,0.0,local_alpha)

class Trainer:
    def __init__(self,trainable_module,lr=None,
                scheduler=None,opt=None,test_every=25):

        self.trainable_module=trainable_module
        self.logs=[]
        self.test_logs=[]
        self.test_every=test_every

        if opt is None:
            lr = 1e-3 if (lr is None) else lr
            self.opt=tf.optimizers.Adam(learning_rate=lr)
        else:
            if lr is not None:
                raise ValueError("supplied both optimizer and learning rate")
            self.opt=opt

        self.scheduler=scheduler
        self.lr=lr
        self.i=0


    def get_lossinfo_summary(self,lossinfo):
        dct=dict(
            epoch=self.i,
            timestamp=time.time(),
            loss=lossinfo['loss'].numpy(),
            lr=self.opt.learning_rate.numpy(),
            info={}
        )
        if 'loginfo' in lossinfo:
            for x in lossinfo['loginfo']:
                dct['info'][x]=lossinfo['loginfo'][x].numpy()

        return dct

    def snapshot(self):
        return dict(
            model=[x.numpy() for x in self.trainable_module.trainable_variables],
            opt=[x.numpy() for x in self.opt.variables()]
        )

    def load_snapshot(self,snap):
        for (x,y) in zip(self.trainable_module.trainable_variables,snap['model']):
            x.assign(tf.convert_to_tensor(y,dtype=x.dtype))
        for (x,y) in zip(self.opt.variables(),snap['opt']):
            x.assign(tf.convert_to_tensor(y,dtype=x.dtype))

    def get_tq_description(self,additional_statii=None):
        s=''
        if len(self.logs)>0:
            v=self.logs[-1]['loss']
            s=s+f'train loss={v:.3e}'
            if additional_statii is not None:
                for nm in additional_statii:
                    v=self.logs[-1]['info'][nm]
                    s+=f', train {nm}={v:.3e}'
        if len(self.test_logs)>0:
            v=self.test_logs[-1]['loss']
            s=s+f', test loss={v:.3e}'
            if additional_statii is not None:
                for nm in additional_statii:
                    v=self.test_logs[-1]['info'][nm]
                    s+=f', test {nm}={v:.3e}'
        return s

    def train_grads(self,*args):
        with tf.GradientTape() as t:
            lossinfo=self.trainable_module.lossinfo(*args)
            loss=lossinfo['loss']
        gradz=t.gradient(
            loss,
            self.trainable_module.trainable_variables,
            unconnected_gradients='zero'
        )
        return lossinfo,gradz

    def _check_finite(self,lossinfo,gradz):
        tf.debugging.assert_all_finite(lossinfo['loss'],'loss bad')
        for g,v in zip(gradz,self.trainable_module.trainable_variables):
            tf.debugging.assert_all_finite(g,v.name)

    def train_step_debug(self,*args):
        lossinfo,gradz=self.train_grads(*args)
        self._check_finite(lossinfo,gradz)
        self.opt.apply_gradients(
            zip(gradz,self.trainable_module.trainable_variables))
        return lossinfo

    def train_step(self,*args):
        lossinfo,gradz=self.train_grads(*args)
        self.opt.apply_gradients(
            zip(gradz,self.trainable_module.trainable_variables))

        return lossinfo

    def test_step_debug(self,*args):
        # if not hasattr(self,"_everything"):
        #     self._everything=[]
        # self._everything.append(self.snapshot()['model'])

        lossinfo,gradz=self.train_grads(*args)
        self._check_finite(lossinfo,gradz)
        return lossinfo

    def test_step(self,*args):
        lossinfo,gradz=self.train_grads(*args)
        return lossinfo

    def train(self,dataset,nepochs,debug=False,
              testing_dataset=None,model_averaging=False,
              additional_statii=None,patience=None):

        curloss=np.inf
        boredom=0

        if testing_dataset is None:
            testing_dataset=DatasetNull()

        with dataset.load() as get_train, testing_dataset.load() as get_test:
            tq=tqdm.trange(self.i,self.i+nepochs,ncols=100)

            if debug:
                def train_step():
                    for i in tf.range(self.test_every-1):
                        self.train_step_debug(*get_train())
                    return self.train_step_debug(*get_train())

                def test_step():
                    v=get_test()
                    lossinfo=self.test_step_debug(*v)
                    return lossinfo
            else:
                @tf.function
                def train_step():
                    for i in tf.range(self.test_every-1):
                        self.train_step(*get_train())
                    return self.train_step(*get_train())

                @tf.function
                def test_step():
                    v=get_test()
                    lossinfo=self.test_step(*v)
                    return lossinfo

            for i in tq:
                # set lr
                if self.scheduler is not None:
                    self.opt.learning_rate.assign(
                        self.scheduler(self.i,nepochs,self.lr))

                # train for test_every steps
                lossinfo=train_step()
                self.logs.append(self.get_lossinfo_summary(lossinfo))

                # test
                if get_test is not None:
                    lossinfo=test_step()
                    self.test_logs.append(self.get_lossinfo_summary(lossinfo))

                    if self.test_logs[-1]['loss']<curloss:
                        curloss=self.test_logs[-1]['loss']
                        self.snap=self.snapshot()
                        boredom=0
                    else:
                        boredom+=1

                # done!
                self.i=self.i+1
                tq.set_description(self.get_tq_description(
                    additional_statii=additional_statii))

                if (patience is not None) and (boredom>patience):
                    return
