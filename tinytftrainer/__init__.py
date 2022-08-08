import tensorflow as tf
import tqdm.notebook
import time
import numpy as np
rng=np.random.default_rng()

def _linterp(st,en,alpha):
    return st*(1-alpha)+en*alpha

def snapshot(model):
    return [x.numpy() for x in model.variables]

def load_snapshot(model,snap):
    for x,v in zip(snap,model.variables):
        v.assign(x)

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

class SubEpochDataset:
    '''
    For use when you don't want to
    go through the whole dataset each epoch
    (e.g. if you want to evaluate test loss
    more often)
    '''
    def __init__(self,generator,n=1):
        self.i=0
        self.generator=generator
        self.current_iterator=iter(self.generator)
        self.n=n

    def __iter__(self):
        for i in range(self.n):
            yield self.endless_supply()

    def endless_supply(self):
        try:
            x=next(self.current_iterator)
        except StopIteration:
            self.current_iterator=iter(self.generator)
            x=next(self.current_iterator)
        self.i=self.i+1
        return x

class Trainer:
    def __init__(self,trainable_module,lr=None,
                scheduler=None,opt=None):

        self.trainable_module=trainable_module
        self.logs=[]
        self.test_logs=[]

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

        self.traintime=0

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

    def get_tq_description(self):
        s=''
        if len(self.logs)>0:
            v=self.logs[-1]['loss']
            s=s+f'trainloss={v:.3e}'
        if len(self.test_logs)>0:
            v=self.test_logs[-1]['loss']
            s=s+f', testloss={v:.3e}'
        return s

    def train_grads(self,*args):
        with tf.GradientTape() as t:
            lossinfo=self.trainable_module.lossinfo(*args)
            loss=lossinfo['loss']
        gradz=t.gradient(loss,self.trainable_module.trainable_variables)
        return lossinfo,gradz

    def train_step_uncompiled(self,*args):
        lossinfo,gradz=self.train_grads(*args)
        tf.debugging.assert_all_finite(lossinfo['loss'],'loss bad')
        for g,v in zip(gradz,self.trainable_module.trainable_variables):
            tf.debugging.assert_all_finite(g,v.name)
        self.opt.apply_gradients(
            zip(gradz,self.trainable_module.trainable_variables))
        return lossinfo

    @tf.function
    def train_step(self,*args):
        lossinfo,gradz=self.train_grads(*args)
        self.opt.apply_gradients(
            zip(gradz,self.trainable_module.trainable_variables))
        return lossinfo

    def test_step_uncompiled(self,*args):
        lossinfo,gradz=self.train_grads(*args)
        return lossinfo

    @tf.function
    def test_step(self,*args):
        lossinfo,gradz=self.train_grads(*args)
        return lossinfo

    def train(self,dataset,nepochs,debug=False,testing_dataset=None):
        startt=time.time()
        try:
            tq=tqdm.notebook.trange(self.i,self.i+nepochs)

            train_step=(self.train_step_uncompiled if debug else self.train_step)
            test_step=(self.test_step_uncompiled if debug else self.test_step)

            for i in tq:
                # set lr
                if self.scheduler is not None:
                    self.opt.learning_rate.assign(
                        self.scheduler(self.i,nepochs,self.lr))

                # train one epoch
                for v in dataset:
                    lossinfo=train_step(*v)
                    self.logs.append(self.get_lossinfo_summary(lossinfo))

                # test one epoch
                if testing_dataset is not None:
                    for v in testing_dataset:
                        lossinfo=test_step(*v)
                        self.test_logs.append(self.get_lossinfo_summary(lossinfo))

                # done!
                self.i=self.i+1
                tq.set_description(self.get_tq_description())
        finally:
            self.traintime+=time.time()-startt