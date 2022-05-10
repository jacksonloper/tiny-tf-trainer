import tensorflow as tf
import tqdm.notebook
import time

def _linterp(st,en,alpha):
    return st*(1-alpha)+en*alpha

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
    def __init__(self,trainable_module,lr=1e-3,
                scheduler=None,opt=None):

        self.trainable_module=trainable_module
        self.logs=[]
        self.test_logs=[]

        if opt is None:
            self.opt=tf.optimizers.Adam(learning_rate=lr)
        else:
            self.opt=opt

        self.scheduler=scheduler
        self.lr=lr
        self.i=0

    def get_lossinfo_summary(self,lossinfo):
        return dict(
            epoch=self.i,
            timestamp=time.time(),
            loss=lossinfo['loss'].numpy()
        )
    def train_callback(self,lossinfo):
        self.logs.append(self.get_lossinfo_summary(lossinfo))
    def test_callback(self,lossinfo):
        self.test_logs.append(self.get_lossinfo_summary(lossinfo))

    def get_tq_description(self):
        s=''
        if len(self.logs)>0:
            v=self.logs[-1]['loss']
            s=s+f'trainloss={v:.3e}'
        if len(self.test_logs)>0:
            v=self.test_logs[-1]['loss']
            s=s+f', testloss={v:.3e}'
        return s

    def onestep(self,opt,dataloader,debug=False,callback=None):
        for v in dataloader:
            if debug:
                lossinfo=self.trainable_module.train_step_uncompiled(
                    opt,*v)
            else:
                lossinfo=self.trainable_module.train_step(
                    opt,*v)
            if callback is not None:
                callback(lossinfo)

    def train(self,dataloader,nepochs,debug=False,testing_dataloader=None):
        tq=tqdm.notebook.trange(self.i,self.i+nepochs)
        for i in tq:
            # set lr
            if self.scheduler is not None:
                self.opt.learning_rate.assign(
                    self.scheduler(self.i,nepochs,self.lr))

            # train
            self.onestep(
                self.opt,dataloader,debug=debug,
                callback=self.train_callback)

            # test
            if testing_dataloader is not None:
                self.onestep(
                    None,testing_dataloader,debug=debug,
                    callback=self.test_callback)

            # done!
            self.i=self.i+1
            tq.set_description(self.get_tq_description())

class TrainableModule(tf.Module):
    '''
    Should implement lossinfo function,
    which returns a dictionary which
    includes a 'loss' key which points
    to a scalar.
    '''
    def train_grads(self,*args):
        with tf.GradientTape() as t:
            lossinfo=self.lossinfo(*args)
            loss=lossinfo['loss']
        gradz=t.gradient(loss,self.variables)
        return lossinfo,gradz

    def train_step_uncompiled(self,opt,*args):
        lossinfo,gradz=self.train_grads(*args)
        if opt is not None:
            opt.apply_gradients(zip(gradz,self.variables))
        return lossinfo

    @tf.function
    def train_step(self,opt,*args):
        return self.train_step_uncompiled(opt,*args)