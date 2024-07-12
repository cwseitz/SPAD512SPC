from skimage.io import imread, imsave
from hmmlearn import hmm
import numpy as np
import matplotlib.pyplot as plt

class PhotonCountHMM:
    def __init__(self,stack):
        self.stack=stack
        
    def fit_hmm(self,data,min_comp=1,max_comp=5):
        scores = list()
        bics = list()
        models = list()
        for n_components in range(min_comp,max_comp):
            for idx in range(10):
                model = hmm.PoissonHMM(n_components=n_components, random_state=idx,
                                       n_iter=10)
                model.fit(data)
                models.append(model)
                loglike = model.score(data)
                num_params = sum(model._get_n_fit_scalars_per_param().values())
                bic = -2*loglike + num_params * np.log(len(data))
                scores.append(loglike)
                bics.append(bic)
               
                print(f'Converged: {model.monitor_.converged}\t\t'
                      f'NLL: {scores[-1]},BIC: {bics[-1]}')

        model = models[np.argmin(bics)]
        print(f'The best model had a BIC of {min(bics)} and '
              f'{model.n_components} components')
        
        return model
                
    def plot_hmm(self,data,states,model):
        time = np.arange(0,len(data),1)*0.01
        fig, ax = plt.subplots(1,3,figsize=(10,3))
        ax[0].plot(time,model.lambdas_[states], ".-", color='cyan')
        ax[0].plot(time,data,color='gray',alpha=0.5)
        ax[0].set_xlabel('Time')
        ax[0].set_ylabel('cts')
        unique, counts = np.unique(states, return_counts=True)
        counts = counts/len(states)
        rates = model.lambdas_.flatten()
        ax[1].bar(unique,counts,color='blue')
        ax[1].set_xlabel('State')
        ax[1].set_ylabel('Proportion')
        ax[2].bar(unique,rates,color='red')
        ax[2].set_xlabel('State')
        ax[2].set_ylabel('Rate (cts/frame)')
        plt.tight_layout()
        plt.show()
                
    def fit(self,show_hmm=True,min_comp=1,max_comp=10):
        stack = self.stack
        nt,nx,ny = stack.shape
        data = np.sum(stack,axis=(1,2))
        data = data.reshape((-1,1))
        model = self.fit_hmm(data,min_comp=min_comp,max_comp=max_comp)
        states = model.predict(data)
        if show_hmm:
            self.plot_hmm(data,states,model)

path = '/research2/shared/cwseitz/Data/SPAD/240702/data/intensity_images/'
file8bit = '240702_SPAD-QD-500kHz-1k-10ms-8bit-1-snip.tif'

stack = imread(path+file8bit)
model = PhotonCountHMM(stack)
model.fit(min_comp=1,max_comp=6)
print(stack.shape)

