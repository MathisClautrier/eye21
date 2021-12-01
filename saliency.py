from scipy.ndimage.filters import gaussian_filter
from torchvision.transforms import GaussianBlur
import numpy as np
import torch

class Saliency:
    """Object used to perform each step required to compute the saliency map and to process them"""
    def __init__(self, sig_mask,kernel_size,sigma_kernel,dim_actions,maximize = True):
        self.mask = torch.zeros((84,84,4,84,84))
        self.sigma = sig_mask #masks' variance
        self.create_mask()
        self.blurred_state = torch.zeros((1,1,4,84,84))
        self.GaussianB = GaussianBlur((kernel_size,kernel_size),sigma_kernel) #Gaussian blur to compute A
        self.dim_actions = dim_actions
        if maximize == True: #discuss if we consider the argmax (to discuss)
            self.maximization = torch.nn.Softmax(1)
        else:
            self.maximization = torch.nn.Identity()
    
    def get_mask(self,i,j):
        """inspired by the official implementation of the paper 'Visualizing and Understanding Atari Agents' \
            https://arxiv.org/pdf/1711.00138.pdf, compute the max centered in (i,j)"""
        y,x = np.ogrid[-i:84-i, -j:84-j]
        keep = x*x + y*y <= 1
        maskij = np.zeros((84,84)) 
        maskij[keep] = 1 # select a circle of pixels
        maskij = gaussian_filter(maskij, sigma=self.sigma) # blur the circle of pixels.
        return torch.Tensor(maskij/maskij.max()) #normalize
    
    def create_mask(self):
        for n in range(4):
            for i in range(84):
                for j in range(84):
                    self.mask[i,j,n,:,:] = self.get_mask(i,j) #compute all the masks
                
    def blur(self,state):
        #Faster to double unsqueeze than assign blurred_state[0,0] (20% faster)
        self.blurred_state = (self.GaussianB(state) - state).unsqueeze(0).unsqueeze(0) #change the dimension for tensor ops
        
    def obtain_gradient(self,agent,state):
        state = state.repeat(self.dim_actions,1,1,1) #We want dim_actions gradients
        state.requires_grad = True
        output = self.maximization((agent.online_net(state)*agent.support).sum(2))
        output = torch.diagonal(output) #we are only interested by the diagonal terms 
        gradient = torch.autograd.grad(outputs=output, inputs=state, grad_outputs=torch.ones_like(output),
                                           retain_graph=True)
        return gradient
    
    def compute_saliency(self,agent,state):
        self.blur(state)
        D = self.blurred_state * self.mask #(84,84,4,84,84)
        state = state.unsqueeze(0)
        gradient = self.obtain_gradient(agent,state)[0].unsqueeze(0) #(1,dim_actions,4,84,84)
        saliency = torch.tensordot(D,gradient,dims = ([2,3,4],[2,3,4])) #(84,84,1,nb_actions)
        saliency = torch.square(saliency).sum(-1).squeeze(-1)
        return saliency.detach()