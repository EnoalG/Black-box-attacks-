import torch
import numpy as np
from scipy.fftpack import dct, idct
from typing import Any


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


"""
L2 distance between two tensors

Parameters
----------
a : torch.Tensor
    first tensor
b : torch.Tensor
    second tensor

Returns
-------
torch.Tensor
    L2 distance between a and b

"""
def distance(a, b):
    return (a - b).flatten(1).norm(dim=1)


"""
Class to perform multpile SurFree-based attacks

SurFree attack
SurFree attack with normal vector
SurFree attack with estimated normal vector
Direct attack with normal vector
"""

class Attacks():
    """
    Parameters
    ----------
    steps : int
        number of steps for the attack
    N : int
        number of samples to estimate the normal vector if the normal vector is estimated
    std_dev : float
        standard deviation for the noise added to estimate the normal vector if the normal vector is estimated
    delta : float
        delta for the HSJA method if the normal vector is estimated
    method : str
        method to estimate the normal vector (GeoDA or HSJ)
    max_queries : int
        maximum number of queries allowed
    dct_size : int
        size of the dct block if the dct is used
    frequency_to_remove : int
        number of high frequencies to remove if the dct is used
    starting_point : torch.Tensor
        starting point for the attack (must be a point of the boundary)
    with_normal_vector : bool
        if True, the attack performed is: SurFree attack with normal vector
    without_normal_vector : bool
        if True, the attack performed is: SurFree attack
    with_estimated_normal_vector : bool
        if True, the attack performed is: SurFree attack with estimated normal vector
    without_surfree : bool
        if True, the attack performed is: Direct attack with normal vector
    with_dct : bool
        if True, the attack is performed with the dct
    """
    def __init__(self,
                 steps = 50, 
                 N = 50, 
                 std_dev = 0.0005, 
                 delta = 1e-3, 
                 method = 'surfree', 
                 max_queries = 5000,
                 dct_size = 8,
                 frequency_to_remove = 1,
                 starting_point = None, 
                 with_dct = False,
                 increasing = False,
                 decreasing = False,
                 keep_directions = False,
                 theta_eps = 1e-2
                 ) -> None:

        self._steps =steps
        self._N = N
        self._std_dev = std_dev
        self._delta = delta
        self._method = method
        self._starting_point = starting_point
        self._max_queries = max_queries
        self._with_dct = with_dct
        self._dct_size = dct_size
        self._frequency_to_remove = frequency_to_remove
        self._increasing = increasing
        self._decreasing = decreasing
        self._keep_directions = keep_directions
        self._theta_eps = theta_eps
        

    """
    Run the attack

    Parameters
    ----------
    model : torch.nn.Module
        model to attack
    X : torch.Tensor
        input to attack

    Returns
    -------
    torch.Tensor
        optimal adversarial example found
    """

    def __call__(self,model, X) -> Any:
        
        self._model = model
        self._X = X
        self._nquery = torch.zeros(len(self._X))
        self._badquery = torch.zeros(len(self._X))
        self._labels = self._model(self._X).argmax(1)
        self._images_finished = self._model(self._X).argmax(1) != self._labels
        self._orthogonal_directions = []
        self._last_normal_vector = torch.zeros_like(self._X)

        
        if self._method == 'cgba':
            if self._with_dct:
                best_adv = self._cgba_with_dct()
            else:
                best_adv = self._cgba()
            return best_adv
        
        elif self._method == 'geoda':
            if self._with_dct:
                best_adv = self._geoda_with_dct()
            else:
                best_adv = self._geoda()
            return best_adv
        else:
            if not self._with_dct:
                best_adv = self._surfree()
            else:
                best_adv = self._surfree_with_dct()
            return best_adv
   
    """
    Check if the input is an adversarial example

    Parameters
    ----------
    Y : torch.Tensor
        input to check

    Returns
    -------
    torch.Tensor
        boolean tensor indicating if the input is adversarial
    """
    def _is_adversarial(self,Y):
        p = self._model(Y).argmax(1)
        return (p != self._labels)
    

    """
    Give an adversarial example for the input 

    Parameters
    ----------
    X : torch.Tensor
        input to attack
    ind : int
        index of the input
    max_iter : int
        maximum number of iterations to find the adversarial example
        
    Returns
    -------
    torch.Tensor
        adversarial example found

    Warning
    -------
    The input must be a tensor of shape (C, H, W)
    """
    def _get_adversarial(self,X, ind, max_iter = 10000):
        adv = torch.zeros_like(X)
        adv2 = torch.zeros_like(X)
        adv2 = adv2 + X
        labels = self._model(X).argmax(1)
        self._nquery[ind] += 1
        p = self._model(adv2).argmax(1)
        if p == labels:
            self._badquery[ind] += 1
        n = 0
        while torch.any(p == labels) and n < max_iter:
            n+=1
            noise = 2 * torch.rand_like(X) -1
            adv2 = (adv + noise)
            p = self._model(adv2).argmax(1)
            if p == labels:
                self._badquery[ind] += 1
            self._nquery[ind] += 1

        if (n == max_iter):
            print('Point adverse introuvé')

        return adv2
    
    """
    Give a boundary point for the input

    Parameters
    ----------
    X : torch.Tensor
        input to attack
    eps : float
        precision for the boundary point
    max_iter : int
        maximum number of iterations to find the boundary point

    Returns
    -------
    torch.Tensor
        boundary point found

    Warning
    -------
    The input must be a tensor of shape (n, C, H, W) where n is the number of inputs to attack
    """
    def _find_boundary(self,X, eps = 0.01, max_iter = 100000):
        labels = self._model(X).argmax(1)
        best_adv = []
        d = np.prod(X[0].shape)
        eps = eps/(d*np.sqrt(d))
        for i in range(len(X)):
            starting_point = self._get_adversarial(X[i].unsqueeze(0), i)
            upper_bound = starting_point
            lower_bound = X[i].unsqueeze(0)
            n = 0
            while (torch.norm(lower_bound-upper_bound) > eps and n < max_iter):
                mid_point = torch.zeros_like(X[i].unsqueeze(0))
                mid_point = (upper_bound + lower_bound)/2
                self._nquery[i] += 1
                if self._model(X[i].unsqueeze(0)).argmax(1) != self._model(mid_point).argmax(1):
                    upper_bound = mid_point
                else :
                    lower_bound = mid_point
                    self._badquery[i] += 1

            if self._model(X[i].unsqueeze(0)).argmax(1) == self._model(mid_point).argmax(1):
                mid_point = upper_bound

            best_adv.append(mid_point[0])
        best_adv = torch.stack(best_adv)

        return best_adv

    """
    Find the exact normal vector for the input

    Parameters
    ----------
    n_max : int
        maximum number of iterations to find the normal vector
    eps : float
        precision for the normal vector

    Returns
    -------
    torch.Tensor
        normal vector found
    """
    def _find_normal_vector(self, n_max = 100, eps = 1e-4):
        dim = np.prod(self._X[0].shape)
        normal_vector = []
        for i in range(len(self._X)):
            data_to_attack = torch.zeros_like(self._X[i].unsqueeze(0))
            data_to_attack = data_to_attack + self._X[i].unsqueeze(0)
            points_on_boundary =  [self._find_boundary(data_to_attack)[0] for j in range(dim)]
            points_on_boundary = torch.stack(points_on_boundary)

            points_on_boundary = points_on_boundary.view(dim,dim)

            _, _, V = np.linalg.svd(points_on_boundary)

            solution = V[-1]

            tenseur_vecteur = torch.tensor(solution/np.linalg.norm(solution))
            normal_vector.append(tenseur_vecteur.reshape(data_to_attack[0].shape))
        normal_vector = torch.stack(normal_vector)
    
        return normal_vector
    
    """
    Find the adversarial example in the direction given

    Parameters
    ----------
    X : torch.Tensor
        input to attack
    labels : torch.Tensor
        labels of the input
    direction : torch.Tensor
        direction to follow
    max_iter : int
        maximum number of iterations to find the adversarial example

    Returns
    -------
    torch.Tensor
        adversarial example found

    Warning
    -------
    The input must be a tensor of shape (C, H, W)
    """
    def _get_adversarial_in_direction(self, X, labels,ind,direction, max_iter = 10000):
        adv = torch.zeros_like(X)
        adv = adv + X
        epsilon = 0 
        n = 0
        p = self._model(adv).argmax(1)
        epsilon = 0.3 
        while torch.any(p == labels) and n < max_iter:
            n+=1
            adv = X + epsilon *direction
            epsilon = -1.1*epsilon
            if p == labels:
                self._badquery[ind] += 1
            self._nquery[ind] += 1
            p = self._model(adv).argmax(1)


        if (n == max_iter):
            print('Point adverse introuvé')

        return adv
    
    """
    Find the boundary point in the direction given

    Parameters
    ----------
    direction : torch.Tensor
        direction to follow
    eps : float
        precision for the boundary point
    max_iter : int
        maximum number of iterations to find the boundary point

    Returns
    -------
    torch.Tensor
        boundary point found
    """
    def _find_boundary_in_direction(self,direction, eps = 1e-2, max_iter = 1000):
        best_adv = []
        for i in range(len(self._X)):
            data_to_attack = torch.zeros_like(self._X[i].unsqueeze(0))
            data_to_attack = data_to_attack + self._X[i].unsqueeze(0)
            labels = self._model(data_to_attack).argmax(1)
            adv = self._get_adversarial_in_direction(data_to_attack,labels,i, direction[i].unsqueeze(0))
        
            upper_bound = torch.zeros_like(adv)
            upper_bound = upper_bound + adv
            lower_bound = torch.zeros_like(data_to_attack)
            lower_bound = lower_bound + data_to_attack
            n = 0
            while (torch.norm(lower_bound-upper_bound) > eps and n < max_iter):
                n+=1
                mid_point = torch.zeros_like(data_to_attack)
                mid_point = (upper_bound + lower_bound)/2
                self._nquery[i] += 1
                if self._model(data_to_attack).argmax(1) == self._model(mid_point).argmax(1):
                    lower_bound = mid_point
                    self._badquery[i] += 1
                else :
                    upper_bound = mid_point

            if self._model(data_to_attack).argmax(1) == self._model(mid_point).argmax(1):
                mid_point = upper_bound

            best_adv.append(mid_point[0])
        best_adv = torch.stack(best_adv)
        return best_adv
    
    """
    Point on the circle

    Parameters
    ----------
    X : torch.Tensor
        center of the circle
    u : torch.Tensor
        first vector of the plane
    v : torch.Tensor
        second vector of the plane orthogonal to u
    r : torch.Tensor
        radius of the circle
    theta : float
        angle to find the point on the circle

    Returns
    -------
    torch.Tensor
        point on the circle
    """
    def _point_on_circle(self,X, u,v,r, theta):
        res =  X + r*u + r*(np.cos(theta)*u + np.sin(theta)*v)
        return res

    """
    Binary search along the circle

    Parameters
    ----------
    M : torch.Tensor
        center of the circle
    u : torch.Tensor
        first vector of the plane
    v : torch.Tensor
        second vector of the plane orthogonal to u
    r : torch.Tensor
        radius of the circle
    theta_eps : float
        precision for the angle
    eps : float
        precision for the point on the circle
    max_iter : int
        maximum number of iterations to find the point on the circle

    Returns
    -------
    torch.Tensor
        pointat the intersection of the circle and the boundary
    """
    def _binary_search_along_circle(self,b_point, M, u, v,r,theta_eps = 0.1,  eps = 1e-2, max_iter = 10000):
        best_adv = []
        for i in range(len(self._X)):
            data_to_attack = torch.zeros_like(self._X[i].unsqueeze(0))
            data_to_attack = data_to_attack + self._X[i].unsqueeze(0)
            m = M[i].unsqueeze(0)
            _u = u[i].unsqueeze(0)
            _v = v[i].unsqueeze(0)
            lab = self._labels[i]
            n = 0
            do = True
            if self._model(self._point_on_circle(data_to_attack,_u, _v, r[i], theta_eps)).argmax(1) != lab:
                upper_theta = np.pi
                lower_theta = 0
                self._nquery[i] += 1

            elif self._model(self._point_on_circle(data_to_attack,_u, _v, r[i], -theta_eps)).argmax(1) != lab:
                upper_theta = -np.pi
                lower_theta = 0
                self._nquery[i] += 2
                self._badquery[i] += 1
            else:
                point = b_point[i].unsqueeze(0)
                lower_theta = 0
                upper_theta = 0
                do = False
                self._nquery[i] += 2
                self._badquery[i] += 2
            last_adv = upper_theta
            
            while (do and np.abs(lower_theta-upper_theta) > eps and n < max_iter):
                theta_mid = (lower_theta + upper_theta)/2
                point = self._point_on_circle(data_to_attack,_u, _v,r[i], theta_mid)
                self._nquery[i] += 1
                if self._model(point).argmax(1) != lab:
                    last_adv = theta_mid
                    lower_theta = theta_mid
                else:
                    last_adv = theta_mid
                    upper_theta = theta_mid
                    self._badquery[i] += 1
                n+=1
            if do and self._model(point).argmax(1) == lab:
                point = self._point_on_circle(data_to_attack,_u, _v, r[i], last_adv)
            best_adv.append(point[0])
        best_adv = torch.stack(best_adv)
        
        return best_adv
    
    """
    Random vector

    Parameters
    ----------
    u : torch.Tensor
        first vector of the plane
    
    Returns
    -------
    torch.Tensor
        random vector orthogonal to u
    """
    def _random_vector(self,u):
        random_v = torch.rand(u.shape)*2 - 1
        vec_v = []
        for i in range(len(random_v)):
            if self._keep_directions:
                random_v = self._gram_schmidt_list(self._orthogonal_directions,random_v[i])
            else:
                random_v[i] = self._gram_schmidt(u[i],random_v[i])
            norms = torch.norm(random_v[i].view(len(random_v[i]),-1),dim=1)
            random_v[i] = random_v[i]/norms
            vec_v.append(random_v[i])

        vec_v = torch.stack(vec_v)
        return vec_v
    
    """
    Gram-Schmidt orthogonalization for a list of vectors

    Parameters
    ----------
    u : torch.Tensor
        first vector of the plane
    n : torch.Tensor
        second vector to orthogonalize

    Returns
    -------
    torch.Tensor
        orthogonalized vector of n with respect to u
    """
    def _gram_schmidt_list(self, u, n):
        v = torch.clone(n)
        for i in range(len(self._orthogonal_directions)):
            v_dim = v.reshape(-1).type(torch.float64)
            u_dim = self._orthogonal_directions[i].reshape(-1).type(torch.float64)
            v = v - (torch.dot(u_dim,v_dim)/torch.dot(u_dim,u_dim))*self._orthogonal_directions[i]
        v = v/v.norm()
        return v


    """
    Gram-Schmidt orthogonalization

    Parameters
    ----------
    u : torch.Tensor
        first vector of the plane
    n : torch.Tensor
        second vector to orthogonalize

    Returns
    -------
    torch.Tensor
        orthogonalized vector of n with respect to u
    """
    def _gram_schmidt(self,u,n):
        n_dim = n.reshape(np.prod(n.shape))
        u_dim = u.reshape(np.prod(u.shape))
        v = n - (torch.dot(u_dim,n_dim)/torch.dot(u_dim,u_dim))*u
        v = v / v.norm()
        return v
    
    """
    GeoDA method to estimate the normal vector

    Parameters
    ----------
    X : torch.Tensor
        input to attack
    u : torch.Tensor
        first vector of the plane
    N : int
        number of samples to estimate the normal vector
    std_dev : float
        standard deviation for the noise added to estimate the normal vector

    Returns
    -------
    torch.Tensor
        estimated normal vector with GeoDA estimator

    Warning
    -------
    The input must be a tensor of shape (n, C, H, W) where n is the number of inputs to attack
    """    
    def _estimated_normal_vector(self,X, u, N, std_dev):
        res = []
        for k in range(len(X)):
            x = X[k].unsqueeze(0)
            s = torch.zeros_like(x)
            for i in range(N):
                noise = torch.normal(mean=0,std=std_dev,size=x.shape)
                new_point = torch.zeros_like(x)
                new_point = new_point + x + noise
                self._nquery[k] += 1
                if self._labels[k] == self._model(new_point).argmax(1):
                    s = s - noise
                    self._badquery[k] += 1
                else:
                    s = s + noise
            res.append(s[0]/N)
        res = torch.stack(res)
        norms = torch.norm(res.view(len(res),-1),dim=1)
        for i in range(len(res)):
            res[i] = res[i]/norms[i]
        vec_v = []
        for i in range(len(res)):
            if self._keep_directions and self._method != 'geoda':
                res[i] = self._gram_schmidt_list(self._orthogonal_directions[:][i],res[i])
            else:
                res[i] = self._gram_schmidt(u[i],res[i])
            norms = torch.norm(res[i].view(len(res[i]),-1),dim=1)
            res[i] = res[i]/norms
            vec_v.append(res[i])

        vec_v = torch.stack(vec_v)
        return vec_v
    
    """
    Get the number of queries

    Returns
    -------
    torch.Tensor
        number of queries for each input
    """
    def _get_nquery(self):
        return self._nquery
    
    """
    Apply the DCT on a block

    Parameters
    ----------
    block : np.array
        block to apply the DCT

    Returns
    -------
    np.array
        block after the DCT
    """
    def _apply_dct(self,block):
        return dct(dct(block.T, norm='ortho').T, norm='ortho')

    """
    Apply the Inverse DCT on a block

    Parameters
    ----------
    block : np.array
        block to apply the Inverse DCT

    Returns
    -------
    np.array
        block after the Inverse DCT
    """
    def _apply_idct(self,block):
        return idct(idct(block.T, norm='ortho').T, norm='ortho')

    """
    Process the image with the DCT

    Parameters
    ----------
    image_tensor : torch.Tensor
        image to process
    block_size : int
        size of the block for the DCT
    num_high_freq_to_remove : int
        number of high frequencies to remove

    Returns
    -------
    torch.Tensor
        image after the DCT

    Warning
    -------
    The input must be a tensor of shape (n, C, H, W) where n is the number of inputs to attack
    """
    def _process_image_with_dct(self,image_tensor, block_size=8,num_high_freq_to_remove=1):
        res_dct = []
        for cpt in range(len(image_tensor)):
            image = image_tensor[cpt]
            c, h, w = image.size()
            dct_blocks = torch.zeros_like(image)
            mask = self._create_mask(block_size, num_high_freq_to_remove)

            for i in range(0, h, block_size):
                for j in range(0, w, block_size):
                    for k in range(c):  
                        block = image[k, i:i+block_size, j:j+block_size].cpu().numpy()
                        if block.shape == (block_size, block_size):             
                            dct_block = self._apply_dct(block)
                            if num_high_freq_to_remove > 0:
                                dct_block *= mask  
                            dct_blocks[k, i:i+block_size, j:j+block_size] = torch.from_numpy(dct_block)

            res_dct.append(dct_blocks)
        dct_blocks = torch.stack(res_dct)

        return dct_blocks
    
    """
    Create the mask to remove the high frequencies

    Parameters
    ----------
    block_size : int
        size of the block for the DCT
    num_high_freq_to_remove : int
        number of high frequencies to remove
    
    Returns
    -------
    np.array
        mask to remove the high frequencies
    """
    def _create_mask(self,size, num_high_freq_to_remove):
        mask = np.zeros((size, size))
        num_coefficients = size * size
        num_low_freq_to_keep = num_coefficients - num_high_freq_to_remove
        zigzag_indices = []
        for i in range(2 * size - 1):
            if i % 2 == 0:
                for j in range(i + 1):
                    x = j
                    y = i - j
                    if x < size and y < size:
                        zigzag_indices.append((x, y))
            else:
                for j in range(i + 1):
                    x = i - j
                    y = j
                    if x < size and y < size:
                        zigzag_indices.append((x, y))
        for idx in zigzag_indices[:num_low_freq_to_keep]:
            mask[idx] = 1
        
        return mask
    
    """
    Reconstruct the image from the DCT

    Parameters
    ----------
    dct_image : torch.Tensor
        image after the DCT
    block_size : int
        size of the block for the DCT

    Returns
    -------
    torch.Tensor
        reconstructed image from the DCT
    
    Warning
    -------
    The input must be a tensor of shape (n, C, H, W) where n is the number of inputs to attack
    """
    def _reconstruct_image_from_dct(self,dct_image, block_size=8, do_clamping = True):
        res = []
        for cpt in range(len(dct_image)):
            image = dct_image[cpt]
            c, h, w = image.size()
            reconstructed_image = torch.zeros_like(image)

            for i in range(0, h, block_size):
                for j in range(0, w, block_size):
                    for k in range(c): 
                        dct_block = image[k, i:i+block_size, j:j+block_size].cpu().numpy()
                        if dct_block.shape == (block_size, block_size):
                            block = self._apply_idct(dct_block)
                            reconstructed_image[k, i:i+block_size, j:j+block_size] = torch.from_numpy(block)

            res.append(reconstructed_image)
        res = torch.stack(res)
        if do_clamping:
            return res.clamp(0,1)
        else: 
            return res
    
    """
    SurFree method for the random vector with the DCT

    Returns
    -------
    torch.Tensor
        Random vector for SurFree method with the DCT
    """
    def _random_vector_dct(self):
        random_v = torch.tanh(torch.abs(self._process_image_with_dct(self._X)))
        r = torch.rand(random_v.shape).to(device)
        r [r < 1/3 ] = -1
        r [(1/3 <= r) & (r < 2/3)] = 0
        r [r >= 2/3] = 1
        random_v = random_v * r
        norms = torch.norm(random_v.view(len(random_v),-1),dim=1)
        for i in range(len(random_v)):
            random_v[i] = random_v[i]/norms[i]
        return random_v
    
    """
    GeoDA method for the normal vector estimation with the DCT

    Parameters
    ----------
    X : torch.Tensor
        input to attack
    N : int
        number of samples to estimate the normal vector
    std_dev : float
        standard deviation for the noise added to estimate the normal vector
    block_size : int
        size of the block for the DCT
    num_high_freq_to_remove : int
        number of high frequencies to remove

    Returns
    -------
    torch.Tensor
        estimated normal vector with GeoDA estimator with the DCT

    Warning
    -------
    The input must be a tensor of shape (n, C, H, W) where n is the number of inputs to attack
    """
    def _estimated_normal_vector_dct(self,X, N, std_dev,block_size, num_high_freq_to_remove):
        res = []
        D =self._process_image_with_dct(X, block_size=block_size,num_high_freq_to_remove=0)
        for k in range(len(X)):
            x = D[k].unsqueeze(0)
            s = torch.zeros_like(x)
            for i in range(N):
                noise = torch.normal(mean=0,std=std_dev,size=(x.shape))
                mask=torch.from_numpy(self._create_mask(block_size, num_high_freq_to_remove))
                mask = torch.tile(mask,(int(D[k].shape[1]/block_size),int(D[k].shape[1]/block_size))).unsqueeze(0).repeat(3,1,1)
                noise = noise*mask
                noise = noise.to(torch.float32)   
         

                new_point = torch.zeros_like(x)
                noise = x*noise.to(device)
                new_point = new_point + x + noise
                n = self._reconstruct_image_from_dct(new_point, block_size=block_size, do_clamping=False)
                self._nquery[k] += 1
                if self._labels[k] == self._model(n).argmax(1):
                    s = s - noise
                    self._badquery[k] += 1
                else:
                    s = s + noise
            res.append(s[0]/N)
        res = torch.stack(res)
        norms = torch.norm(res.view(len(res),-1),dim=1)
        for i in range(len(res)):
            res[i] = res[i]/norms[i]
        return self._reconstruct_image_from_dct(res,block_size=block_size,do_clamping=False)

    """
    SurFree method with the random vector

    Returns
    -------
    torch.Tensor
        optimal adversarial example found after n_steps iterations with the SurFree method
    """
    def _surfree(self):
        if self._starting_point is None:
            points_on_boundary = self._find_boundary(self._X)
        else:
            points_on_boundary = torch.zeros_like(self._starting_point)
            points_on_boundary = points_on_boundary + self._starting_point
        vec_u = torch.zeros_like(self._X)
        vec_u = vec_u + points_on_boundary - self._X
        u = []
        for i in range(len(vec_u)):
            t = vec_u[i].reshape(np.prod(vec_u[i].shape))
            u.append((t/torch.norm(t)).reshape(vec_u[i].shape))
        u = torch.stack(u)

        b_point = torch.zeros_like(points_on_boundary)
        b_point = b_point + points_on_boundary
        u_vec = torch.zeros_like(u)
        u_vec = u_vec + u
        self._orthogonal_directions.append(u_vec)
        D = torch.zeros_like(self._X)
        D = D + self._X
        results1 = []
        quer = []
        badquer = []
        quer.append(torch.zeros_like(self._nquery))
        badquer.append(torch.zeros_like(self._badquery))
        results1.append(distance(D,b_point))
        for i in range(self._steps):
            v_vec = self._random_vector(u_vec)
            M = (D + b_point)/2
            r = distance(M,D)
            b_point = self._binary_search_along_circle(b_point, M,u_vec, v_vec,r, eps = self._theta_eps)
            if i%100 == 83:
                print(i, distance(D,b_point))
            results1.append(distance(D,b_point))
            u_vec = b_point - D
            self._orthogonal_directions.append(v_vec)
            norms = torch.norm(u_vec.view(len(u_vec),-1),dim=1)
            for j in range(len(u_vec)):
                u_vec[j] = u_vec[j]/norms[j]
            if i%100 == 83:
                print(self._get_nquery(), self._badquery)
            n_tensor = torch.zeros_like(self._nquery)
            n_tensor+= self._nquery
            quer.append(n_tensor)
            n_btensor = torch.zeros_like(self._badquery)
            n_btensor+= self._badquery
            badquer.append(n_btensor)
            if (self._nquery >= self._max_queries).any():
                print('Maximum number of queries reached')
                break
        results1 = torch.stack(results1)
        quer = torch.stack(quer)
        badquer = torch.stack(badquer)
        return b_point, results1,quer,badquer

    """
    CGBA

    Returns
    -------
    torch.Tensor
        optimal adversarial example found after n_steps iterations with the CGBA
    """
    def _cgba(self):
        if self._starting_point is None:
            points_on_boundary = self._find_boundary(self._X)
        else:
            points_on_boundary = torch.zeros_like(self._starting_point)
            points_on_boundary = points_on_boundary + self._starting_point
        vec_u = torch.zeros_like(self._X)
        vec_u = vec_u + points_on_boundary - self._X
        u = []
        for i in range(len(vec_u)):
            t = vec_u[i].reshape(np.prod(vec_u[i].shape))
            u.append((t/torch.norm(t)).reshape(vec_u[i].shape))
        u = torch.stack(u)
        b_point = torch.zeros_like(points_on_boundary)
        b_point = b_point + points_on_boundary
        u_vec = torch.zeros_like(u)
        u_vec = u_vec + u
        self._orthogonal_directions.append(u_vec)
        D = torch.zeros_like(self._X)
        D = D + self._X
        results2 = []
        quer = []
        badquer = []
        quer.append(torch.zeros_like(self._nquery))
        badquer.append(torch.zeros_like(self._badquery))
        results2.append(distance(D,b_point))
        for i in range(self._steps):
            v_vec = self._estimated_normal_vector(b_point,u_vec,self._N, self._std_dev)
            M = (D + b_point)/2
            r = distance(M,D)
            b_point = self._binary_search_along_circle(b_point, M,u_vec, v_vec, r,eps=self._theta_eps)
            if i%100 == 83:
                print(i,distance(D,b_point))
            results2.append(distance(D,b_point))
            u_vec = b_point - D
            norms = torch.norm(u_vec.view(len(u_vec),-1),dim=1)
            for j in range(len(u_vec)):
                u_vec[j] = u_vec[j]/norms[j]
            self._orthogonal_directions.append(v_vec)
            if i%100 == 83:
                print(self._get_nquery(), self._badquery)
            n_tensor = torch.zeros_like(self._nquery)
            n_tensor+= self._nquery
            quer.append(n_tensor)
            n_btensor = torch.zeros_like(self._badquery)
            n_btensor+= self._badquery
            badquer.append(n_btensor)
            if (self._nquery >= self._max_queries).any():
                print('Maximum number of queries reached')
                break
        results2 = torch.stack(results2)
        quer = torch.stack(quer)
        badquer = torch.stack(badquer)
        return b_point, results2,quer,badquer
    
    """
    SurFree method with the DCT

    Returns
    -------
    torch.Tensor
        optimal adversarial example found after n_steps iterations with the SurFree method with the DCT
    """
    def _surfree_with_dct(self):
        if self._starting_point is None:
            points_on_boundary = self._find_boundary(self._X)
        else:
            points_on_boundary = torch.zeros_like(self._starting_point)
            points_on_boundary = points_on_boundary + self._starting_point
        vec_u = torch.zeros_like(self._X)
        vec_u = vec_u + points_on_boundary - self._X
        u = []
        for i in range(len(vec_u)):
            t = vec_u[i].reshape(np.prod(vec_u[i].shape))
            u.append((t/torch.norm(t)).reshape(vec_u[i].shape))
        u = torch.stack(u)
        
        b_point = torch.zeros_like(points_on_boundary)
        b_point = b_point + points_on_boundary
        u_vec = torch.zeros_like(u)
        u_vec += u
        self._orthogonal_directions.append(u_vec)
        D = torch.zeros_like(self._X)
        D = D + self._X
        results1 = []
        quer = []
        badquer = []
        results1.append(distance(D,b_point))
        quer.append(torch.zeros_like(self._nquery))
        badquer.append(torch.zeros_like(self._badquery))

        for i in range(self._steps):
            v_vec = self._reconstruct_image_from_dct(self._random_vector_dct(),block_size=self._dct_size)
            v_vec = self._gram_schmidt_list(self._orthogonal_directions,v_vec)
            v_vec = v_vec/torch.norm(v_vec.view(len(v_vec),-1),dim=1).view(-1,1,1,1)
            M = (D + b_point)/2
            r = distance(M,D)
            b_point = self._binary_search_along_circle(b_point, M,u_vec, v_vec,r)
            if i%4 == 0:
                print(i, distance(D,b_point))
            results1.append(distance(D,b_point))
            u_vec = (b_point - D)
            norms = torch.norm(u_vec.view(len(u_vec),-1),dim=1)
            for j in range(len(u_vec)):
                u_vec[j] = u_vec[j]/norms[j]
            self._orthogonal_directions.append(v_vec)
            if i%4 == 0:
                print(self._get_nquery(), self._badquery)
            n_tensor = torch.zeros_like(self._nquery)
            n_tensor+= self._nquery
            quer.append(n_tensor)
            n_btensor = torch.zeros_like(self._badquery)
            n_btensor+= self._badquery
            badquer.append(n_btensor)
            self._images_finished = n_tensor >= self._max_queries
            if (torch.all(self._images_finished)):
                print('Maximum number of queries reached')
                break
        results1 = torch.stack(results1)
        quer = torch.stack(quer)
        badquer = torch.stack(badquer)
        return b_point, results1,quer, badquer
    
    """
    CGBA with DCT

    Returns
    -------
    torch.Tensor
        optimal adversarial example found after n_steps iterations with the CGBA with the DCT
    """
    def _cgba_with_dct(self):
        if self._starting_point is None:
            points_on_boundary = self._find_boundary(self._X)
        else:
            points_on_boundary = torch.zeros_like(self._starting_point)
            points_on_boundary = points_on_boundary + self._starting_point
        vec_u = torch.zeros_like(self._X)
        vec_u = vec_u + points_on_boundary - self._X
        u = []
        for i in range(len(vec_u)):
            t = vec_u[i].reshape(np.prod(vec_u[i].shape))
            u.append((t/torch.norm(t)).reshape(vec_u[i].shape))
        u = torch.stack(u)
        
        b_point = torch.zeros_like(points_on_boundary)
        b_point = b_point + points_on_boundary
        
        u_vec = torch.zeros_like(u)
        u_vec += u
        self._orthogonal_directions.append(u_vec)
        D = torch.zeros_like(self._X)
        D = D + self._X
        results1 = []
        quer = []
        badquer = []
        results1.append(distance(D,b_point))
        quer.append(torch.zeros_like(self._nquery))
        badquer.append(torch.zeros_like(self._badquery))

        for i in range(self._steps):
            if self._increasing:
                dct_v = self._estimated_normal_vector_dct(b_point,min(250,self._N*int(np.sqrt(i+1))), self._std_dev, block_size=self._dct_size, num_high_freq_to_remove=self._frequency_to_remove)
            elif self._decreasing:
                dct_v = self._estimated_normal_vector_dct(b_point,max(5,self._N//int(np.sqrt(i+1))), self._std_dev, block_size=self._dct_size, num_high_freq_to_remove=self._frequency_to_remove)
            else:
                dct_v = self._estimated_normal_vector_dct(b_point,self._N, self._std_dev, block_size=self._dct_size, num_high_freq_to_remove=self._frequency_to_remove)
            v_vec = dct_v
            v_vec = self._gram_schmidt_list(self._orthogonal_directions,v_vec)
            v_vec = v_vec/torch.norm(v_vec.view(len(v_vec),-1),dim=1).view(-1,1,1,1)
            M = (D + b_point)/2
            r = distance(M,D)
            b_point = self._binary_search_along_circle(b_point, M,u_vec, v_vec,r)
            if i%4 == 0:
                print(i, distance(D,b_point))
            results1.append(distance(D,b_point))
            u_vec = (b_point - D)
            norms = torch.norm(u_vec.view(len(u_vec),-1),dim=1)
            for j in range(len(u_vec)):
                u_vec[j] = u_vec[j]/norms[j]
            self._orthogonal_directions.append(v_vec)
            if i%4 == 0:
                print(self._get_nquery(), self._badquery)
            n_tensor = torch.zeros_like(self._nquery)
            n_tensor+= self._nquery
            quer.append(n_tensor)
            n_btensor = torch.zeros_like(self._badquery)
            n_btensor+= self._badquery
            badquer.append(n_btensor)
            if (self._nquery.mean() >= self._max_queries):
                print('Maximum number of queries reached')
                break
        results1 = torch.stack(results1)
        quer = torch.stack(quer)
        badquer = torch.stack(badquer)
        return b_point, results1,quer,badquer


    """
    GeoDA with DCT

    Returns
    -------
    torch.Tensor
        optimal adversarial example found after n_steps iterations with the GeoDA  with the DCT
    """
    def _geoda_with_dct(self):
        if self._starting_point is None:
            points_on_boundary = self._find_boundary(self._X)
        else:
            points_on_boundary = torch.zeros_like(self._starting_point)
            points_on_boundary = points_on_boundary + self._starting_point
        vec_u = torch.zeros_like(self._X)
        vec_u = vec_u + points_on_boundary - self._X
        u = []
        for i in range(len(vec_u)):
            t = vec_u[i].reshape(np.prod(vec_u[i].shape))
            u.append((t/torch.norm(t)).reshape(vec_u[i].shape))
        u = torch.stack(u)
        
        b_point = torch.zeros_like(points_on_boundary)
        b_point = b_point + points_on_boundary
        u_vec = torch.zeros_like(u)
        u_vec += u
        D = torch.zeros_like(self._X)
        D = D + self._X
        results1 = []
        quer = []
        badquer = []

        for i in range(self._steps):
            n_e = self._estimated_normal_vector_dct(b_point,self._N, self._std_dev, block_size=self._dct_size, num_high_freq_to_remove=self._frequency_to_remove)
            n_e = n_e/torch.norm(n_e.view(len(n_e),-1),dim=1).view(-1,1,1,1)
  
            b_point1 = self._find_boundary_in_direction(n_e)
            dis = distance(D,b_point1)
            if dis < distance(D,b_point):
                b_point = b_point1
            print(i, distance(D,b_point))
            results1.append(distance(D,b_point))
            u_vec = (b_point - D)
            norms = torch.norm(u_vec.view(len(u_vec),-1),dim=1)
            for j in range(len(u_vec)):
                u_vec[j] = u_vec[j]/norms[j]
            print(self._get_nquery(), self._badquery)
            n_tensor = torch.zeros_like(self._nquery)
            n_tensor+= self._nquery
            quer.append(n_tensor)
            n_btensor = torch.zeros_like(self._badquery)
            n_btensor+= self._badquery
            badquer.append(n_btensor)
            if (self._nquery.mean() >= self._max_queries):
                print('Maximum number of queries reached')
                break
        results1 = torch.stack(results1)
        quer = torch.stack(quer)
        badquer = torch.stack(badquer)
        return b_point, results1,quer,badquer
    


    """
    GeoDA

    Returns
    -------
    torch.Tensor
        optimal adversarial example found after n_steps iterations with the GeoDA
    """
    def _geoda(self):
        if self._starting_point is None:
            points_on_boundary = self._find_boundary(self._X)
        else:
            points_on_boundary = torch.zeros_like(self._starting_point)
            points_on_boundary = points_on_boundary + self._starting_point
        vec_u = torch.zeros_like(self._X)
        vec_u = vec_u + points_on_boundary - self._X
        u = []
        for i in range(len(vec_u)):
            t = vec_u[i].reshape(np.prod(vec_u[i].shape))
            u.append((t/torch.norm(t)).reshape(vec_u[i].shape))
        u = torch.stack(u)
        
        b_point = torch.zeros_like(points_on_boundary)
        b_point = b_point + points_on_boundary
        u_vec = torch.zeros_like(u)
        u_vec += u
        D = torch.zeros_like(self._X)
        D = D + self._X
        results1 = []
        quer = []
        badquer = []

        for i in range(self._steps):
            n_e = self._estimated_normal_vector(b_point,u_vec,self._N, self._std_dev)
            n_e = n_e/torch.norm(n_e.view(len(n_e),-1),dim=1).view(-1,1,1,1)
  
            b_point1 = self._find_boundary_in_direction(n_e)
            dis_old = distance(D,b_point)
            dis = distance(D,b_point1)
            for k in range(len(dis)):
                if dis[k] < dis_old[k]:
                    b_point[k] = b_point1[k]
            print(i, distance(D,b_point))
            results1.append(distance(D,b_point))
            u_vec = (b_point - D)
            norms = torch.norm(u_vec.view(len(u_vec),-1),dim=1)
            for j in range(len(u_vec)):
                u_vec[j] = u_vec[j]/norms[j]
            print(self._get_nquery(), self._badquery)
            n_tensor = torch.zeros_like(self._nquery)
            n_tensor+= self._nquery
            quer.append(n_tensor)
            n_btensor = torch.zeros_like(self._badquery)
            n_btensor+= self._badquery
            badquer.append(n_btensor)
            if (self._nquery.mean() >= self._max_queries):
                print('Maximum number of queries reached')
                break
        results1 = torch.stack(results1)
        quer = torch.stack(quer)
        badquer = torch.stack(badquer)
        return b_point, results1,quer,badquer
    