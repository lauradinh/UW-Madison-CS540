from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def load_and_center_dataset(filename):
    dataset = np.load(filename)

    return dataset - np.mean(dataset, axis = 0)

def get_covariance(dataset):
    n = len(dataset)
    covariance = np.dot(np.transpose(dataset), dataset)
    
    return covariance * (1/(n-1))


def get_eig(S, m):
    evalue, evector = eigh(S, subset_by_index= [len(S) - m,len(S) - 1])
    diagonal = np.diag(evalue)
    return np.flip(diagonal), np.flip(evector, axis=1)
    

def get_eig_perc(S, perc):
    evalue, evector = eigh(S)
    evalue = np.flip(evalue)
    evector = np.flip((evector), axis=1)
    #change eigenvectors from columns to rows
    evector = np.transpose(evector)
    
    #sum of all eigenvalues
    sum = 0
    for i in evalue:
        sum += i
    
    cer_perc_evalues = []
    cer_perc_evectors = []
    
    
    for val, vector in zip(evalue,evector):
        if val/sum > perc:
            cer_perc_evalues.append(val)
            cer_perc_evectors.append(vector)
        
    
    diagonal = np.diag(np.asarray(cer_perc_evalues))
    return diagonal, np.transpose(np.asarray(cer_perc_evectors))


def project_image(img, U):
    sum = 0
    for vector in np.transpose(U):
        alpha = np.dot(np.transpose(vector), img)
        sum += np.dot(alpha, vector)
    return sum


def display_image(orig, proj):
    #reshape images
    reshape_orig = np.transpose(orig.reshape((32, 32)))
    reshape_proj = np.transpose(proj.reshape((32, 32)))
    #creating figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (9,3))
    ax1.set_title('Original')
    ax2.set_title('Projection')
    
    #adding colorbars
    axes_orig = ax1.imshow(reshape_orig, aspect = 'equal')
    orig_cbar = fig.colorbar(axes_orig, ax=ax1)
    
    axes_proj = ax2.imshow(reshape_proj, aspect = 'equal')
    proj_cbar = fig.colorbar(axes_proj, ax=ax2)
    
    plt.show()
