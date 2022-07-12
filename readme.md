## Readme

If using any code in this repo, please cite: 

@article{chung2018classification,
  title={Classification and geometry of general perceptual manifolds},
  author={Chung, SueYeon and Lee, Daniel D and Sompolinsky, Haim},
  journal={Physical Review X},
  volume={8},
  number={3},
  pages={031003},
  year={2018},
  publisher={APS}
}

# To begin Jupyter-Lab tutorial

Open terminal
```
gh repo clone chung-neuroai-lab/cshl2022-deep-learning-manifolds-tutorial
pip install -r requirements.txt
pip install -e .
cd cshl2022-deep-learning-manifolds-tutorial
jupyter-lab
```
# Other notes
*Alternatively, you may download the notebook to run on google colab. However, you won't be able to do Part 3. Furthermore, for Part 1 and 2, you may have to run 
```!pip install torch torchvision==0.13```*

## Basic Usage of Manifold Geometry 
The input data X in most of these functions should be preprocessed such that X is a list of numpy arrays. The length of list X should be the number of manifolds P. Each array should have shape (N,M) where N is the number of dimensions/features/neurons and M is the number of examples that the manifold consists of.  

**X = [(N_1,M_1), (N_2,M_2), ... (N_P,M_P)]**

# Theoretical Manifold Analysis
To determine system capacity (**cap**), average manifold radius (**rad**), and average manifold dimension (**dim**) of a layer in X, 
```
from manifold_analysis import * 
kappa = 0    # Specify the margin (usually 0) 
n_t = 300    # Specify the number of Gaussian vectors to sample (200 or 300 is a good default) 
alpha, radius, dimension = manifold_analysis(X,kappa,n_t)
cap = 1/np.mean(1/alpha)
rad = np.mean(radius)
dim = np.mean(dimension)
```
**Please note that manifold_analysis does not account for correlation among manifolds. As a result, capacity using manifold_analysis is a very rough estimate. For a better capacity estimate, use manifold_analysis_corr shown below.**
```
from manifold_analysis_correlation import * 
alpha, radius, dimension, C0, KK = manifold_analysis_corr(X,kappa,n_t)
cap = 1/np.mean(1/alpha)
rad = np.mean(radius)
dim = np.mean(dimension)
```

