#This is the python version of SueYeon's code on matlab
import numpy as np
from scipy.linalg import qr

from cvxopt import solvers, matrix


# Configure cvxopt solvers
solvers.options['show_progress'] = False
solvers.options['maxiters'] = 1000000
solvers.options['abstol'] = 1e-10
solvers.options['reltol'] = 1e-10
solvers.options['feastol'] = 1e-10


def manifold_analysis(XtotT, kappa, n_t, t_vecs=None):
    '''
    Carry out the analysis on multiple manifolds.

    Args:
        XtotT: Sequence of 2D arrays of shape (N, P_i) where N is the dimensionality
                of the space, and P_i is the number of sampled points for the i_th manifold.
        kappa: Margin size to use in the analysis (scalar, kappa > 0)
        n_t: Number of gaussian vectors to sample per manifold
        t_vecs: Optional sequence of 2D arrays of shape (Dm_i, n_t) where Dm_i is the reduced
                dimensionality of the i_th manifold. Contains the gaussian vectors to be used in
                analysis.  If not supplied, they will be randomly sampled for each manifold.

    Returns:
        a_Mfull_vec: 1D array containing the capacity calculated from each manifold
        R_M_vec: 1D array containing the calculated anchor radius of each manifold
        D_M_vec: 1D array containing the calculated anchor dimension of each manifold.
    '''
    # Number of manifolds to analyze
    num_manifolds = len(XtotT)
    # Compute the global mean over all samples
    Xori = np.concatenate(XtotT, axis=1)
    X_origin = np.mean(Xori, axis=1, keepdims=True)
    # Subtract the mean from each manifold
    Xtot0 = [XtotT[i] - X_origin for i in range(num_manifolds)]
    # Compute the mean for each manifold
    means = [np.mean(Xtot0[i], axis=1, keepdims=True) for i in range(num_manifolds)]
    # Normalize the center for each manifold
    XtotInput = [(Xtot0[i] - means[i])/np.linalg.norm(means[i]) for i in range(num_manifolds)]

    a_Mfull_vec = np.zeros(num_manifolds)
    R_M_vec = np.zeros(num_manifolds)
    D_M_vec = np.zeros(num_manifolds)
    # Make the D+1 dimensional data
    for i in range(num_manifolds):
        S_r = XtotInput[i]
        D, m = S_r.shape
        # Project the data onto a smaller subspace
        if D > m:
            Q, R = qr(S_r, mode='economic')
            S_r = np.matmul(Q.T, S_r)
            # Get the new sizes
            D, m = S_r.shape
        # Add the center dimension
        sD1 = np.concatenate([S_r, np.ones((1, m))], axis=0)

        # Carry out the analysis on the i_th manifold
        if t_vecs is not None:
            a_Mfull, R_M, D_M = each_manifold_analysis_D1(sD1, kappa, n_t, t_vec=t_vecs[i])
        else:
            a_Mfull, R_M, D_M = each_manifold_analysis_D1(sD1, kappa, n_t)

        # Store the results
        a_Mfull_vec[i] = a_Mfull
        R_M_vec[i] = R_M
        D_M_vec[i] = D_M
    return a_Mfull_vec, R_M_vec, D_M_vec


def each_manifold_analysis_D1(sD1, kappa, n_t, eps=1e-8, t_vec=None):
    '''
    This function computes the manifold capacity a_Mfull, the manifold radius R_M, and manifold dimension D_M
    with margin kappa using n_t randomly sampled vectors for a single manifold defined by a set of points sD1.

    Args:
        sD1: 2D array of shape (D+1, m) where m is number of manifold points 
        kappa: Margin size (scalar)
        n_t: Number of randomly sampled vectors to use
        eps: Minimal distance (default 1e-8)
        t_vec: Optional 2D array of shape (D+1, m) containing sampled t vectors to use in evaluation

    Returns:
        a_Mfull: Calculated capacity (scalar)
        R_M: Calculated radius (scalar)
        D_M: Calculated dimension (scalar)
    '''
    # Get the dimensionality and number of manifold points
    D1, m = sD1.shape # D+1 dimensional data
    D = D1-1
    # Sample n_t vectors from a D+1 dimensional standard normal distribution unless a set is given
    if t_vec is None:
        t_vec = np.random.randn(D1, n_t)
    # Find the corresponding manifold point for each random vector
    ss, gg = maxproj(t_vec, sD1)
    
    # Compute V, S~ for each random vector
    s_all = np.empty((D1, n_t))
    f_all = np.zeros(n_t)
    for i in range(n_t):
        # Get the t vector to use (keeping dimensions)
        t = np.expand_dims(t_vec[:, i], axis=1)
        # TODO: Double check signs here
        if gg[i] + kappa < 0:
            # For this case, a solution with V = T is allowed by the constraints, so we don't need to 
            # find it numerically
            v_f = t
            s_f = ss[:, i].reshape(-1, 1)
        else:
            # Get the solution for this t vector
            v_f, _, _, alpha, vminustsqk = minimize_vt_sq(t, sD1, kappa=kappa)
            f_all[i] = vminustsqk
            # If the solution vector is within eps of t, set them equal (interior point)
            if np.linalg.norm(v_f - t) < eps:
                v_f = t
                s_f = ss[:, i].reshape(-1, 1)
            else:
                # Otherwise, compute S~ from the solution
                scale = np.sum(alpha)
                s_f = (t - v_f)/scale
        # Store the calculated values
        s_all[:, i] = s_f[:, 0]

    # Compute the capacity from eq. 16, 17 in 2018 PRX paper.
    max_ts = np.maximum(np.sum(t_vec * s_all, axis=0) + kappa, np.zeros(n_t))
    s_sum = np.sum(np.square(s_all), axis=0)
    lamb = np.asarray([max_ts[i]/s_sum[i] if s_sum[i] > 0 else 0 for i in range(n_t)])
    slam = np.square(lamb) * s_sum
    a_Mfull = 1/np.mean(slam)

    # Compute R_M from eq. 28 of the 2018 PRX paper
    ds0 = s_all - s_all.mean(axis=1, keepdims=True)
    ds = ds0[0:-1, :]/s_all[-1, :]
    ds_sq_sum = np.sum(np.square(ds), axis=0)
    R_M = np.sqrt(np.mean(ds_sq_sum))

    # Compute D_M from eq. 29 of the 2018 PRX paper
    t_norms = np.sum(np.square(t_vec[0:D, :]), axis=0, keepdims=True)
    t_hat_vec = t_vec[0:D, :]/np.sqrt(t_norms)
    s_norms = np.sum(np.square(s_all[0:D, :]), axis=0, keepdims=True)
    s_hat_vec = s_all[0:D, :]/np.sqrt(s_norms)
    ts_dot = np.sum(t_hat_vec * s_hat_vec, axis=0)
    # TODO: double check order of operations here
    D_M = D * np.square(np.mean(ts_dot))

    return a_Mfull, R_M, D_M


def maxproj(t_vec, sD1, sc=1):
    '''
    This function finds the point on a manifold (defined by a set of points sD1) with the largest projection onto
    each individual t vector given by t_vec.

    Args:
        t_vec: 2D array of shape (D+1, n_t) where D+1 is the dimension of the linear space, and n_t is the number
            of sampled vectors
        sD1: 2D array of shape (D+1, m) where m is number of manifold points
        sc: Value for center dimension (scalar, default 1)

    Returns:
        s0: 2D array of shape (D+1, n_t) containing the points with maximum projection onto corresponding t vector.
        gt: 1D array of shape (D+1) containing the value of the maximum projection of manifold points projected
            onto the corresponding t vector.
    '''
    # get the dimension and number of the t vectors
    D1, n_t = t_vec.shape
    D = D1 - 1
    # Get the number of samples for the manifold to be processed
    m = sD1.shape[1]
    # For each of the t vectors, find the maximum projection onto manifold points
    # Ignore the last of the D+1 dimensions (center dimension)
    #TODO: vectorize this loop
    s0 = np.zeros((D1, n_t))
    gt = np.zeros(n_t)
    for i in range(n_t):
        t = t_vec[:, i]
        # Project t onto the SD vectors and find the S vector with the largest projection
        max_S = np.argmax(np.dot(t[0:D], sD1[0:D]))
        sr = sD1[0:D, max_S]
        # Append sc to this vector
        s0[:, i] = np.append(sr, [sc])
        # Compute the projection of this onto t
        gt[i] = np.dot(t, s0[:, i])
    return s0, gt


def minimize_vt_sq(t, sD1, kappa=0):
    '''
    This function carries out the constrained minimization decribed in Sec IIIa of the 2018 PRX paper.
    Instead of minimizing F = ||V-T||^2, The actual function that is minimized will be
        F' = 0.5 * V^2 - T * V
    Which is related to F by F' = 0.5 * (F - T^2).  The solution is the same for both functions.

    This makes use of cvxopt.

    Args:
        t: A single T vector encoded as a 2D array of shape (D+1, 1)
        sD1: 2D array of shape (D+1, m) where m is number of manifold points
        kappa: Size of margin (default 0)

    Returns:
        v_f: D+1 dimensional solution vector encoded as a 2D array of shape (D+1, 1)
        vt_f: Final value of the objective function (which does not include T^2). May be negative.
        exitflag: Not used, but equal to 1 if a local minimum is found.
        alphar: Vector of lagrange multipliers at the solution. 
        normvt2: Final value of ||V-T||^2 at the solution.
    '''   
    D1 = t.shape[0]
    m = sD1.shape[1]
    # Construct the matrices needed for F' = 0.5 * V' * P * V - q' * V.
    # We will need P = Identity, and q = -T
    P = matrix(np.identity(D1))
    q = - t.astype(np.double)
    q = matrix(q)

    # Construct the constraints.  We need V * S - k > 0.
    # This means G = -S and h = -kappa
    # TODO: Double check the signs here
    G = sD1.T.astype(np.double)
    G = matrix(G)
    h =  - np.ones(m) * kappa
    h = h.T.astype(np.double)
    h = matrix(h)

    # Carry out the constrained minimization
    output = solvers.qp(P, q, G, h)

    # Format the output
    v_f = np.array(output['x'])
    vt_f = output['primal objective']
    if output['status'] == 'optimal':
        exitflag = 1
    else:
        exitflag = 0
    alphar = np.array(output['z'])

    # Compute the true value of the objective function
    normvt2 = np.square(v_f - t).sum()
    return v_f, vt_f, exitflag, alphar, normvt2
