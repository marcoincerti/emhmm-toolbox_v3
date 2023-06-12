import numpy as np
from scipy.stats import multivariate_normal

def vbhmm_init(datai, K, ini):
    VERBOSE_MODE = ini['verbose']
    data = np.concatenate(datai, axis=0)
    N, dim = data.shape

    mix_t = {}
    mix_t['dim'] = dim
    mix_t['K'] = K
    mix_t['N'] = N

    # initialize GMM with random initialization
    if ini['initmode'] == 'random':
        try:
            if ini['random_gmm_opt'] is not None:
                random_gmm_opt = ini['random_gmm_opt']
                if VERBOSE_MODE >= 3:
                    print('random gmm init: passing random_gmm_opt =', random_gmm_opt)
            else:
                random_gmm_opt = {}

            # use the scipy.stats multivariate_normal.fit function to find the GMM components
            mix = multivariate_normal.fit(data, K, **random_gmm_opt)

        except Exception as e:
            if 'IllCondCovIter' in str(e):
                if VERBOSE_MODE >= 2:
                    print('using shared covariance')

                # fit GMM with shared covariance
                mix = multivariate_normal.fit(data, K, cov_type='diag', shared_covariance=True, regularize=0.0001)

            else:
                if VERBOSE_MODE >= 2:
                    print('using built-in GMM')

                # use built-in GMM function (if multivariate_normal.fit is not available)
                gmmopt = {'cvmode': 'full', 'initmode': 'random', 'verbose': 0}
                gmmmix = gmm_learn(data.T, K, gmmopt)
                mix = {
                    'PComponents': gmmmix['pi'],
                    'mu': np.concatenate(gmmmix['mu'], axis=1).T,
                    'Sigma': np.concatenate(gmmmix['cv'], axis=2)
                }

    # initialize GMM with given GMM
    elif ini['initmode'] == 'initgmm':
        mix = {}
        mix['Sigma'] = np.concatenate(ini['initgmm']['cov'], axis=2)
        mix['mu'] = np.concatenate(ini['initgmm']['mean'], axis=0)
        if mix['mu'].shape[0] != K:
            raise ValueError('bad initgmm dimensions -- possibly mean is not a row vector')

        mix['PComponents'] = ini['initgmm']['prior']

    # initialize GMM with component splitting
    elif ini['initmode'] == 'split':
        gmmopt = {'cvmode': 'full', 'initmode': 'split', 'verbose': 0}
        gmmmix = gmm_learn(data.T, K, gmmopt)
        mix = {
            'PComponents': gmmmix['pi'],
            'mu': np.concatenate(gmmmix['mu'], axis=1).T,
            'Sigma': np.concatenate(gmmmix['cv'], axis=2)
        }

    else:
        raise ValueError('bad initmode')

    mix_t['alpha0'] = ini['alpha']
    mix_t['epsilon0'] = ini['epsilon']
    mix_t['m0'] = ini['mu']
    mix_t['beta0'] = ini['beta']
    if isinstance(ini['W'], float):
        # isotropic W
        mix_t['W0'] = ini['W'] * np.eye(dim)
    else:
        # diagonal W
        if len(ini['W']) != dim:
            raise ValueError(f'vbopt.W should have dimension D={dim} for diagonal matrix')
        mix_t['W0'] = np.diag(ini['W'])
    if ini['v'] <= dim - 1:
        raise ValueError('v not large enough')
    mix_t['v0'] = ini['v']
    mix_t['W0inv'] = np.linalg.inv(mix_t['W0'])

    # setup model (M-step)
    mix_t['Nk'] = N * mix['PComponents']
    mix_t['Nk2'] = N * np.tile(1/K, K)

    mix_t['xbar'] = mix['mu']

    if mix['Sigma'].shape[2] == K:
        mix_t['S'] = mix['Sigma']
    elif mix['Sigma'].shape[2] == 1:
        # handle shared covariance
        mix_t['S'] = np.tile(mix['Sigma'], (1, 1, K))

    if mix_t['S'].shape[0] == 1 or mix_t['S'].shape[1] == 1 and dim > 1:
        oldS = mix_t['S']
        mix_t['S'] = np.zeros((dim, dim, K))
        for j in range(K):
            mix_t['S'][:, :, j] = np.diag(oldS[:, :, j])

    mix_t['alpha'] = mix_t['alpha0'] + mix_t['Nk2']
    mix_t['epsilon'] = mix_t['epsilon0'] + mix_t['Nk2']
    mix_t['beta'] = mix_t['beta0'] + mix_t['Nk']
    mix_t['v'] = mix_t['v0'] + mix_t['Nk'] + 1
    mix_t['m'] = ((mix_t['beta0'] * mix_t['m0']) * np.ones((1, K)) + (mix_t['Nk'].reshape((-1, 1)) * mix_t['xbar'])) / (mix_t['beta'].reshape((-1, 1)))
    mix_t['W'] = np.zeros((dim, dim, K))
    for k in range(K):
        mult1 = mix_t['beta0'] * mix_t['Nk'][k] / (mix_t['beta0'] + mix_t['Nk'][k])
        diff3 = mix_t['xbar'][:, k] - mix_t['m0']
        mix_t['W'][:, :, k] = np.linalg.inv(mix_t['W0inv'] + mix_t['Nk'][k] * mix_t['S'][:, :, k] + mult1 * np.outer(diff3, diff3))

    mix_t['C'] = np.zeros((dim, dim, K))
    mix_t['const'] = dim * np.log(2)
    mix_t['const_denominator'] = (dim * np.log(2 * np.pi)) / 2

    return mix_t
