import numpy as np
from scipy.special import psi, gammaln

def vbhmm_em(data, K, ini):
    VERBOSE_MODE = ini["verbose"]

    # get length of each chain
    trial = len(data)
    datalen = [d.shape[0] for d in data]
    lengthT = max(datalen)  # find the longest chain
    totalT = sum(datalen)

    # initialize the parameters
    mix_t = vbhmm_init(data, K, ini)  # initialize the parameters
    mix = mix_t
    dim = mix["dim"]  # dimension of the data
    K = mix["K"]  # no. of hidden states
    N = trial  # no. of chains
    maxT = lengthT  # the longest chain
    alpha0 = mix["alpha0"]  # hyper-parameter for the priors
    epsilon0 = mix["epsilon0"]  # hyper-parameter for the transitions
    m0 = mix["m0"]  # hyper-parameter for the mean
    beta0 = mix["beta0"]  # hyper-parameter for beta (Gamma)
    v0 = mix["v0"]  # hyper-parameter for v (Inverse-Wishart)
    W0inv = mix["W0inv"]  # hyper-parameter for Inverse-Wishart
    alpha = mix["alpha"]  # priors
    epsilon = mix["epsilon"]  # transitions
    beta = mix["beta"]  # beta (Gamma)
    v = mix["v"]  # v (Inverse-Wishart)
    m = mix["m"]  # mean
    W = mix["W"]  # Inverse-Wishart
    C = mix["C"]  # covariance
    const = mix["const"]  # constants
    const_denominator = mix["const_denominator"]  # constants
    maxIter = ini["maxIter"]  # maximum iterations allowed
    minDiff = ini["minDiff"]  # termination criterion

    L = -np.inf  # log-likelihood
    lastL = -np.inf  # log-likelihood

    # setup groups
    if "groups" in ini and ini["groups"]:
        usegroups = True
        group_ids = np.unique(ini["groups"])  # unique group ids
        numgroups = len(group_ids)
        group_inds = [np.where(ini["groups"] == g)[0] for g in group_ids]
        group_map = np.zeros(len(ini["groups"]))
        for g in range(numgroups):
            group_map[group_inds[g]] = g + 1
        # reshape alpha, epsilon into list
        # also Nk1 and M are lists
        tmp = epsilon
        tmpa = alpha
        epsilon = [tmp] * numgroups
        alpha = [tmpa] * numgroups
    else:
        usegroups = False

    for _ in range(maxIter):
        if VERBOSE_MODE:
            print("Current iteration:", _)

        # E-step
        fbhmm_varpar = {
            "v": v,
            "W": W,
            "epsilon": epsilon,
            "alpha": alpha,
            "m": m,
            "beta": beta
        }
        fbopt = {"usegroups": usegroups}
        if usegroups:
            fbopt["group_map"] = group_map
            fbopt["numgroups"] = numgroups
        
        # call FB algorithm
        fbstats = vbhmm_fb(data, fbhmm_varpar, fbopt)

        Nk = fbstats["Nk"]
        sum_xbar = fbstats["sum_xbar"]
        sum_xxT = fbstats["sum_xxT"]
        Tcount = fbstats["Tcount"]
        Tgamma = fbstats["Tgamma"]
        # update parameters
        if usegroups:
            for g in range(numgroups):
                gind = group_inds[g]
                Nk1 = Nk[gind, :]
                if np.any(Nk1):
                    alpha[g] = alpha0 + Nk1
                    epsilon[g] = epsilon0 + Tcount[gind, :, :]

                    # update mean and covariance
                    beta[g] = beta0 + Nk1
                    m[g, :] = (beta0 * m0 + sum_xbar[gind, :]) / beta[g]
                    v[g] = v0 + Nk1
                    W[g] = W0inv + sum_xxT[gind, :, :] + \
                        (beta0 * Nk1.reshape((-1, 1, 1))) * \
                        np.dot(m0.reshape((1, -1)), m0.reshape((1, -1)).T) - \
                        beta[g].reshape((-1, 1, 1)) * \
                        np.dot(m[g, :].reshape((1, -1)), m[g, :].reshape((1, -1)).T)
                    W[g] = np.linalg.inv(W[g])

                    # compute the responsibilities
                    logR = psi(epsilon[g]) - psi(np.sum(epsilon[g], axis=1)).reshape((-1, 1))
                    logR0 = np.log(alpha[g]) - np.log(np.sum(alpha[g]))
                    R = np.exp(logR + logR0)
                else:
                    alpha[g] = alpha0
                    epsilon[g] = epsilon0
                    beta[g] = beta0
                    m[g, :] = m0
                    v[g] = v0
                    W[g] = W0inv
                    R = np.zeros((T, K))

                mix_t = {
                    "dim": dim,
                    "K": K,
                    "alpha0": alpha0,
                    "epsilon0": epsilon0,
                    "m0": m0,
                    "beta0": beta0,
                    "v0": v0,
                    "W0inv": W0inv,
                    "alpha": alpha[g],
                    "epsilon": epsilon[g],
                    "beta": beta[g],
                    "v": v[g],
                    "m": m[g, :],
                    "W": W[g],
                    "C": C[g, :, :],
                    "const": const,
                    "const_denominator": const_denominator
                }
                mix[g] = mix_t
        else:
            Nk1 = np.sum(Nk, axis=0)
            alpha = alpha0 + Nk1
            epsilon = epsilon0 + Tcount

            # update mean and covariance
            beta = beta0 + Nk1
            m = (beta0 * m0 + sum_xbar) / beta
            v = v0 + Nk1
            W = W0inv + sum_xxT + \
                (beta0 * Nk1.reshape((-1, 1, 1))) * \
                np.dot(m0.reshape((1, -1)), m0.reshape((1, -1)).T) - \
                beta.reshape((-1, 1, 1)) * \
                np.dot(m.reshape((1, -1)), m.reshape((1, -1)).T)
            W = np.linalg.inv(W)

            # compute the responsibilities
            logR = psi(epsilon) - psi(np.sum(epsilon, axis=1)).reshape((-1, 1))
            logR0 = np.log(alpha) - np.log(np.sum(alpha))
            R = np.exp(logR + logR0)

            mix_t = {
                "dim": dim,
                "K": K,
                "alpha0": alpha0,
                "epsilon0": epsilon0,
                "m0": m0,
                "beta0": beta0,
                "v0": v0,
                "W0inv": W0inv,
                "alpha": alpha,
                "epsilon": epsilon,
                "beta": beta,
                "v": v,
                "m": m,
                "W": W,
                "C": C,
                "const": const,
                "const_denominator": const_denominator
            }
            mix = mix_t

        # M-step
        L = vbhmm_computeLikelihood(Tgamma, alpha, epsilon)
        for k in range(K):
            if usegroups:
                for g in range(numgroups):
                    gind = group_inds[g]
                    Nk1 = Nk[gind, :]
                    if np.any(Nk1):
                        C[g, k, :] = (1 / beta[g, k]) * \
                            (np.sum(R[gind, :, k].reshape((-1, 1)) *
                                     (data[gind, :] - m[g, :]), axis=0) + beta0 * m0)
            else:
                Nk1 = np.sum(Nk, axis=0)
                if np.any(Nk1):
                    C[k, :] = (1 / beta[k]) * \
                        (np.sum(R[:, k].reshape((-1, 1)) *
                                 (data - m[k, :]), axis=0) + beta0 * m0)

        if VERBOSE_MODE:
            print("Log-likelihood:", L)

        # check termination criterion
        if (L - lastL) < minDiff:
            break
        lastL = L

    if VERBOSE_MODE:
        print("Final log-likelihood:", L)

    result = {
        "mix": mix,
        "L": L
    }
    return result
