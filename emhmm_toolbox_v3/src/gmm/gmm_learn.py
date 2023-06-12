import numpy as np
import tensorflow_probability as tfp

from tqdm import tqdm # progress-meter

def gmm_learn(dataset, n_classes, n_iterations, random_seed):
    n_samples = dataset.shape[0]

    np.random.seed(random_seed)

    # Initial guesses for the parameters
    mus = np.random.rand(n_classes)
    sigmas = np.random.rand(n_classes)
    class_probs = np.random.dirichlet(np.ones(n_classes))

    for em_iter in tqdm(range(n_iterations)):
        # E-Step
        responsibilities = tfp.distributions.Normal(loc=mus, scale=sigmas).prob(
            dataset.reshape(-1, 1)
        ).numpy() * class_probs
        
        responsibilities /= np.linalg.norm(responsibilities, axis=1, ord=1, keepdims=True)

        class_responsibilities = np.sum(responsibilities, axis=0)

        # M-Step
        for c in range(n_classes):
            class_probs[c] = class_responsibilities[c] / n_samples
            mus[c] = np.sum(responsibilities[:, c] * dataset) / class_responsibilities[c]
            sigmas[c] = np.sqrt(
                np.sum(responsibilities[:, c] * (dataset - mus[c])**2) / class_responsibilities[c]
            )
    
    return class_probs, mus, sigmas
