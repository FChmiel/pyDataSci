"""Functions for testing significance"""


def binary_significance(dist1, dist2, conf=0.95):
    """
    Tests the null hypothesis that two binary distributions are drawn from the same distribution.
    
    Parameters:
    -----------
    dist1, np.array
        The first distribution, containing binary values only
    
    dist2, np.array
        The second distribution, containing binary values only
        
    Returns:
    --------
    z, float
        The test statistic
        
    reject, bool
        Whether to reject the null hypothesis 
    """
    
    p1, p2 = np.mean(dist1), np.mean(dist2)
    n1, n2 = len(dist1), len(dist2)
    p = (p1*n1 + p2*n2) / (n1+n2)
    
    # pooled version of z-test
    numerator = np.abs(p1-p2)
    denominator = p*(1-p) * ((1/n1) + (1/n2))
    z = numerator / denominator
    
    z_alpha = 1.96 # 95% confidence
    if z>z_alpha:
        reject = True
    
    return z, reject