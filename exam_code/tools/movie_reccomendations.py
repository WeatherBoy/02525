import numpy as np
from copy import deepcopy
    
def euclSimilarity(R_a, R_b):
    """
    Implementation of the similarity measure based on the euclidian distance
    (commonly known as the euclidian similarity).
    
    Parameters
    ----------
    R_a : list, required
        First vector (or list of scalars) of two which this function finds 
        the euclidian similarity between.
    R_b : list, required
        Second vector (or list of scalars) of two which this function finds 
        the euclidian similarity between.
    
    Raises
    ------
    Exception
        If the parameters aren't of equal size.
        (This function isn't sophisticated enough to handle such a case).
        
    Converted to Python from original MatLab script for the DTU course 02525,
    by Felix Bo Caspersen (s183319@student.dtu.dk)
    on the 20th of October 2022
    """
    
    # Local copies
    R_a_loc = deepcopy(np.array(R_a))
    R_b_loc = deepcopy(np.array(R_b))
    
    # input variable check
    if R_a_loc.size != R_b_loc.size:
        raise Exception("The two vectors to compare must have the same number of elements.") 
    
    # Dot product/ matrix multiplication in numpy: @
    # elementwise multiplication in numpy: * 
    d = np.sqrt((R_a_loc - R_b_loc) @ np.transpose(R_a_loc - R_b_loc))
    
    return 1/(1 + d)


def pearsonSimilarity(R_a, R_b):
    """
    Implementation of Pearson's similarity
    (Also known as Pearson's correlation coefficient)
    
    Parameters
    ----------
    R_a : list, required
        First vector (or list of scalars) of two which this function finds 
        Pearson's similarity between.
    R_b : list, required
        Second vector (or list of scalars) of two which this function finds 
        Pearson's similarity between.
    
    Raises
    ------
    Exception
        If the parameters aren't of equal size.
        (This function isn't sophisticated enough to handle such a case).
        
    Converted to Python from original MatLab script for the DTU course 02525,
    by Felix Bo Caspersen (s183319@student.dtu.dk)
    on the 20th of October 2022
    
    """
    
    # local copies
    R_a_loc = deepcopy(np.array(R_a))
    R_b_loc = deepcopy(np.array(R_b))
    
    # number of ratings
    n = R_a_loc.size

    # input variable check
    if n != R_b_loc.size:
        raise Exception("The two vectors to compare must have the same number of elements.") 

    # find mean values and standard deviations
    mu_a = np.mean(R_a_loc) # or np.sum(R_a_loc)/n
    mu_b = np.mean(R_b_loc) # or np.sum(R_b_loc)/n
    
    # ddof means Delta Degrees of Freedom.
    # The divisor in the calculation becomes n - ddof.
    # Both 02450 and Wikipedia denotes this divisor as the one for the
    # 'corrected sample standard deviation'.
    # See:
    # https://en.wikipedia.org/wiki/Standard_deviation#:~:text=sample%20standard%20deviation.-,Corrected,-sample%20standard%20deviation
    sigma_a = np.std(R_a_loc, ddof=1) # or np.sqrt(np.sum(np.power(R_a_loc - mu_a, 2))/(n - 1))
    sigma_b = np.std(R_b_loc, ddof=1) # or np.sqrt(np.sum(np.power(R_b_loc - mu_b, 2))/(n - 1))

    # find standardized rating vectors
    R_a_stand = (R_a_loc-mu_a)/sigma_a
    R_b_stand = (R_b_loc-mu_b)/sigma_b 

    # find the Pearson correlation
    return (R_b_stand @ np.transpose(R_a_stand))/(n - 1)

if __name__ == "__main__":
    print("This file is not meant to be executed directly.")