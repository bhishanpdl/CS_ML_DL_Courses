import numpy as np
import scipy.stats
import pandas as pd

def compare_averages(filename):
    """
    Performs a t-test on two sets of baseball data (left-handed and right-handed hitters).

    You will be given a csv file that has three columns.  A player's
    name, handedness (L for lefthanded or R for righthanded) and their
    career batting average (called 'avg'). You can look at the csv
    file by downloading the baseball_stats file from Downloadables below. 
    
    Write a function that will read that the csv file into a pandas data frame,
    and run Welch's t-test on the two cohorts defined by handedness.
    
    One cohort should be a data frame of right-handed batters. And the other
    cohort should be a data frame of left-handed batters.
    
    We have included the scipy.stats library to help you write
    or implement Welch's t-test:
    http://docs.scipy.org/doc/scipy/reference/stats.html
    
    With a significance level of 95%, if there is no difference
    between the two cohorts, return a tuple consisting of
    True, and then the tuple returned by scipy.stats.ttest.  
    
    If there is a difference, return a tuple consisting of
    False, and then the tuple returned by scipy.stats.ttest.
    
    For example, the tuple that you return may look like:
    (True, (9.93570222, 0.000023))
    """
    df = pd.read_csv(filename)
    r_avg = df[df['handedness']=='R'].avg
    l_avg = df[df['handedness']=='L'].avg
    p_crit = 0.05
    t,p = scipy.stats.ttest_ind(r_avg,l_avg,equal_var=False)
    if p > p_crit:
      ans = (True, (t,p))
      print(ans)
    
    
    else:
      ans = (False, (t,p)) 
      print(ans)
      
    return ans

def main():
    """Run main function."""
    filename = '../data/baseball_stats.csv'
    compare_averages(filename)

if __name__ == "__main__":
    main()