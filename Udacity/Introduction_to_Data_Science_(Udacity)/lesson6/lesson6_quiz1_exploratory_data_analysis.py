import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def entries_histogram(turnstile_weather):
    '''
    Before we perform any analysis, it might be useful to take a
    look at the data we're hoping to analyze. More specifically, let's 
    examine the hourly entries in our NYC subway data and determine what
    distribution the data follows. This data is stored in a dataframe
    called turnstile_weather under the ['ENTRIESn_hourly'] column.
    
    Let's plot two histograms on the same axes to show hourly
    entries when raining vs. when not raining. Here's an example on how
    to plot histograms with pandas and matplotlib:
    turnstile_weather['column_to_graph'].hist()
    
    Your histogram may look similar to bar graph in the instructor notes below.
    
    You can read a bit about using matplotlib and pandas to plot histograms here:
    http://pandas.pydata.org/pandas-docs/stable/visualization.html#histograms
    
    You can see the information contained within the turnstile weather data here:
    https://s3.amazonaws.com/content.udacity-data.com/courses/ud359/turnstile_data_master_with_weather.csv
    '''
    
    plt.figure()
    ent_rain = turnstile_weather['ENTRIESn_hourly'][turnstile_weather.rain ==1] 
    ent_norain = turnstile_weather['ENTRIESn_hourly'][turnstile_weather.rain ==0]
    
    
    # the histogram of the data
    plt.xlabel('ENTRIESn_hourly')
    plt.ylabel('Frequency')
    plt.title('Histogram of ENTRIESn_hourly')
    plt.axis([0, 6000, 0, 45000])
    plt.hist([ent_norain], bins = 180, color=['blue'], alpha=1, label = "No rain")
    plt.hist([ent_rain], bins = 180, color=['green'], alpha=1, label = "Rain")
    plt.legend()
    return plt

if __name__ == "__main__":
    filename = '../data/turnstile_data_master_with_weather.csv'
    df = pd.read_csv(filename, parse_dates = True)
    entries_histogram(df)