#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author      : Bhishan Poudel; Physics PhD Student, Ohio University
# Date        : May 09, 2017
# Last update : 
#

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mp


# Get the data Position_Salaries.csv
dataset = pd.read_csv('Position_Salaries.csv')
d = pd.DataFrame(dataset)
#print(d)
# It has 3 columns Position, Level, and Salary with 10 positions.

#print(d['Position']) # [].values gives 1d array

# First visualize the data
def plot_data():
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()

    ax1.ticklabel_format(style='sci', axis='y')
    ax1.scatter(d['Level'],d['Salary']/1000,color='red')
    xlabels = d['Level']
    ax1.set_xlim(ax1.get_xlim())
    ax1.set_xticks(xlabels)
    ax1.set_xticklabels(xlabels)

    ylabels = np.arange(0, 1e3+1, step=100,dtype='int')
    ax1.set_ylim(ax1.get_ylim())
    ax1.set_yticks(ylabels)
    ax1.set_yticklabels(ylabels)

    ax1.set_xlabel(r"Position")
    ax1.set_ylabel(r"Salary (Thousands)")
    ax1.yaxis.grid(True)
    ax1.xaxis.grid(True)

    x2labels = np.arange(1,11,step=1)

    ax1.set_xlim(ax1.get_xlim())
    ax1.set_xticks(x2labels)
    ax1.set_xticklabels(x2labels)

    ax2.set_xlabel(r"Level")
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(x2labels)
    ax2.set_xticklabels(x2labels)


    x1labels = d['Position']
    ax1.set_xticklabels(x1labels,rotation='vertical')
    ax1.xaxis.grid(True)
    ax1.yaxis.grid(True)


    # show the plot
    plt.subplots_adjust(bottom=0.35)
    plt.subplots_adjust(left=0.2)
    plt.show()


# Now plot the dataset
plot_data()

# 
