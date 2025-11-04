""" 
Takes averaged fitness data over multiple runs and plots it

usage: python plotter_avg.py < runIDs.txt

runIDs.txt contains a list of runIDs to average over, one per line
"""

import csv
import matplotlib.pyplot as plt



if __name__ == "__main__":
    for 