import math
import RNA
import random
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
#from scipy.optimize import curve_fit
from os.path import exists
from scipy.integrate import quad
from scipy.stats import ks_2samp

# SEED = 827865
SEED = None
NUMBER_OF_SEQUENCES = 10
LENGTH = 100
SAMPLE_SIZE = 100000
bases =["A","G","C","U"]

def generate_output(sequence, sample_size):
    md = RNA.md() #metadata
    md.uniq_ML = 1
    fc = RNA.fold_compound(sequence, md) #args: (sequence, metadata)
    _, mfe = fc.mfe()
    fc.exp_params_rescale(mfe)
    fc.pf() #partition function precomputes some things
    return(fc.pbacktrack(sample_size)) #args: (number of samples, default backtracking method)

def create_random_sequence(seed, length):
    if seed != None:
        random.seed(seed)
    # print(''.join(random.choices(bases, k=length)))
    return ''.join(random.choices(bases, k=length))

#CODE FOR PLOTTING DISTRIBUTIONS MOVED TO THIS FILE
#ToDo: CLEANUP
# Holds the hypothesised probability distribution
x_graph = []
y_graph = []

def version_2(df):
    global x_avg
    global y_avg
    plt.title("Frequencies of sequences by rank")
    plt.ylabel("Frequency of structure")
    plt.xlabel("Frequency rank")
    avg_df = pd.DataFrame()
    for i in range(NUMBER_OF_SEQUENCES):
        row = df.iloc[i]
        freq_df = row.value_counts().rename_axis('unique_values').reset_index(name='counts')
        freq_df['rank'] = np.arange(1,freq_df.shape[0]+1)
        freq_df['frequency'] = freq_df['counts']/SAMPLE_SIZE
        # print(freq_df)
        x_data = freq_df[['rank']].to_numpy()
        y_data = freq_df[['frequency']].to_numpy()
        # print(x_data,y_data)
        # freq_df.plot.bar()
        # plt.loglog(x_data,y_data)
        plt.yscale("log")
        plt.plot(x_data,y_data,'grey')
        # print(kl_divergence(y_data,y))
        avg_df[i] = freq_df['frequency']
    avg_df['mean'] = avg_df.mean(axis=1)
    avg_df['rank'] = np.arange(1,avg_df.shape[0]+1)
    x_avg = avg_df[['rank']].to_numpy()
    y_avg = avg_df[['mean']].to_numpy()
    plt.plot(x_avg, y_avg,'r')

def record_divergences(df):
    divergences = []
    ks_scores = []
    for i in range(NUMBER_OF_SEQUENCES):
        print("Running: ",i)
        for j in range(0,NUMBER_OF_SEQUENCES):
            if i!=j:
                x_p, y_p = arrange_data(df.iloc[i])
                x_q, y_q = arrange_data(df.iloc[j])
                # print(kl_divergence(y_p, y_q))
                divergences.append(kl_divergence(y_p, y_q))
                # ks_scores.append(ks_2samp(y_p, y_q))
    return np.mean(divergences), np.mean(ks_scores)

def record_divergences_against_average(df):
    divergences_avg = []
    ks_scores_avg = []
    for i in range(NUMBER_OF_SEQUENCES):
        x_p, y_p = arrange_data(df.iloc[i])
        # print(kl_divergence(y_p, y_avg))
        divergences_avg.append(kl_divergence(y_p, y_avg))
        # ks_scores_avg.append(ks_2samp(y_p, y_avg))
    return np.mean(divergences_avg), np.mean(ks_scores_avg)

def record_divergences_against_function(df):
    divergences_function = []
    for i in range(NUMBER_OF_SEQUENCES):
        x_p, y_p = arrange_data(df.iloc[i])
        divergences_function.append(kl_divergence(y_p, y_graph))
    return np.mean(divergences_function)

def arrange_data(df_row):
    freq_df = df_row.value_counts().rename_axis('unique_values').reset_index(name='counts')
    freq_df['rank'] = np.arange(1,freq_df.shape[0]+1)
    freq_df['counts'] = freq_df['counts']/SAMPLE_SIZE
    x_data = freq_df[['rank']].to_numpy()
    y_data = freq_df[['counts']].to_numpy()
    return x_data, y_data



def write_plot_to_file():
    file_suffix = 1
    file_exists = True
    while file_exists:
        filename = str(NUMBER_OF_SEQUENCES) + "-" + str(LENGTH) + "-" + str(SAMPLE_SIZE) + "-" + str(file_suffix) + ".png"
        file_exists = exists(filename)
        file_suffix += 1
    print(filename)
    plt.savefig(filename)

def plot_inverse_distribution(a,b):
    f = lambda x:a/(x**b)
    k, err = quad(f,1,LENGTH*10)
    for i in range(1,LENGTH*10):
        x_graph.append((i))
        # y_graph.append(a/(i**b))
        # y_graph.append(a/(math.e**i))
        k, err = quad(f,i,i+1)
        y_graph.append(k)
    plt.yscale("log")
    plt.plot(x_graph, y_graph, 'b')
    # plt.plot(x,y)
    
def kl_divergence(p, q):
	return sum(p[i] * np.log2(p[i]/q[i]) for i in range(min(len(p),len(q))))

def test_func(x, a, b):
    return a/(x**b)



def main():
    open("csvoutput.csv","w").close() #initialises output file
    fw = open("csvoutput.csv","w")
    for _ in range(NUMBER_OF_SEQUENCES):
        seq = create_random_sequence(SEED, LENGTH)
        print(seq)
        # print(str(generate_output(seq, SAMPLE_SIZE))[1:-1])
        fw.writelines(str(generate_output(seq, SAMPLE_SIZE))[1:-1])
        fw.writelines('\n')
    fw.close()

    df = pd.read_csv("csvoutput.csv", header=None)
    plot_inverse_distribution(1,4)
    version_2(df)
    # print("AVERAGE KL DIVERGENCE BETWEEN ALL PAIRS: ",record_divergences(df)[0])
    print("AVERAGE KL DIVERGENCE WITH AVERAGE GRAPH: ",record_divergences_against_average(df)[0])
    print("AVERAGE KL DIVERGENCE WITH FUNCTION: ",record_divergences_against_function(df))

    # print("AVERAGE KS DIVERGENCE BETWEEN ALL PAIRS: ",record_divergences(df)[1])
    # print("AVERAGE KS DIVERGENCE WITH AVERAGE GRAPH: ",record_divergences_against_average(df)[1])

    write_plot_to_file()
    plt.show()

main()
