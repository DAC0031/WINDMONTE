# WINDMONTE is licensed under GNU GPL v3, see COPYING.txt

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from tqdm import tqdm
import sys
import os
import csv
import DREs
import pickle

"""
Variable classes are used to help make the program more robust.  By defining the classes for data input, each input will be defined in the way the MCM 
simulation and plotting code expects hopefully reducing issues in the main portion of the code due to differences in input definitions.  The inputs defined are:

u_source: an elemental error source definition including distribution parameters
Var: An input variable to the DREs, typically a measurement
DataPoint: A data point consisting of multiple measurements (Var) or input variables pertaining to that data point
Run:  A One-Factor-At-a-Time (OFAT) test run, consisting of multiple data points and typically sweeping one variable.
"""

# Function to check input data and ensure it is in the correct form
def check_list_of_dicts(data):
    # Check if data is a list
    if not isinstance(data, list):
        raise Exception("data is not a list.  Ensure run data is a list of data points, each data point a dictionary in measurement:value form")

    # Check if each element of data is a dictionary
    for elem in data:
        if not isinstance(elem, dict):
            raise Exception("data contains non-dictionary elements.  Ensure run data is a list of data points, each data point a dictionary in measurement:value form")
        
        # Check if each dictionary entry has a key with a numerical value
        for key, value in elem.items():
            if not isinstance(value, (int, float)):
                raise Exception(f"Value of key '{key}' is not numerical.  Ensure run data is a list of data points, each data point a dictionary in measurement:value form")
    
    return True

class U_systematic:
    def __init__(self):
        self.sources = []  # list to contain all systematic error sources
        self.number = 0  # list to contain all random error sources

    def add_error_source(self, **args):
        try: 
            self.sources.append(u_source(**args))
            self.number = len(self.sources)
        except:
            raise Exception('Error in argurments for error source. Arguments: {}'.format([arg for arg in args]))
        
    def updateLHS(self, LHS): # take LHS input data in a k by M array (k=# error sources, M=# MCM iterations) and update each of the error sources with their percentiles for each MCM iteration
        for k in range(self.number):
            self.sources[k].LHS = LHS[:,k]
        
class U_random:
    def __init__(self):
        self.sources = []  # list to contain all systematic error sources
        self.number = 0  # list to contain all random error sources
        self.propagate = True  # Boolean on whether these uncertainties should be propagated by MCM.  Select False if using direct comparison of replicates for random uncertainty assessment

    def add_error_source(self, **args):
        try: 
            self.sources.append(u_source(**args))
            self.number = len(self.sources)
        except:
            raise Exception('Error in argurments for error source. Arguments: {}'.format([arg for arg in args]))
    
    def updateLHS(self, LHS): # take LHS input data in a k by M array (k=# error sources, M=# MCM iterations) and update each of the error sources with their percentiles for each MCM iteration
        if not LHS:
            for k in range(self.number):
                self.sources[k].LHS = False
        else:
            for k in range(self.number):
                self.sources[k].LHS = LHS[:,k]

# Define class for defining elemental uncertainties.  These are the uncertainties of elemental error sources (beta, epsilon), not input variables to the DREs
class u_source:   
    def __init__(self, measurements, distribution, params, percent_scale=False,scaling_factor=None, source='', units='',notes=None):
        self.measurements = measurements # define the DRE input variables that this elemental error source effects in list format (i.e. ['measurement1','measurement2',etc.]).  List entries must match the dictionary keys for each measurement the error source applies to.
        self.distribution = distribution
        self.params = params
        self.percent_scale = percent_scale  # Boolean on whether or not to scale PDF by nominal value
        self.scaling_factor = scaling_factor
        self.distribution_object = self._create_distribution_object()
        self.sim_error = None  # Initialize sim_error attribute for saving simulated error values for each MCM iteration
        self.source = source  # label the source of the elemental error
        self.UPCflag = True  # Flag to turn off contributions from this error source when conducting UPC calculations (True = contributing, False = not contributing).
        self.units = units
        self.notes = notes  # any pertinent notes for the error source, such as calibration date, calibration range, documentation, etc.
        self.LHS = None  # container for Latin Hypercube Sampling percentiles (0-1) pertaining to this error source for each MCM trial.
    
    def _create_distribution_object(self):
        distribution_map = {
            'norm': stats.norm,
            'uniform': stats.uniform,
            'triang': stats.triang,
            'beta': stats.beta,
            # Add more distributions as needed
        }
        distribution_class = distribution_map.get(self.distribution)
        if distribution_class is None:
            raise ValueError(f"Unsupported distribution: {self.distribution}")
        
        if self.scaling_factor is not None:
            scaled_params = [param * self.scaling_factor for param in self.params]
        else:
            scaled_params = self.params

        return distribution_class(*scaled_params)
    
    def sample(self, M, scale=1):
        self.scaling_factor = scale  # update scaling factor

        distribution_map = {
            'norm': stats.norm,
            'uniform': stats.uniform,
            'triang': stats.triang,
            'beta': stats.beta,
            # Add more distributions as needed
        }
        distribution_class = distribution_map.get(self.distribution)
        if distribution_class is None:
            raise ValueError(f"Unsupported distribution: {self.distribution}")
        
        if self.percent_scale:
            if self.scaling_factor != 0:
                scaled_params = [param * np.abs(self.scaling_factor) for param in self.params]
            else:  # if scaling factor equals zero (e.g. at a zero point), replace with 0.00001
                scaled_params = [param * np.abs(0.00001) for param in self.params]
        else:
            scaled_params = self.params

        if isinstance(self.LHS, bool) and not self.LHS:  # if self.LHS is False, do a pseudo-random draw (for random uncertainties) any of the LHS values are numerical, do the LHS draw
            sampled_value = distribution_class(*scaled_params).ppf(stats.uniform.rvs(size=M))  
        else: # otherwise, use LHS percentiles
            sampled_value = distribution_class(*scaled_params).ppf(self.LHS)  
            
        return sampled_value
    
    def __str__(self):
        return f"Variable with {self.distribution} distribution, params={self.params}, scaling_factor={self.scaling_factor}, notes:{self.notes}"

# Test Run class is a subclass of list that adds additional behaviors to plot results from MCM.  TestRun is a series of DataPoints, usually with zero points as the first and last points and doing an OFAT sweep.  
class TestRun(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def trialupdate(self, sim_data_out, sim_data_in):
        for j in range(len(self)):
            for key in self[j].keys():
                if key in sim_data_out[j].keys():
                    self[j][key].sim.append(sim_data_out[j][key])
                else:
                    self[j][key].sim.append(sim_data_in[j][key])

    def trialclear(self, sim_data_out, sim_data_in):  # clear trail data to run iterations for UPC calculations
        for j in range(len(self)):
            for key in self[j].keys():
                if key in sim_data_out[j].keys():
                    self[j][key].sim = []
                else:
                    self[j][key].sim = []

    def simupdate(self):
        for j in range(len(self)):
            for key in self[j].keys():    
                self[j][key].simupdate()      
            
    def plot_errorbars(self, xvar, y_list, y_labels=None, ncols=3):
        """
        Generate a multifigure plot with different y-axis VOIs and error bars based on u_low to u_high interval.

        Parameters:
            xvar (string): key defining the x-axis.
            y_list (list of strings): List of keys for the y-axis.
            y_labels (list of str, optional): List of labels for the y-axes.
            ncols (int, optional): Number of columns for subplots. Default is 2.

        Returns:
            fig, axes (matplotlib.figure.Figure, numpy.ndarray): Figure and axes objects.
        """
        # Determine the number of subplots needed
        num_plots = len(y_list)
        nrows = -(-num_plots // ncols)  # Ceiling division to determine the number of rows
        plotpoints = range(2,len(self)-1)

        # Create the subplots
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows))
        if num_plots > 1: axes = axes.flatten()  # Flatten the axes array for easier indexing
        else: axes = [axes]

        # Iterate over the y-axis parameters and plot each one
        for i, (yvar, ax) in enumerate(zip(y_list, axes)):
            x = [self[j][xvar].nom for j in plotpoints]
            y = [self[j][yvar].nom for j in plotpoints]
            Ux = ([np.abs(self[j][xvar].u_low) for j in plotpoints],[np.abs(self[j][xvar].u_high) for j in plotpoints])
            Uy = ([np.abs(self[j][yvar].u_low) for j in plotpoints],[np.abs(self[j][yvar].u_high) for j in plotpoints])
            
            ax.errorbar(x,y,Uy,Ux,'b.')
            ax.set_xlabel(xvar)
            ax.set_ylabel(yvar)


        # Hide any unused subplots
        for j in range(num_plots, nrows * ncols):
            axes[j].axis('off')

        # Adjust layout to prevent overlap of subplots
        plt.tight_layout()

        return fig, axes
    
    def plot_U_VOI(self, xvar, y_list, y_labels=None, ncols=3):
        """
        Generate a multifigure plot with different y-axis parameters.

        Parameters:
            xvar (string): key defining the x-axis.
            y_list (list of strings): List of keys for the y-axis.
            y_labels (list of str, optional): List of labels for the y-axes.
            ncols (int, optional): Number of columns for subplots. Default is 2.

        Returns:
            fig, axes (matplotlib.figure.Figure, numpy.ndarray): Figure and axes objects.
        """
        # Determine the number of subplots needed
        num_plots = len(y_list)
        nrows = -(-num_plots // ncols)  # Ceiling division to determine the number of rows
        plotpoints = range(2,len(self)-1)

        # Create the subplots
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows))
        if num_plots > 1: axes = axes.flatten()  # Flatten the axes array for easier indexing
        else: axes = [axes]

        # Iterate over the y-axis parameters and plot each one
        for i, (yvar, ax) in enumerate(zip(y_list, axes)):
            x = [self[j][xvar].nom for j in plotpoints]
            Uy = [self[j][yvar].U for j in plotpoints]
            
            ax.bar(x,Uy)
            ax.set_xlabel(xvar)
            ax.set_ylabel(yvar)

        # Hide any unused subplots
        for j in range(num_plots, nrows * ncols):
            axes[j].axis('off')

        # Adjust layout to prevent overlap of subplots
        plt.tight_layout()

        return fig, axes
    
    def boxplot(self, xvar, y_list, y_labels=None, ncols=3):
        # Determine the number of subplots needed
        num_plots = len(y_list)
        nrows = -(-num_plots // ncols)  # Ceiling division to determine the number of rows
        plotpoints = range(2,len(self)-1)

        # Create the subplots
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows))
        if num_plots > 1: axes = axes.flatten()  # Flatten the axes array for easier indexing
        else: axes = [axes]

        # Iterate over the y-axis parameters and plot each one
        for i, (yvar, ax) in enumerate(zip(y_list, axes)):
            x = [self[j][xvar].nom for j in plotpoints]
            y = [self[j][yvar].sim for j in plotpoints]
            boxplot_width = 0.01 * (max(x) - min(x))  # Width of the boxplot
            mean_marker_size = 3  # Adjust the multiplier as needed
            meanpointprops = dict(marker='s', markeredgecolor='black', markerfacecolor='firebrick', markersize=mean_marker_size)
            ax.boxplot(y,positions=x,usermedians=[self[j][yvar].nom for j in plotpoints],showmeans=True,manage_ticks=False,meanprops=meanpointprops,notch=True,bootstrap=2000,whis=((5,95)),sym='',widths=boxplot_width)
            ax.set_xlabel(xvar)
            #ax.set_xticklabels([f'{pos:.2g}' for pos in x])
            ax.set_ylabel(yvar)

        # Hide any unused subplots
        for j in range(num_plots, nrows * ncols):
            axes[j].axis('off')

        # Adjust layout to prevent overlap of subplots
        plt.tight_layout()

        return fig, axes
    
    def plot_UPCs(self, xvar, y_list, y_labels=None, ncols=3):
        # Determine the number of subplots needed
        num_plots = len(y_list)
        nrows = -(-num_plots // ncols)  # Ceiling division to determine the number of rows
        plotpoints = range(2,len(self)-1)

        # Create the subplots
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows))
        if num_plots > 1: axes = axes.flatten()  # Flatten the axes array for easier indexing
        else: axes = [axes]

        # Sort the UPCs by the largest average across all yvars, so that it shows the largest contributors first and in the same order for every plot
        UPC_averages = {}
        for el in self[1][y_list[0]].UPCs.keys():  # for each elemental error source, find the average of a list containing all UPCs across data points and y_list VOIs
            UPC_averages[el] = np.mean(np.array([[self[i][yvar].UPCs[el] for i in plotpoints] for yvar in y_list]).flatten())
        UPC_sortorder = sorted(UPC_averages.items(),key=lambda x:x[1], reverse=True)
        UPC_orderedkeys = [UPC_sortorder[i][0] for i in range(len(UPC_sortorder))]

        # Define a list of colors and cross-hatch styles
        hatch_styles = ['', '////', 'xxxx', '||||', '----']
        #colors = ['red', 'green', 'blue', 'orange', 'purple', 'yellow', 'cyan', 'magenta','black']  # if you want to define colors directly
        colors = [list(plt.cm.Set1(ix)) for ix in np.linspace(0,1,9)]  # if you want to use the 'Set1' matplotlib colormap
        
        # Generate all unique combinations of colors and hatch styles
        style_combos = [(color, hatch) for hatch in hatch_styles for color in colors]
        
        handles, labels = None, None

        # Iterate over the y-axis parameters and plot each one
        for i, (yvar, ax) in enumerate(zip(y_list, axes)):
            x = [self[j][xvar].nom for j in plotpoints]
            bottom = np.zeros(len(x))
            style_ix = 0
            for source in UPC_orderedkeys:
                ax.bar(x, [self[point][yvar].UPCs[source] for point in plotpoints], width=0.9, color=style_combos[style_ix][0], hatch=style_combos[style_ix][1], label=source, bottom=bottom)
                bottom += [self[point][yvar].UPCs[source] for point in plotpoints]
                style_ix += 1

            ax.set_xlabel(xvar)
            ax.set_ylabel(yvar + ' UPCs')
            ax.set_title(yvar)
            if handles is None: handles, labels = ax.get_legend_handles_labels()

        
        
        fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.subplots_adjust(right=0.85)
            
        # Hide any unused subplots
        for j in range(num_plots, nrows * ncols):
            axes[j].axis('off')

        # Adjust layout to prevent overlap of subplots
        #plt.tight_layout()

        return fig, axes
    
    def plot_U_and_UPCs(self, xvar, y_list, y_labels=None, ncols=3):
        # Determine the number of subplots needed
        num_plots = len(y_list)
        nrows = -(-num_plots // ncols)  # Ceiling division to determine the number of rows
        plotpoints = range(2,len(self)-1)

        # Create the subplots
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows))
        if num_plots > 1: axes = axes.flatten()  # Flatten the axes array for easier indexing
        else: axes = [axes]

        # Sort the UPCs by the largest average across all yvars, so that it shows the largest contributors first and in the same order for every plot
        UPC_averages = {}
        for el in self[1][y_list[0]].UPCs.keys():  # for each elemental error source, find the average of a list containing all UPCs across data points and y_list VOIs
            UPC_averages[el] = np.mean(np.array([[self[i][yvar].UPCs[el] for i in plotpoints] for yvar in y_list]).flatten())
        UPC_sortorder = sorted(UPC_averages.items(),key=lambda x:x[1], reverse=True)
        UPC_orderedkeys = [UPC_sortorder[i][0] for i in range(len(UPC_sortorder))]

        # Define a list of colors and cross-hatch styles
        hatch_styles = ['', '////', 'xxxx', '||||', '----']
        #colors = ['red', 'green', 'blue', 'orange', 'purple', 'yellow', 'cyan', 'magenta','black']  # if you want to define colors directly
        colors = [list(plt.cm.Set1(ix)) for ix in np.linspace(0,1,9)]  # if you want to use the 'Set1' matplotlib colormap
        
        # Generate all unique combinations of colors and hatch styles
        style_combos = [(color, hatch) for hatch in hatch_styles for color in colors]
        
        handles, labels = None, None

        # Iterate over the y-axis parameters and plot each one
        for i, (yvar, ax) in enumerate(zip(y_list, axes)):
            x = [self[j][xvar].nom for j in plotpoints]
            bottom = np.zeros(len(x))
            style_ix = 0
            for source in UPC_orderedkeys:
                Uy_UPCs = np.multiply([self[point][yvar].UPCs[source] for point in plotpoints],np.divide([self[point][yvar].U for point in plotpoints],100))
                ax.bar(x, Uy_UPCs, width=0.9, color=style_combos[style_ix][0], hatch=style_combos[style_ix][1], label=source, bottom=bottom)
                bottom += Uy_UPCs
                style_ix += 1

            ax.set_xlabel(xvar)
            ax.set_ylabel(yvar + ' Uncertainty')
            ax.set_title(yvar)
            if handles is None: handles, labels = ax.get_legend_handles_labels()

        
        
        fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.subplots_adjust(right=0.85)
            
        # Hide any unused subplots
        for j in range(num_plots, nrows * ncols):
            axes[j].axis('off')

        # Adjust layout to prevent overlap of subplots
        #plt.tight_layout()

        return fig, axes

# DataPoints include all measurements, constants, and conditions
class DataPoint(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

# Define "Var" (Variable) data class.  This class is used to define each input variable into the MCM simulation and DREs, and each output variable
class VOI:
    def __init__(self,key,nom,units='', confidence=0.95):
        self.key = key
        self.nom = nom # nominal value of datapoint
        self.sim = [] # values of self with simulated systematic error
        self.U = [] # combined expanded uncertainty interval
        self.r_low = None # value for VOI at lower limit of confidence interval
        self.r_high = None # value for VOI at upper limit of confidence interval
        self.u_low = None # lower limit of uncertainty interval for given confidence (from nominal value)
        self.u_high = None # upper limit of uncertainty interval for given confidence (from nominal value)
        self.UPCs = {}  # container to store UPCs if calculated
        self.confidence = confidence  # confidence level for given uncertainties
        self.units = units # specify units for variable

    def plot_histogram(self):
        plt.figure()
        plt.hist(self.sim,bins='auto', density=True, alpha=0.7, color='blue', label='Histogram')
        plt.axvline(self.nom, color='black', linestyle='dashed', linewidth=2, label=f'Nominal: {self.nom:.3g}')
        plt.axvline(np.average(self.sim), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {np.average(self.sim):.3g}')
        plt.axvline(self.r_low, color='green', linestyle='dashed', linewidth=2, label=f'95% CI: [{self.r_low:.3g}, {self.r_high:.3g}]')
        plt.axvline(self.r_high, color='green', linestyle='dashed', linewidth=2)

        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Histogram with Mean and 95% Confidence Interval')
        plt.legend()
        plt.show()

    def plot_QQ(self,distribution='norm', line=True, ax=None, confidence=95,alpha = 0.05):
        
        """
        Create a Q-Q plot for comparing data to a theoretical distribution.

        Parameters:
        - data: array-like, the sample data to be plotted.
        - distribution: str or scipy.stats distribution object, the theoretical distribution to compare against.
                        Default is 'norm' for the normal (Gaussian) distribution.
        - line: bool, whether to include the identity line (45-degree line) on the plot. Default is True.
        - ax: figure axis to be used if plotting in an already established figure.  Default is make a new figure.
        - confidence: Confidence level in percent.  Default is 95%
        - alpha: significance level, threshold for concluding the hypothesis test.

        Returns:
        - None (displays the Q-Q plot).
        """
        # Sort and normalize data
        sorted_data = np.divide(np.subtract(np.sort(self.sim),np.average(self.sim)),np.std(self.sim))
        
        # Generate theoretical quantiles based on the specified distribution
        if isinstance(distribution, str):
            dist = getattr(stats, distribution)
        else:
            dist = distribution
        n = len(sorted_data)
        theoretical_quantiles = dist.ppf(np.arange(1, n + 1) / (n + 1))
        
        # Create Q-Q plot
        plt.figure()
        plt.scatter(theoretical_quantiles, sorted_data, color='blue', alpha=0.6)
        plt.title('Q-Q Plot for VOI {}, theoretical distribution: {}'.format(self.key,distribution))
        plt.xlabel('Theoretical Quantiles')
        plt.ylabel('MCM Data Quantiles')

        # Create Q-Q plot
        plt.scatter(theoretical_quantiles, sorted_data, color='blue', alpha=0.6)

        # Add identity line
        if line:
            min_val = min(np.min(theoretical_quantiles), np.min(sorted_data))
            max_val = max(np.max(theoretical_quantiles), np.max(sorted_data))
            plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')

        plt.grid(True)

        # Plot 95% confidence limits
        Ulow_ix = int(np.floor(len(sorted_data)*(100-confidence)/(2*100)))
        Uhigh_ix = int(np.ceil(len(sorted_data)*(1+confidence/100)/2))
        plt.plot([theoretical_quantiles[Ulow_ix], theoretical_quantiles[Uhigh_ix]],[sorted_data[Ulow_ix], sorted_data[Uhigh_ix]],markerfacecolor='y',markeredgecolor='k',marker='d')
        # Calculate and display Shapiro-Wilk statistic and p-value
        
        # Calculate observed frequencies
        statistic, p_value = stats.shapiro(self.sim)

        text = f'Shapiro-Wilk Statistic: {statistic:.5f}\n'
        text += f'p-value: {p_value:.4f}\n'
        if p_value >= alpha:  
            text += 'Fail to reject null hypothesis\n'
            text += 'Data follows normal distribution'
        else: 
            text += 'Reject Null Hypothesis\n'
            text += 'Data does not follow normal distribution'
        plt.text(0.95, 0.05, text, transform=plt.gca().transAxes, ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.5))
        interval = np.sort(self.sim)[Uhigh_ix]-np.average(self.sim)
        text = '95% Interval: [+{:.3g},-{:.3g}]\n'.format(np.sort(self.sim)[Uhigh_ix]-np.average(self.sim),np.average(self.sim)-np.sort(self.sim)[Ulow_ix])
        text += r'$2\sigma: \pm$ {:.3g}'.format(np.std(self.sim)*2)
        plt.text(0.05, 0.95, text, transform=plt.gca().transAxes, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.5))
        plt.show()

    def plot_UPCs(self):
        filtered_data = {k: v for k, v in self.UPCs.items() if v >= 1e-10}  # Include this line if desiring to filter out anything contributing less than e-10 (essentially numerical error)
        sorted_data = dict(sorted(filtered_data.items(), key=lambda item: item[1], reverse=False))

        # Extract keys and sorted values from the dictionary
        keys = list(sorted_data.keys())
        values = list(sorted_data.values())

        # Create a horizontal bar plot
        plt.figure()
        plt.barh(keys, values)

        # Set the x-axis scale to log
        plt.xscale('log')
        plt.xlim(1e-10, 100)

        # Set labels and title
        plt.xlabel('UPCs')
        plt.ylabel('Error Sources')
        plt.title('Horizontal Bar Plot of UPCs for VOI {}'.format(self.key))
        ax = plt.gca()
        ax.grid(True, which='both', axis='x', linestyle=':', color='gray')
        plt.show()


    def simupdate(self):
        # Calculate the expanded uncertainty interval (assuming Gaussian distribution)
        K = stats.norm.ppf(0.5 + self.confidence / 2) - stats.norm.ppf(0.5 - self.confidence / 2)
        self.U = np.std(self.sim)*K
        # Calculate the lower and upper values based on the given confidence interval
        lower_percentile = 100 * (1 - self.confidence) / 2
        upper_percentile = 100 * (1 + self.confidence) / 2
        self.r_low = np.percentile(self.sim, lower_percentile)
        self.r_high = np.percentile(self.sim, upper_percentile)
        # Calculate the lower and upper limits of the confidence interval
        self.u_low = self.r_low-self.nom
        self.u_high = self.r_high-self.nom

# This function generates seed data with simulated error using the inputs of the nominal data, error sources, and a trial number "i"
def simulate_error(data,M,U_systematic,U_random):

    # Build sim_data structure of seed data for the DREs (list of M elements, each being a version of "data" but with simulated error added)
    sim_data = []
    for i in range(M):
        sim_data.append([])
        for point in range(len(data)):
            sim_data[i].append({})

    for point in range(len(data)):
        for measurement, value in data[point].items():
            # if measurement has no error source, copy nominal value
            for i in range(M): sim_data[i][point][measurement] = value
            # for each measurement with an error source, determine the simulated error from each contributing error source.  Generate M simulated errors, and add those to the sim_data seed data
            for beta_i in U_systematic.sources:
                if measurement in beta_i.measurements and beta_i.UPCflag:
                    tmp_error = beta_i.sample(M,value)
                    for i in range(M): sim_data[i][point][measurement] += tmp_error[i]
            for epsilon_i in U_random.sources:
                if measurement in epsilon_i.measurements and epsilon_i.UPCflag:  # only have one PDF for random error per measurement
                    tmp_error = epsilon_i.sample(M,value)
                    for i in range(M): sim_data[i][point][measurement] += tmp_error[i]  # set LHS=False to use a pseudo-random draw 

            # simulated error is added to the nominal value of the measurement as described in Figure 6-4.1-1 of ASME PTC 19.1-2018

        
    return sim_data

def replicate_uncert():
    pass

def load_LSWT_rawdata(Test,Run):
    sys.argv.append('Test:{}'.format(Test))
    sys.argv.append(str(Run))
    # Set directory and check for file presence
    if os.path.dirname(__file__):
        os.chdir(os.path.dirname(__file__)) #Changes cwd to where Decompose lives

    if not os.path.exists(os.getcwd()+"/Raw Data, Processed"):
        os.makedirs(os.getcwd()+"/Raw Data, Processed") # Create folder for raw...txt files if it doesn't exist

    filepresent = 0
    RDPcontents = os.listdir(os.getcwd()+"/Raw Data, Processed")
    if len(RDPcontents)==0:
        pass
    elif "DATA_RAW" in RDPcontents:
        filepresent=1
        if os.path.exists(os.getcwd()+"/Raw Data, Processed/DATA_RAW/raw-\d{4}-0001.txt"):
            filepresent=1
            pass
        
    if filepresent == 0:
        a = 1
        while a > 0:
            answer = str(input("\nNo files exist in Raw Data, Processed. Have you checked tconst.py? \nPlease type 'yes' or 'no'\n"))
            if answer in ['yes','Yes','YES','y','Y','']: a = a-2
            elif answer in ['no','No','NO','n','N']:
                try: os.system("open "+"Common/tconst.py")
                except:	os.system("start "+"Common/tconst.py")
                a = a-2
                print("\nVerify tconst.py before continuing.\n")
                quit()
            else: pass


    #Read/Input data
    data,testinfo = inputs.PARSER()

    for point in data:
        keep_keys = ['Point','Temp', 'Qact', 'Qset', 'Ptot', 'Pstat', 'Baro', 'Theta', 'Psi', 'Phi', 'Gamma', 'Beta', 'Rho', 'Uact', 'a', 'Mach', 'Mu', 'Re', 'TempT', 'NF', 'AF', 'PM', 'RM', 'YM', 'SF']
        del_keys = []
        for key in point.keys():
            if key not in keep_keys:
                del_keys.append(key)
        for key in del_keys:
            del point[key]

    return data, testinfo

def load_csv_data(filename):

    ''' 
    Ensure the .csv file has testinfo as the first two rows (row 1 = keys, row 2 = values), and data starts with keys on row 4 and every row afterwards is the data'''
    
    # Initialize empty lists for the list of dictionaries and single dictionary
    data = []


    with open(filename, mode='r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i == 0:
                keys = row  # Extract keys from the first row
            elif i == 1:
                values = row  # Extract values from the second row
                testinfo = dict(zip(keys, values))  # Create test info dictionary
            elif i == 2:
                continue # skip the blank third row
            elif i == 3:
                keys = row  # Skip the fourth row which contains keys
            else:
                data.append({k: convert_to_datatype(v) for k, v in zip(keys, row)})  # Create dictionaries for data rows

    return data, testinfo

def convert_to_datatype(value):
    try:
        # Try converting to integer
        return int(value)
    except ValueError:
        try:
            # Try converting to float
            return float(value)
        except ValueError:
            # Return as string if conversion fails
            return value

def MCM_sim(data, testinfo, U_systematic, U_random, s_flag, M):
    global VOIs, RunData, data_out
    data_out = DREs.eval(data,testinfo)
    RunData = TestRun()
    VOIs = [el for el in data[1].keys()]+[el for el in data_out[1].keys() if el not in data[1].keys()]# Set VOIs to all outputs and inputs
    for j in range(len(data)):
        RunData.append(DataPoint())
        for el in VOIs:
            RunData[j][el] = VOI(key=el,nom=data_out[j][el])

    # 2.2 Perform Latin Hypercube Sampling (LHS) and update elemental error sources with the LHS percentiles

    LHSsystematic = stats.qmc.LatinHypercube(d=U_systematic.number).random(n=M)  # get LHS percentiles based on number of systematic error sources and number of MCM trials
    U_systematic.updateLHS(LHSsystematic)  # update the LHS percentiles for each systematic error source for simulating error for each measurement and MCM iteration

    if s_flag == 'P': # propagate random uncertainty if applicable
        if True:  # Set LHS to False to use a pseudo-random draw for simulating random error
            U_random.updateLHS(False)
        else:  # if desiring to conduct a LHS sample of random error sources, use the following lines
            LHSrandom = stats.qmc.LatinHypercube(d=U_random.number).random(n=M)  # get LHS percentiles based on number of random error sources and number of MCM trials
            U_random.updateLHS(LHSrandom)  # update the LHS percentiles for each systematic error source for simulating error for each measurement and MCM iteration

    # 2.3 Generate M sets of seed values by simulating error
    print('Generating seed values with simulated error for {} data points'.format(len(data)))
    sim_data_in = simulate_error(data,M,U_systematic,U_random)
    sim_data_out = []

    # 2.4 Run MCM trials
    print('Performing Monte Carlo Simulations, {} trials'.format(M))
    for i in tqdm(range(M)):
        
        # run data reduction equations (DREs)
        sim_data_out.append(DREs.eval(sim_data_in[i],testinfo))
        #sim_data_out.append(r_eval.OBJ4(sim_data_in[i],testinfo,False,1))
        if s_flag == 'DCR': 
            pass # placeholder for adding simulated error from replicate data random uncertainty
        
        RunData.trialupdate(sim_data_out[i],sim_data_in[i])

    # Use simulated data to determing interval limits and standard deviation of simulated result populations
    RunData.simupdate()
    #endregion

    return RunData

def UPCs(RunData, data, testinfo, U_systematic, U_random, s_flag, UPC_M, replicate_data):
    UPC_results = []
    for j in range(len(data)):
        UPC_results.append({})
        for el in VOIs:
            UPC_results[j][el] = {}

    # Update flags for contributing error sources
    for b in U_systematic.sources: b.UPCflag = False # Turn all systematic error sources off
    for s in U_random.sources: s.UPCflag = False # Turn all random error sources off
    
    # Flag each systematic uncertainty separately and run simulation
    source_num = 0
    print('Conducting UPC simulations for systematic error sources')
    for source in tqdm(U_systematic.sources):
        UPCData = TestRun()
        for j in range(len(data)):
            UPCData.append(DataPoint())
            for el in VOIs:
                UPCData[j][el] = VOI(key=el,nom=data_out[j][el])

        # Turn this source UPC flag on
        source.UPCflag = True
        # Run simulation
        LHSsystematic = stats.qmc.LatinHypercube(d=1).random(n=UPC_M)[:,0]  # get LHS percentiles based on number of systematic error sources and number of MCM trials
        source.LHS = LHSsystematic  # update the LHS percentiles for each systematic error source for simulating error for each measurement and MCM iteration
        sim_data_in = simulate_error(data,UPC_M,U_systematic,U_random)
        sim_data_out = []
        for i in range(UPC_M):
            # run data reduction equations (DREs)
            sim_data_out.append(DREs.eval(sim_data_in[i],testinfo))
            UPCData.trialupdate(sim_data_out[i],sim_data_in[i])
        # Use simulated data to determing interval limits and standard deviation of simulated result populations
        UPCData.simupdate()

        # Save in UPC_dict
        source_num += 1
        for point in range(len(data)):
            for el in VOIs:
                try: UPC_results[point][el][source.source] = UPCData[point][el].U
                except: UPC_results[point][el]['b'+str(source_num)] = UPCData[point][el].U

        # turn flag back off
        source.UPCflag = False
        #print('UPCs complete for source:{}'.format(source.source))


    # Flag each random uncertainty separately and run simulation
    source_num = 0
    print('Conducting UPC simulations for random error sources')
    if s_flag == 'P':
        for source in tqdm(U_random.sources):
            UPCData = TestRun()
            for j in range(len(data)):
                UPCData.append(DataPoint())
                for el in VOIs:
                    UPCData[j][el] = VOI(key=el,nom=data_out[j][el])

            # Turn this source UPC flag on
            source.UPCflag = True
            # Run simulation
            LHSrandom = stats.qmc.LatinHypercube(d=1).random(n=UPC_M)[:,0]  # get LHS percentiles based on number of systematic error sources and number of MCM trials
            source.LHS = False  # update the LHS percentiles for each systematic error source for simulating error for each measurement and MCM iteration, or set to False to do random draw
            sim_data_in = simulate_error(data,UPC_M,U_systematic,U_random)
            sim_data_out = []
            for i in range(UPC_M):
                # run data reduction equations (DREs)
                sim_data_out.append(DREs.eval(sim_data_in[i],testinfo))
                UPCData.trialupdate(sim_data_out[i],sim_data_in[i])
            # Use simulated data to determing interval limits and standard deviation of simulated result populations
            UPCData.simupdate()

            # Save in UPC_dict
            source_num += 1
            for point in range(len(data)):
                for el in VOIs:
                    try: UPC_results[point][el][source.source] = UPCData[point][el].U
                    except: UPC_results[point][el]['b'+str(source_num)] = UPCData[point][el].U

            # turn flag back off
            source.UPCflag = False
    elif s_flag == 'DCR':
      
        for el in replicate_data.keys():
            UPCData = TestRun()
            for j in range(len(data)):
                UPCData.append(DataPoint())
                for VOI_i in VOIs:
                    UPCData[j][VOI_i] = VOI(key=VOI_i,nom=data_out[j][VOI_i])

            # Run simulation (could make this faster not running a simulation, but will do that later)
            sim_data_in = simulate_error(data,UPC_M,U_systematic,U_random)
            sim_data_out = []
            for i in range(UPC_M):
                # run data reduction equations (DREs)
                sim_data_out.append(DREs.eval(sim_data_in[i],testinfo))
                UPCData.trialupdate(sim_data_out[i],sim_data_in[i])
            # Use simulated data to determing interval limits and standard deviation of simulated result populations
            UPCData.simupdate()

            UPCData = s_replicates(UPCData,replicate_data)

            # Save in UPC_dict
            for point in range(len(data)):
                UPC_results[point][el]['s_replicate'] = UPCData[point][el].U

    # Save the U_r value from RunData for all error sources included
    for point in range(len(RunData)):
        for el in VOIs:
            tmp_UPC = {}
            Ur2_allsources = np.sum([UPC_results[point][el][source]**2 for source in UPC_results[point][el].keys()])
            if Ur2_allsources == 0:
                pass
            else:
                for source in UPC_results[point][el].keys(): 
                    #print('Point: {}, VOI: {}, Source: {}'.format(point,el,source))
                    #print('Ui = {}, Ur = {}'.format(UPC_results[point][el][source],np.sqrt(Ur2_allsources)))
                    tmp_UPC[source] = 100*UPC_results[point][el][source]**2/Ur2_allsources
            RunData[point][el].UPCs = tmp_UPC


def s_replicates(RunData,replicate_data):

    K = 2 # uncertainty expansion factor

    for el in RunData[1].keys(): # loop through all keys in RunData
        if el in list(replicate_data.keys()): # if the element has random uncertainty defined from direct comparison of replicates
            for point in range(len(RunData)): # update combined uncertainty and add random uncertainty PDF for each datapoint in RunData
                alpha_values, VOI_values = zip(*replicate_data[el]) # zip replicate data into AOA and VOI data
                b = RunData[point][el].U  # propagated systematic uncertainty
                s = K*np.interp(RunData[point]['AlphaC'].nom,np.array(alpha_values),np.array(VOI_values))  # interpolate the random uncertainty at the set condition
                RunData[point][el].U = np.sqrt(b**2+s**2)  # combined random and systematic in quadriture, expand by K, and save as combined uncertainty 

    return RunData

