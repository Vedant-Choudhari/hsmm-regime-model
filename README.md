Hidden Semi-Markov Model (HSMM) for Regime Detection
This repository contains a Python implementation of a Hidden Semi-Markov Model (HSMM) designed for financial regime detection. It uses a Viterbi algorithm adapted for explicit state durations (semi-Markov property) to identify distinct market regimes, such as low-volatility and high-volatility periods, from a time series of asset returns.

The model assumes Gaussian emissions for the returns within each state and Poisson distributions for the duration of each state. Training is performed using a hard-EM (Expectation-Maximization) approach, where the E-step is the semi-Markov Viterbi decoding and the M-step updates the model parameters based on the decoded state path.

Key Features
Explicit State Durations: Avoids the restrictive geometric duration assumption of standard HMMs, allowing for more realistic modeling of regime lengths.

Poisson Durations: Models the length of each regime using a Poisson distribution, with parameters learned directly from the data.

Gaussian Emissions: Assumes that returns within each regime are normally distributed, with the mean and standard deviation learned for each state.

Hard-EM Training: An iterative Viterbi-based training approach that alternates between decoding the most likely state path and re-estimating parameters.

Ready to Use: Can be run directly on a CSV file of prices or with a built-in synthetic data generator for demonstration.

Installation
Clone this repository to your local machine.

It's recommended to create a virtual environment to manage dependencies.

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install the required packages using the requirements.txt file:

pip install -r requirements.txt

Usage
You can run the model in two ways:

On your own price data (CSV file):

Your CSV file must contain a date column and a price column.

python hsmm_regime.py --csv path/to/your_prices.csv --date-col Date --price-col Close --states 2 --dmax 60 --iters 15

With the synthetic demo data generator:

This will generate a sample price series with two distinct volatility regimes and run the model on it.

python hsmm_regime.py --demo

Command-Line Arguments

--csv: Path to your input CSV file.

--date-col: Name of the date column in your CSV. (Default: Date)

--price-col: Name of the closing price column in your CSV. (Default: Close)

--states: The number of regimes (states) to detect. (Default: 2)

--dmax: The maximum duration (in days) allowed for any single regime. (Default: 60)

--iters: The number of hard-EM training iterations. (Default: 15)

--demo: If specified, runs the script on synthetic data.

Outputs
Console Output: The script prints the learned parameters for each state (Gaussian mu and sigma, Poisson lambda) and the transition matrix A to the console after training.

CSV File: A file named hsmm_regimes.csv is saved in the current directory. It contains the original returns along with the detected state and the remaining duration for each time step.

Date: The date of the observation.

Return: The calculated log return.

state: The detected regime (0, 1, ...).

duration_left: The number of days remaining in the current regime segment.
