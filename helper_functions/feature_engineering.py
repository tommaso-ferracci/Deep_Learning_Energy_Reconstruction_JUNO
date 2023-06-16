import numpy as np 
import pandas as pd
from scipy.stats import skew, kurtosis, entropy

def create_features(pmt_pos, charge, fht, spmt_pos, charge_s, fht_s, return_dataframe=False):
    """
    Computes all the engineered features and returns them.

    Parameters:
        pmt_pos (ndarray): 2D array of (x, y, z) positions for each LPMT
        charge (ndarray): 2D array of charges for each LPMT
        fht (ndarray): 2D array of first-hit-times for each LPMT
        spmt_pos (ndarray): 2D array of (x, y, z) positions for each SPMT
        charge_s (ndarray): 2D array of charges for each SPMT
        fht_s (ndarray): 2D array of first-hit-times for each SPMT
        return_dataframe (bool, optional): specify whether to return features as a dataframe

    Returns:
        feature_names (list): list of all the features names
        features (ndarray): matrix containing all the features as column vectors
    """
    features_names = ["accum_charge", "nPMTs", "x_cc", "y_cc", "z_cc", "R_cc", "theta_cc", "phi_cc", "J_cc", "rho_cc", 
                      "gamma_x_cc", "gamma_y_cc", "gamma_z_cc", "pe2_cc", "pe5_cc", "pe10_cc", "pe15_cc", "pe20_cc", "pe25_cc", 
                      "pe30_cc", "pe35_cc", "pe40_cc", "pe45_cc", "pe50_cc", "pe55_cc", "pe60_cc", "pe65_cc", "pe70_cc", 
                      "pe75_cc", "pe80_cc", "pe85_cc", "pe90_cc", "pe95_cc", "pe_mean", "pe_std", "pe_skew", "pe_kurtosis", 
                      "pe_entropy", "x_cht", "y_cht", "z_cht", "R_cht", "theta_cht", "phi_cht", "J_cht", "rho_cht", "gamma_x_cht", 
                      "gamma_y_cht", "gamma_z_cht", "pe2_cht", "pe5_cht", "pe10_cht", "pe15_cht", "pe20_cht", "pe25_cht", 
                      "pe30_cht", "pe35_cht", "pe40_cht", "pe45_cht", "pe50_cht", "pe55_cht", "pe60_cht", "pe65_cht", "pe70_cht", 
                      "pe75_cht", "pe80_cht", "pe85_cht", "pe90_cht", "pe95_cht", "pe_cht_mean", "pe_cht_std", "pe_cht_skew", 
                      "pe_cht_kurtosis", "pe_cht_entropy", "ht5_2", "ht10_5", "ht15_10", "ht20_15", "ht25_20", "ht30_25", 
                      "ht35_30", "ht40_35", "ht45_40", "ht50_45", "ht55_50", "ht60_55", "ht65_60", "ht70_65", "ht75_70", 
                      "ht80_75", "ht85_80", "ht90_85", "ht95_90", "pe2_cc_s", "pe5_cc_s", "pe10_cc_s", "pe15_cc_s", "pe20_cc_s", 
                      "pe25_cc_s", "pe30_cc_s", "pe35_cc_s", "pe40_cc_s", "pe45_cc_s", "pe50_cc_s", "pe55_cc_s", "pe60_cc_s", 
                      "pe65_cc_s", "pe70_cc_s", "pe75_cc_s", "pe80_cc_s", "pe85_cc_s", "pe90_cc_s", "pe95_cc_s", "pe_mean_s", 
                      "pe_std_s", "pe_skew_s", "pe_kurtosis_s", "pe_entropy_s", "pe2_cht_s", "pe5_cht_s", "pe10_cht_s", 
                      "pe15_cht_s", "pe20_cht_s", "pe25_cht_s", "pe30_cht_s", "pe35_cht_s", "pe40_cht_s", "pe45_cht_s", 
                      "pe50_cht_s", "pe55_cht_s", "pe60_cht_s", "pe65_cht_s", "pe70_cht_s", "pe75_cht_s", "pe80_cht_s", 
                      "pe85_cht_s", "pe90_cht_s", "pe95_cht_s", "pe_cht_mean_s", "pe_cht_std_s", "pe_cht_skew_s", 
                      "pe_cht_kurtosis_s", "pe_cht_entropy_s", "ht5_2_s", "ht10_5_s", "ht15_10_s", "ht20_15_s", "ht25_20_s", 
                      "ht30_25_s", "ht35_30_s", "ht40_35_s", "ht45_40_s", "ht50_45_s", "ht55_50_s", "ht60_55_s", "ht65_60_s", 
                      "ht70_65_s", "ht75_70_s", "ht80_75_s", "ht85_80_s", "ht90_85_s", "ht95_90_s"]

    # compute main features
    accum_charge, nPMTs = np.zeros(len(pmt_pos)), np.zeros(len(pmt_pos))
    for i in range(len(pmt_pos)):
        accum_charge[i] = np.sum(charge[i]) + np.sum(charge_s[i])
        nPMTs[i] = np.sum(fht[i] != 0 )*20/23 + np.sum(fht_s[i] != 0)*3/23

    # compute geometric features (center of charge)
    x_cc, y_cc, z_cc = np.zeros(len(pmt_pos)), np.zeros(len(pmt_pos)), np.zeros(len(pmt_pos))
    for i in range(len(pmt_pos)):
        x_cc[i] = np.sum(pmt_pos[i][:, 0] * charge[i]) / np.sum(charge[i]) + \
                  np.sum(spmt_pos[i][:, 0] * charge_s[i]) / np.sum(charge_s[i])
        y_cc[i] = np.sum(pmt_pos[i][:, 1] * charge[i]) / np.sum(charge[i]) + \
                  np.sum(spmt_pos[i][:, 1] * charge_s[i]) / np.sum(charge_s[i])
        z_cc[i] = np.sum(pmt_pos[i][:, 2] * charge[i]) / np.sum(charge[i]) + \
                  np.sum(spmt_pos[i][:, 2] * charge_s[i]) / np.sum(charge_s[i])
    R_cc = np.sqrt(x_cc**2 + y_cc**2 + z_cc**2)
    theta_cc = np.arctan(np.sqrt(x_cc**2 + y_cc**2)/z_cc)
    phi_cc = np.arctan(y_cc/x_cc)
    J_cc = R_cc**2 * np.sin(theta_cc)
    rho_cc = np.sqrt(x_cc**2 + y_cc**2)
    gamma_x_cc = x_cc/np.sqrt(y_cc**2 + z_cc**2)
    gamma_y_cc = y_cc/np.sqrt(x_cc**2 + z_cc**2)
    gamma_z_cc = z_cc/np.sqrt(x_cc**2 + y_cc**2)

    # compute geometric features (center of first-hit-time)
    x_cht, y_cht, z_cht = np.zeros(len(pmt_pos)), np.zeros(len(pmt_pos)), np.zeros(len(pmt_pos))
    for i in range(len(pmt_pos)):
        x_cht[i] = np.sum(pmt_pos[i][:, 0] / (fht[i] + 50)) / np.sum(1 / (fht[i] + 50)) + \
                   np.sum(spmt_pos[i][:, 0] / (fht_s[i] + 50)) / np.sum(1 / (fht_s[i] + 50))
        y_cht[i] = np.sum(pmt_pos[i][:, 1] / (fht[i] + 50)) / np.sum(1 / (fht[i] + 50)) + \
                   np.sum(spmt_pos[i][:, 1] / (fht_s[i] + 50)) / np.sum(1 / (fht_s[i] + 50))
        z_cht[i] = np.sum(pmt_pos[i][:, 2] / (fht[i] + 50)) / np.sum(1 / (fht[i] + 50)) + \
                   np.sum(spmt_pos[i][:, 2] / (fht_s[i] + 50)) / np.sum(1 / (fht_s[i] + 50))
    R_cht = np.sqrt(x_cht**2 + y_cht**2 + z_cht**2)
    theta_cht = np.arctan(np.sqrt(x_cht**2 + y_cht**2)/z_cht)
    phi_cht = np.arctan(y_cht/x_cht)
    J_cht = R_cht**2 * np.sin(theta_cht)
    rho_cht = np.sqrt(x_cht**2 + y_cht**2)
    gamma_x_cht = x_cht/np.sqrt(y_cht**2 + z_cht**2)
    gamma_y_cht = y_cht/np.sqrt(x_cht**2 + z_cht**2)
    gamma_z_cht = z_cht/np.sqrt(x_cht**2 + y_cht**2)

    # compute charge and first-hit-time percentiles
    pe_cc, pe_cht = np.zeros((len(pmt_pos), 20)), np.zeros((len(pmt_pos), 20))
    pe_cc_s, pe_cht_s = np.zeros((len(spmt_pos), 20)), np.zeros((len(spmt_pos), 20))
    for i in range(len(pmt_pos)):
        for j, x in enumerate([2] + list(range(5, 100, 5))):
            pe_cc[i, j] = np.percentile(charge[i], x)
            pe_cht[i, j] = np.percentile(fht[i], x)
    for i in range(len(spmt_pos)):
        for j, x in enumerate([2] + list(range(5, 100, 5))):
            pe_cc_s[i, j] = np.percentile(charge_s[i], x)
            pe_cht_s[i, j] = np.percentile(fht_s[i], x)

    # compute percentiles deltas for first-hit-time
    ht = np.array([pe_cht[:, i+1] - pe_cht[:, i] for i in range(pe_cht.shape[1]-1)]).T
    ht_s = np.array([pe_cht_s[:, i+1] - pe_cht_s[:, i] for i in range(pe_cht_s.shape[1]-1)]).T

    # compute charge and first-hit-time descriptive statistics
    pe_mean, pe_std = np.zeros(len(pmt_pos)), np.zeros(len(pmt_pos))
    pe_skew, pe_kurtosis, pe_entropy = np.zeros(len(pmt_pos)), np.zeros(len(pmt_pos)), np.zeros(len(pmt_pos))
    pe_cht_mean, pe_cht_std = np.zeros(len(pmt_pos)), np.zeros(len(pmt_pos))
    pe_cht_skew, pe_cht_kurtosis, pe_cht_entropy = np.zeros(len(pmt_pos)), np.zeros(len(pmt_pos)), np.zeros(len(pmt_pos))

    pe_mean_s, pe_std_s = np.zeros(len(spmt_pos)), np.zeros(len(spmt_pos))
    pe_skew_s, pe_kurtosis_s, pe_entropy_s = np.zeros(len(spmt_pos)), np.zeros(len(spmt_pos)), np.zeros(len(spmt_pos))
    pe_cht_mean_s, pe_cht_std_s = np.zeros(len(spmt_pos)), np.zeros(len(spmt_pos))
    pe_cht_skew_s, pe_cht_kurtosis_s, pe_cht_entropy_s = np.zeros(len(spmt_pos)), np.zeros(len(spmt_pos)), np.zeros(len(spmt_pos))

    num_bins = 100 # parameter needed for entropy calculation
    for i in range(len(pmt_pos)):
        pe_mean[i] = np.mean(charge[i])
        pe_std[i] = np.std(charge[i])
        pe_skew[i] = skew(charge[i])
        pe_kurtosis[i] = kurtosis(charge[i])
        hist, _ = np.histogram(charge[i], bins=num_bins)
        probs = hist / np.sum(hist)
        pe_entropy[i] = entropy(probs, base=2)
        pe_cht_mean[i] = np.mean(fht[i])
        pe_cht_std[i] = np.std(fht[i])
        pe_cht_skew[i] = skew(fht[i])
        pe_cht_kurtosis[i] = kurtosis(fht[i])
        hist, _ = np.histogram(fht[i], bins=num_bins)
        probs = hist / np.sum(hist)
        pe_cht_entropy[i] = entropy(probs, base=2)

    for i in range(len(spmt_pos)):
        pe_mean_s[i] = np.mean(charge_s[i])
        pe_std_s[i] = np.std(charge_s[i])
        pe_skew_s[i] = skew(charge_s[i])
        pe_kurtosis_s[i] = kurtosis(charge_s[i])
        hist, _ = np.histogram(charge_s[i], bins=num_bins)
        probs = hist / np.sum(hist)
        pe_entropy_s[i] = entropy(probs, base=2)
        pe_cht_mean_s[i] = np.mean(fht_s[i])
        pe_cht_std_s[i] = np.std(fht_s[i])
        pe_cht_skew_s[i] = skew(fht_s[i])
        pe_cht_kurtosis_s[i] = kurtosis(fht_s[i])
        hist, _ = np.histogram(fht_s[i], bins=num_bins)
        probs = hist / np.sum(hist)
        pe_cht_entropy_s[i] = entropy(probs, base=2)

    features = np.column_stack((accum_charge, nPMTs, x_cc, y_cc, z_cc, R_cc, theta_cc, phi_cc, J_cc, rho_cc, gamma_x_cc,
                                gamma_y_cc, gamma_z_cc, pe_cc, pe_mean, pe_std, pe_skew, pe_kurtosis, pe_entropy, x_cht,
                                y_cht, z_cht, R_cht, theta_cht, phi_cht, J_cht, rho_cht, gamma_x_cht, gamma_y_cht, gamma_z_cht,
                                pe_cht, ht, pe_cht_mean, pe_cht_std, pe_cht_skew, pe_cht_kurtosis, pe_cht_entropy, pe_cc_s, 
                                pe_mean_s, pe_std_s, pe_skew_s, pe_kurtosis_s, pe_entropy_s, pe_cht_s, ht_s, pe_cht_mean_s, 
                                pe_cht_std_s, pe_cht_skew_s, pe_cht_kurtosis_s, pe_cht_entropy_s))
                                
    if return_dataframe:
        features_dataframe = pd.DataFrame(data=features, columns=features_names)
        return features_names, features, features_dataframe
    else:
        return features_names, features