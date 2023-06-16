import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
from scipy.stats import norm

# perform fit with scipy.optimize.curve_fit
def plot_gaussian_fit(data, n_bins, name, index):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,5), dpi=100)

    counts, bins, patches = ax.hist(data, n_bins, density=True, color="steelblue", edgecolor="k", linewidth=0.5);
    bin_centers = 0.5*(bins[1:] + bins[:-1])

    popt, pcov = curve_fit(lambda x, mean, std : norm.pdf(x, mean, std), bin_centers, counts, p0=[np.mean(data), np.std(data)])

    mean, std = popt[0], np.abs(popt[1])
    err_mean, err_std = np.sqrt(np.abs(np.diag(pcov))[0]), np.sqrt(np.abs(np.diag(pcov))[1])

    x = np.linspace(data.min(), data.max(), 1000)
    ax.plot(x, norm.pdf(x, mean, std), "purple", linewidth=1, label=f"$\mu$ = {mean:.3f} MeV, $\sigma$ = {std:.3f} MeV")
    ax.set_xlabel("$E_{vis} - E_{pred}$ [MeV]", fontsize=15)
    #ax.set_xlim((-0.2, 0.2))
    ax.set_ylim((0, max(counts)+3))
    legend = ax.legend(prop={'size': 12}, fancybox=False, edgecolor="k", loc="upper right")
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.tick_params(axis="both", which='minor', labelsize=12)
    ax.grid()

    fig.savefig(f"/home/ferracci/new_dataset/images/results/{name}_{index}.png", dpi=300, bbox_inches="tight", pad_inches=0.2);

    return mean, std, err_mean, err_std

# energy resolution function
def energy_res_func(x, a, b, c):
    return np.sqrt((a/(x**0.5))**2 + b**2 + (c/x)**2)

# energy resolution fit
def energy_res_fit(x, y, yerr):
    popt, pcov = curve_fit(energy_res_func, x, y, sigma=yerr, p0=[2.7, 0.8, 1.2], maxfev=1000)
    a, b, c = popt
    return a, b, c, pcov

# retrieve a_tilde and its error from the best fit estimates
def get_a_tilde(a, b, c, err_a, err_b, err_c, cov_ab, cov_ac, cov_bc):
    a_tilde = (a**2 + (1.6*b)**2 + (c/1.6)**2)**0.5
    err_a_tilde = np.sqrt(1/a_tilde**2 * ((a*err_a)**2 + (2.56*b*err_b)**2 + (c*err_c/2.56)**2 + 5.12*a*b*cov_ab + 
                          a*c*cov_ac/1.28 + 2*b*c*cov_bc))
    return a_tilde, err_a_tilde