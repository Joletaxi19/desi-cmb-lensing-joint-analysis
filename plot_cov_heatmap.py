import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_covariance(path):
    """Load covariance matrix and parameter names from a file."""
    with open(path) as f:
        header = f.readline().lstrip('#').split()
    cov = np.loadtxt(path, skiprows=1)
    return header, cov


def main():
    ap = argparse.ArgumentParser(description="Trace une heat map de la matrice de covariance")
    ap.add_argument('--covmat', default='yamls/chains/my_pr4.covmat', help='fichier de covariance')
    ap.add_argument('--output', default='covariance_heatmap.png', help="image de sortie")
    args = ap.parse_args()

    params, cov = load_covariance(args.covmat)

    sns.set_context('talk')
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cov, cmap='coolwarm', xticklabels=params, yticklabels=params,
                square=True, cbar_kws={'label': 'Covariance'}, ax=ax)

    ax.set_title('Matrice de covariance des paramètres')
    plt.tight_layout()
    plt.savefig(args.output)
    plt.close(fig)


if __name__ == '__main__':
    main()
