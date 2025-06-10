import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('talk')

# Progress plot
progress_path = 'yamls/chains/my_pr4.progress'
progress = pd.read_csv(
    progress_path,
    sep='\s+',
    comment='#',
    names=['N', 'timestamp', 'acceptance_rate', 'Rminus1', 'Rminus1_cl'],
)

plt.figure(figsize=(8, 5))
plt.plot(progress['N'], progress['Rminus1'], marker='o', ms=4)
plt.xlabel('Nombre de points')
plt.ylabel(r'$\hat{R}-1$')
plt.title('Convergence des chaînes')
plt.grid(True)
plt.tight_layout()
plt.savefig('my_pr4_convergence.png')

# Scatter plot of chains
chain_files = sorted(glob.glob('yamls/chains/my_pr4.[0-9]*.txt'),
                     key=lambda x: int(x.split('.')[-2]))

palette = sns.color_palette('husl', len(chain_files))
plt.figure(figsize=(8, 6))
for idx, (chain_file, color) in enumerate(zip(chain_files, palette)):
    with open(chain_file) as f:
        header = f.readline().lstrip('#').split()
    data = pd.read_csv(chain_file, sep='\s+', comment='#', names=header)
    if len(data) > 2000:
        data = data.sample(2000, random_state=0)
    plt.scatter(data['OmM'], data['sigma8'], s=10, color=color, alpha=0.6,
                label=f'Chaîne {idx+1}')
plt.xlabel(r'$\Omega_m$')
plt.ylabel(r'$\sigma_8$')
plt.title('Exploration des chaînes')
plt.legend(ncol=2, fontsize=10)
plt.tight_layout()
plt.savefig('my_pr4_scatter.png')
