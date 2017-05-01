import pickle as pkl
import numpy as np


output_cavi_wmean = open('outputs/CAVI_Wmean_probit_int.pickle', 'rb')
w_mean_cavi = pkl.load(output_cavi_wmean)

D = len(w_mean_cavi)

output_wvar = open('outputs/Wvar_probit.pickle', 'rb')
w_var = pkl.load(output_wvar)

with open('keywords.txt','r') as f:
	keywords = [s.strip() for s in f.readlines()]

boldface = False
if boldface:
	lbf = '\\textbf{'
	rbf = '} & '
else:
	lbf = ''
	rbf = ' & '

kw_row = lambda inds : '\\textbf{Keyword} & ' + (''.join([lbf + keywords[ind] + rbf for ind in inds]))[:-2] + '\\\\\n\\midrule\n'
wm_row = lambda inds : 'Estimate $\\widehat w$ & ' + ' & '.join(["%.2f" % (w_mean_cavi[ind]) for ind in inds]) + '\\\\\n\\midrule\n'
#zv_row = lambda inds : '$z$-value & ' + ' & '.join(["%.2f" % (w_mean_cavi[i]/np.sqrt(w_var[i,i])) for i in DD[inds]]) + '\\\\\n\\midrule\n'

I1 = [0,1,4,15,16,17,27]
I2 = [29,36,41,42,52,53,54]

row1 = kw_row(I1)
row2 = wm_row(I1)
row4 = kw_row(I2)
row5 = wm_row(I2)

table_txt = \
"""
\\begin{table}
\\vspace{2ex}
\\begin{tabular}{c | c | c | c | c | c | c | c}
\\toprule
""" + \
row1 + row2 + row4 + row5[:-1] + \
"""
\\bottomrule
\\end{tabular}
\\caption{Table caption}
\\end{table}
"""

print table_txt
#print row2
#print row5
#print row8
