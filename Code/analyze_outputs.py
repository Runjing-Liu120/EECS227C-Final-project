import pickle as pkl
import numpy as np


output_cavi_wmean = open('outputs/CAVI_Wmean_probit.pickle', 'rb')
w_mean_cavi = pkl.load(output_cavi_wmean)

output_pxvb_wmean = open('outputs/PXVB_Wmean_probit.pickle', 'rb')
w_mean_pxvb = pkl.load(output_pxvb_wmean)

output_wvar = open('outputs/Wvar_probit.pickle', 'rb')
w_var = pkl.load(output_wvar)


print np.sqrt(np.diag(w_var))

with open('keywords.txt','r') as f:
	keywords = [s.strip() for s in f.readlines()]

boldface = False
if boldface:
	lbf = '\\textbf{'
	rbf = '} & '
else:
	lbf = ''
	rbf = ' & '

kw_row = lambda inds : '\\textbf{Keyword} & ' + (''.join([lbf + word + rbf for word in keywords[inds]]))[:-2] + '\\\\\n'
wm_row = lambda inds : 'Estimate $\\widehat w$ & ' + ' & '.join(["%.2f" % v for v in w_mean_cavi[inds]]) + '\\\\\n'

row1 = kw_row(slice(0,19))
row2 = wm_row(slice(0,19))
#row3 = 's.e. & ' + ' & '.join(["%.2f" % v for v in w_mean_cavi[:19]]) + '\\\\' # $\\sqrt{\\Sigma_{\\widehat w}}$
row4 = kw_row(slice(19,38))
row5 = wm_row(slice(19,38))
row7 = kw_row(slice(38,57))
row8 = wm_row(slice(38,57))

table_txt = \
"""
\\begin{table}
\\vspace{2ex}
\\begin{tabular}{c | c | c | c | c | c | c | c | c | c | c | c | c | c | c | c | c | c | c | c}
\\toprule
""" + \
row1 + '\\midrule\n ' + row2 + row4 + '\\midrule\n' + row5 + row7 + '\\midrule\n' + row8[:-1]  +\
"""
\\bottomrule
\\end{tabular}
\\caption{Table caption}
\\end{table}
"""

print table_txt
