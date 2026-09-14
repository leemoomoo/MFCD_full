with open('D:/12487/NCDM PLUS2/paper/FCD_corrected.tex','r',encoding='utf-8') as f:
    content = f.read()

# Old order: fig2 (lambda figure) -> lambda text -> dim text -> fig3 (dim figure)
# New order: dim text -> fig3 (dim figure) -> lambda text -> fig2 (lambda figure)

old = '\\begin{figure}[t]\n\\centering\n\\includegraphics[width=0.8\\columnwidth]{fig2_lambda.png}\n\\caption{Sensitivity analysis of weight $\\lambda$.}\n\\label{fig:lambda_sensitivity}\n\\end{figure}\nTo investigate the impact of the weight \\(\\lambda\\) on model performance, Figure \\ref{fig:lambda_sensitivity} presents the AUC and ACC on the ASSIST09 dataset for $\\lambda\\in\\{0.0, 0.1, 0.3, 0.5, $ .7, 0.9, 1.0\\}$, where \\(\\lambda = 0.0\\) denotes pure mastery-based prediction and \\(\\lambda = 1.0\\) places maximum emphasis on the fluency stream. As shown, \\mymethod\\ attains its highest AUC at \\(\\lambda = 0.3\\), achieving an optimal trade-off between the two learning objectives.\n\nWe further examine the sensitivity of the latent dimension size \\(D\\) in the fluency branch. Figure \\ref{fig:dim_sensitivity} reports the AUC and ACC on the ASSIST09 dataset for \\(D \\in \\{4, 8, 16, 32, 64\\}\\). As shown, \\mymethod\\ achieves peak performance at \\(D = 32\\), suggesting that this configuration strikes the most favorable balance between representational capacity and generalization.\n\n\n\\begin{figure}[t]\n\\centering\n\\includegraphics[width=0.7\\columnwidth]{fig3_latent_dim.png}\n\\caption{Sensitivity analysis of dimension $ of latent fluency state $\\bm{z}_{i,k}$.}\n\\label{fig:dim_sensitivity}\n\\end{figure}'

new = 'We further examine the sensitivity of the latent dimension size \\(D\\) in the fluency branch. Figure \\ref{fig:dim_sensitivity} reports the AUC and ACC on the ASSIST09 dataset for \\(D \\in \\{4, 8, 16, 32, 64\\}\\). As shown, \\mymethod\\ achieves peak performance at \\(D = 32\\), suggesting that this configuration strikes the most favorable balance between representational capacity and generalization.\n\n\\begin{figure}[h!]\n\\centering\n\\includegraphics[width=0.7\\columnwidth]{fig3_latent_dim.png}\n\\caption{Sensitivity analysis of dimension $ of latent fluency state $\\bm{z}_{i,k}$.}\n\\label{fig:dim_sensitivity}\n\\end{figure}\n\nTo investigate the impact of the weight \\(\\lambda\\) on model performance, Figure \\ref{fig:lambda_sensitivity} presents the AUC and ACC on the ASSIST09 dataset for $\\lambda\\in\\{0.0, 0.1, 0.3, 0.5, $ .7, 0.9, 1.0\\}$, where \\(\\lambda = 0.0\\) denotes pure mastery-based prediction and \\(\\lambda = 1.0\\) places maximum emphasis on the fluency stream. As shown, \\mymethod\\ attains its highest AUC at \\(\\lambda = 0.3\\), achieving an optimal trade-off between the two learning objectives.\n\n\\begin{figure}[h!]\n\\centering\n\\includegraphics[width=0.8\\columnwidth]{fig2_lambda.png}\n\\caption{Sensitivity analysis of weight $\\lambda$.}\n\\label{fig:lambda_sensitivity}\n\\end{figure}'

if old in content:
    content = content.replace(old, new)
    with open('D:/12487/NCDM PLUS2/paper/FCD_corrected.tex','w',encoding='utf-8') as f:
        f.write(content)
    print('SUCCESS: Figures swapped and [t] -> [h!]')
else:
    print('ERROR: Could not find the old block')
    # Find key pieces
    for p in ['fig2_lambda.png','fig3_latent_dim.png']:
        pos = content.find(p)
        if pos > 0:
            print(f'{p} found at position {pos}')
