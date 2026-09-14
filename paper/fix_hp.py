import sys
content = open('D:/12487/NCDM PLUS2/paper/FCD_with_assist09.tex', 'r', encoding='utf-8').read()
start = content.find('subsection{Hyperparameter Sensitivity Analysis}')
end = content.find('subsection{Ablation Study}')
if start >= 0 and end > start:
    # Find the actual start (back to backslash before the marker)
    start_marker = content.rfind('\\', 0, start)
    if start_marker >= 0:
        # Build new content
        new_part = (
            '\\subsection{Hyperparameter Sensitivity Analysis}\n'
            'To systematically evaluate the impact of the learning rate on our framework, we conduct a sensitivity analysis. Figure \\ref{fig:lr_sensitivity} illustrates the effect of varying the learning rate on the overall AUC. An excessively small learning rate leads to slow convergence, while an overly large one causes unstable training. As shown, the model achieves peak AUC of 0.9579 at lr=0.0008, with performance degrading monotonically as the learning rate deviates from this optimum.\n\n'
            '\\begin{figure}[t]\n'
            '\\centering\n'
            '\\includegraphics[width=0.58\\columnwidth]{fig1_lr.png}\n'
            '\\caption{Learning rate sensitivity analysis. Optimal AUC of 0.9579 at lr=0.0008.}\n'
            '\\label{fig:lr_sensitivity}\n'
            '\\end{figure}\n\n'
        )
        content = content[:start_marker] + new_part + content[end:]
        with open('D:/12487/NCDM PLUS2/paper/FCD_with_assist09.tex', 'w', encoding='utf-8') as f:
            f.write(content)
        print('Written')
    else:
        print('Start marker not found')
else:
    print('Not found', start, end)
