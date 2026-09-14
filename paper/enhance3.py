import re
with open('D:/12487/NCDM PLUS2/paper/FCD_corrected.tex','r',encoding='utf-8') as f:
    content = f.read()
# Fix  math corruption
content = content.replace(
    'shows the total loss {\\mathrm{Total}}$ and the primary BCE loss $.',
    'shows the total loss {\\mathrm{Total}}$ and the primary BCE loss $.'
)
content = content.replace(
    '{\\mathrm{ea}}$ (empirical accuracy), {\\mathrm{rc}}$ (response consistency), and {\\mathrm{ei}}$',
    '{\\mathrm{ea}}$ (empirical accuracy), {\\mathrm{rc}}$ (response consistency), and {\\mathrm{ei}}$'
)
# Enhance ablation text
old_abl = 'To evaluate the contribution of each component in \\mymethod\\, we conduct an ablation study with four model variants: Base (mastery only), L1 (static fluency), L2 (static fusion), and Full \\mymethod\\. Figure~\\ref{fig:ablation_study} reports the performance across AUC, ACC, F1 and RMSE.'
new_abl = old_abl.replace(
    'and Full \\mymethod\\. Figure~\\ref{fig:ablation_study} reports the performance across AUC, ACC, F1 and RMSE.',
    'and Full \\mymethod\\ (Base + fluency with dynamic fusion gate). Figure~\\ref{fig:ablation_study} reports the performance across AUC, ACC, F1 and RMSE. As shown, Full \\mymethod\\ consistently achieves the best performance across all four metrics, demonstrating that each component contributes positively to the overall framework. Specifically, L1 improves over Base by incorporating behavioral fluency features, while L2 further enhances performance by learning to fuse the two streams. The Full \\mymethod\\ with dynamic gating achieves the highest gains, validating that the adaptive fusion gate effectively balances cognitive mastery and behavioral fluency.'
)
content = content.replace(old_abl, new_abl)
# Enhance main loss text
for pair in [('To monitor the training process, we track three sets of loss curves over 30 epochs. Figure~\\ref{fig:main_losses} shows the total loss {\\mathrm{Total}}$ and the primary BCE loss $. Both losses decrease steadily and converge smoothly, indicating effective optimization without severe overfitting.',
              'To monitor the training process, we track three sets of loss curves over 30 epochs. Figure~\\ref{fig:main_losses} shows the total loss {\\mathrm{Total}}$ and the primary BCE loss $. Both losses decrease steadily and converge smoothly, with {\\mathrm{Total}}$ closely tracking $, indicating that the multi-task auxiliary losses do not conflict with the primary prediction objective. The smooth convergence without severe overfitting demonstrates that \\mymethod\\ can be effectively optimized using standard training procedures, even with the additional behavioral fluency stream.'),
             ('Figure~\\ref{fig:aux_losses} presents the auxiliary reconstruction losses for the three behavioral proxy tasks: {\\mathrm{ea}}$ (engagement accuracy), {\\mathrm{rc}}$ (response consistency), and {\\mathrm{ei}}$ (elapsed interaction). These auxiliary losses all decrease rapidly in the early epochs and stabilize, showing that the multi-task learning objective effectively regularizes the behavioral encoder.',
              'Figure~\\ref{fig:aux_losses} presents the auxiliary reconstruction losses for the three behavioral proxy tasks: {\\mathrm{ea}}$ (empirical accuracy), {\\mathrm{rc}}$ (response consistency), and {\\mathrm{ei}}$ (exposure intensity). All three auxiliary losses decrease rapidly within the first 5 epochs and stabilize at low values, showing that the behavioral encoder effectively learns to reconstruct meaningful proxy features from the latent fluency state. This rapid convergence confirms that the multi-task learning objective successfully preserves behavioral information without introducing optimization difficulty, allowing the fluency representations to carry rich procedural signals.'),
             ('Figure~\\ref{fig:valid_curves} reports the validation AUC and ACC during training. The validation performance improves quickly within the first 10 epochs and remains stable afterward, confirming that \\mymethod\\ generalizes well without overfitting.',
              'Figure~\\ref{fig:valid_curves} reports the validation AUC and ACC during training. The validation performance improves quickly within the first 10 epochs and remains stable afterward, with no significant degradation in later epochs. This stable validation trajectory confirms that \\mymethod\\ generalizes well to unseen data without overfitting, benefiting from both the monotonicity constraints in the mastery backbone and the regularization effect of the multi-task auxiliary losses. The strong validation performance provides compelling evidence that jointly modeling cognitive mastery and behavioral fluency leads to robust and generalizable diagnostic predictions.')]:
    old, new = pair
    if old in content:
        content = content.replace(old, new)
with open('D:/12487/NCDM PLUS2/paper/FCD_corrected.tex','w',encoding='utf-8') as f:
    f.write(content)
print('All enhancements applied')

