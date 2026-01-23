import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
import math

def calculate_bias_metrics(adata, layer=None, 
                           gene_type_col='GeneType', 
                           target_type='protein_coding',
                           gc_col='GC_Percent', 
                           len_col='log10_Length'):

    print(f"--- Calculating Bias Metrics (Target Layer: {layer if layer else 'X'}) ---")
    
    # 1. Setup Data Matrix (Dense array needed for calculations)
    if layer and layer in adata.layers:
        X_data = adata.layers[layer]
    else:
        X_data = adata.X
        
    # Ensure dense format if sparse
    if hasattr(X_data, "toarray"):
        X_data = X_data.toarray()
    
    # DataFrame to store results
    metrics_df = pd.DataFrame(index=adata.obs_names)

    # Helper: Compute correlation/bias score
    def _compute_score(X, feat_vals, mode='spearman'):
        scores = []
        # Ensure feature values are float array
        feat_vals = np.array(feat_vals, dtype=float)

        for i in range(X.shape[0]):
            sample_expr = np.ravel(X[i, :])
            mask = sample_expr > 0 # Filter for detected genes only
            
            # Skip if too few genes detected
            if np.sum(mask) < 50:
                scores.append(0)
                continue
            
            valid_expr = sample_expr[mask]
            valid_feat = feat_vals[mask]

            if mode == 'lowess':
                smoothed = lowess(valid_expr, valid_feat, frac=0.3, it=0)
                bias_val = np.ptp(smoothed[:, 1]) 
                scores.append(bias_val)
            else:
                corr, _ = spearmanr(valid_expr, valid_feat)
                scores.append(corr if not np.isnan(corr) else 0)
        return np.array(scores)

    if gene_type_col in adata.var.columns:
        coding_mask = (adata.var[gene_type_col] == target_type).values
        if not np.any(coding_mask):
            print(f"  [Warning] No genes found for type '{target_type}'. Using all genes.")
            coding_mask = np.ones(adata.n_vars, dtype=bool)
    else:
        coding_mask = np.ones(adata.n_vars, dtype=bool)
        
    subset_X = X_data[:, coding_mask]
    subset_var = adata.var.iloc[coding_mask]

    if gc_col in subset_var.columns:
        print("  > Computing GC bias score (LOESS)...")
        metrics_df['gc_bias_score'] = _compute_score(
            subset_X, subset_var[gc_col], mode='lowess'
        )
    else:
        print(f"  [Skip] GC column '{gc_col}' not found.")

    # 4. Calculate Length Bias
    if len_col in subset_var.columns:
        print("  > Computing Length bias score (Spearman)...")
        metrics_df['len_bias_score'] = _compute_score(
            subset_X, subset_var[len_col], mode='spearman'
        )
    else:
        print(f"  [Skip] Length column '{len_col}' not found.")

    # 5. Calculate Platelet Score (Using scanpy score_genes)
    if platelet_col in adata.var.columns:
        platelet_genes = adata.var_names[adata.var[platelet_col]].tolist()
        if platelet_genes:
            print(f"  > Computing Platelet scores ({len(platelet_genes)} genes)...")
            # score_genes requires the full object, so we create a temporary one
            temp_adata = sc.AnnData(X=X_data, obs=adata.obs, var=adata.var)
            sc.tl.score_genes(temp_adata, gene_list=platelet_genes, score_name='platelet_score')
            metrics_df['platelet_score'] = temp_adata.obs['platelet_score'].values
        else:
             print("  [Skip] No platelet genes identified in var.")

    # 6. Basic stats
    metrics_df['total_counts'] = np.ravel(X_data.sum(axis=1))
    metrics_df['log1p_total_counts'] = np.log1p(metrics_df['total_counts'])
    
    print("Calculation Done.\n")
    return metrics_df


# ==============================================================================
# 2. Class: Analysis Pipeline
# ==============================================================================
class RNASeqQCPipeline:
    def __init__(self, adata, 
                 bias_metrics_df=None,
                 phenotype_col='Type', 
                 batch_col='Batch_Granular',
                 analysis_metrics=None):
        """
        Args:
            adata: AnnData object.
            bias_metrics_df: DataFrame containing calculated bias metrics (index must match adata.obs).
                             If None, assumes metrics are already in adata.obs.
            phenotype_col: Column name for biological condition of interest.
            batch_col: Column name for batch info.
            analysis_metrics: List of column names in obs to use for QC analysis. 
                              If None, defaults to ['gc_bias_score', 'len_bias_score', 'platelet_score', 'log1p_total_counts'].
        """
        self.adata = adata.copy()
        
        # Merge external metrics if provided
        if bias_metrics_df is not None:
            # Join carefully to avoid duplicates
            cols_to_use = bias_metrics_df.columns.difference(self.adata.obs.columns)
            self.adata.obs = self.adata.obs.join(bias_metrics_df[cols_to_use])
            
        self.cols = {
            'phenotype': phenotype_col,
            'batch': batch_col
        }
        
        # Define which metrics to track
        default_metrics = ['gc_bias_score', 'len_bias_score', 'platelet_score', 'log1p_total_counts']
        self.target_metrics = analysis_metrics if analysis_metrics else default_metrics
        
        # Validate metrics existence
        missing = [m for m in self.target_metrics if m not in self.adata.obs.columns]
        if missing:
            print(f"[Warning] The following metrics are missing from obs: {missing}")
        
        self._prepare_metadata()
        
    def _prepare_metadata(self):
        """Prepare categorical data for analysis (Unknown handling, etc.)"""
        print("--- [Init] Preparing Metadata ---")
        
        # Filter NaNs in phenotype
        if self.cols['phenotype'] in self.adata.obs.columns:
            n_orig = self.adata.n_obs
            self.adata = self.adata[~self.adata.obs[self.cols['phenotype']].isna()].copy()
            if self.adata.n_obs < n_orig:
                print(f"  Dropped {n_orig - self.adata.n_obs} samples with missing phenotype.")

        # Handle Batch NaNs
        if self.cols['batch'] in self.adata.obs.columns:
            if self.adata.obs[self.cols['batch']].isna().any():
                if self.adata.obs[self.cols['batch']].dtype.name == 'category':
                    self.adata.obs[self.cols['batch']] = self.adata.obs[self.cols['batch']].cat.add_categories(['Unknown'])
                self.adata.obs[self.cols['batch']] = self.adata.obs[self.cols['batch']].fillna('Unknown')
        
        print(f"  Active Metrics for Analysis: {[m for m in self.get_active_metrics()]}")

    def get_active_metrics(self):
        """Return list of metrics that actually exist in the current adata.obs"""
        return [m for m in self.target_metrics if m in self.adata.obs.columns]

    def set_layer(self, layer_name=None):
        if layer_name is None:
            print("--- Set Layer: Using default X ---")
        elif layer_name in self.adata.layers:
            print(f"--- Set Layer: Switching to '{layer_name}' ---")
            self.adata.X = self.adata.layers[layer_name].copy()
        else:
            raise ValueError(f"Layer '{layer_name}' not found.")

        # Reset PCA
        for key in ['X_pca', 'pca', 'PCs']:
            if key in self.adata.obsm: del self.adata.obsm[key]
            if key in self.adata.uns: del self.adata.uns[key]
            if key in self.adata.varm: del self.adata.varm[key]
        print(f"  PCA reset. Active layer ready.")

    def run_pca_diagnostics(self):
        if 'X_pca' not in self.adata.obsm: 
            sc.tl.pca(self.adata)
        
        print("--- [Analysis] PCA Diagnostics ---")
        fig_top, ax_top = plt.subplots(1, 2, figsize=(20, 5), gridspec_kw={'width_ratios': [3, 7]})
        
        # 1. Scree Plot
        var_ratios = self.adata.uns['pca']['variance_ratio']
        ax_top[0].plot(range(1, len(var_ratios)+1), var_ratios, 'o-k', alpha=0.7)
        ax_top[0].set_title("Scree Plot")
        ax_top[0].set_xlabel("PC"); ax_top[0].set_ylabel("Variance Ratio")
        
        # 2. Heatmap of Metrics
        active_metrics = self.get_active_metrics()
        if active_metrics:
            df_plot = self.adata.obs[active_metrics].copy()
            df_scaled = (df_plot - df_plot.mean()) / df_plot.std()
            sns.heatmap(df_scaled.T, cmap='RdBu_r', center=0, ax=ax_top[1], cbar_kws={'label': 'Z-score'})
            ax_top[1].set_title("Bias Metrics per Sample")
        
        plt.tight_layout(); plt.show()

        # 3. Scatter Plots
        print("--- [Analysis] PCA Scatter Plots ---")
        pc1_var = var_ratios[0]
        pc2_var = var_ratios[1]
        
        plot_keys = active_metrics + [self.cols['phenotype'], self.cols['batch']]
        
        n_cols = 3
        n_rows = math.ceil(len(plot_keys) / n_cols)
        fig_pca, axes_pca = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        axes_pca = axes_pca.flatten()
        
        for i, key in enumerate(plot_keys):
            if key not in self.adata.obs.columns: continue
            
            is_numeric = pd.api.types.is_numeric_dtype(self.adata.obs[key])
            sc.pl.pca(
                self.adata, color=key, ax=axes_pca[i], show=False, 
                cmap='RdBu_r' if is_numeric else None,
                size=100, legend_loc='right margin'
            )
            axes_pca[i].set_xlabel(f"PC1 ({pc1_var:.1%})")
            axes_pca[i].set_ylabel(f"PC2 ({pc2_var:.1%})")
            
        for j in range(i+1, len(axes_pca)): axes_pca[j].axis('off')
        plt.tight_layout(); plt.show()

    def analyze_pc_associations(self, n_pcs=5):
        if 'X_pca' not in self.adata.obsm: sc.tl.pca(self.adata)
        pc_df = pd.DataFrame(self.adata.obsm['X_pca'][:, :n_pcs], 
                             columns=[f'PC{i+1}' for i in range(n_pcs)], index=self.adata.obs_names)
        
        cont_vars = self.get_active_metrics()
        cat_vars = [self.cols['phenotype'], self.cols['batch']]
        assoc_matrix = pd.DataFrame(index=cont_vars + cat_vars, columns=pc_df.columns)

        # Continuous: Spearman
        for col in cont_vars:
            for pc in pc_df.columns:
                corr, _ = spearmanr(self.adata.obs[col], pc_df[pc])
                assoc_matrix.loc[col, pc] = abs(corr) if not np.isnan(corr) else 0
        
        # Categorical: ANOVA
        for col in cat_vars:
            if col not in self.adata.obs.columns or self.adata.obs[col].nunique() < 2:
                assoc_matrix.loc[col, :] = 0
                continue
            for pc in pc_df.columns:
                temp = pd.concat([self.adata.obs[col], pc_df[pc]], axis=1).dropna()
                temp.columns = ['G', 'V']
                try:
                    model = smf.ols('V ~ C(G)', data=temp).fit()
                    anova = sm.stats.anova_lm(model, typ=2)
                    assoc_matrix.loc[col, pc] = anova.loc['C(G)', 'sum_sq'] / anova['sum_sq'].sum()
                except:
                    assoc_matrix.loc[col, pc] = 0

        plt.figure(figsize=(8, len(assoc_matrix)*0.5 + 2))
        sns.heatmap(assoc_matrix.astype(float), annot=True, cmap='Reds', vmin=0, vmax=1, fmt='.2f')
        plt.title(f"PC - Variable Association")
        plt.show()

    def analyze_partial_correlation(self, n_pcs=5):
        print(f"--- [Analysis] Partial Correlation (Control Confounders) ---")
        target = self.cols['phenotype']
        confounders = self.get_active_metrics() + [self.cols['batch']]
        
        if 'X_pca' not in self.adata.obsm: sc.tl.pca(self.adata)
        pc_df = pd.DataFrame(self.adata.obsm['X_pca'][:, :n_pcs], columns=[f'PC{i+1}' for i in range(n_pcs)], index=self.adata.obs_names)
        data = pd.concat([pc_df, self.adata.obs[[target] + [c for c in confounders if c in self.adata.obs.columns]]], axis=1)
        
        # Numerize target
        if data[target].dtype == 'object' or data[target].dtype.name == 'category':
            data['target_num'] = pd.factorize(data[target])[0]
        else: data['target_num'] = data[target]

        results = {}
        for pc in pc_df.columns:
            # Raw Correlation
            raw = spearmanr(data[pc], data['target_num'])[0]
            
            # Partial Correlation
            conf_terms = [f"C({c})" if data[c].dtype.name in ['category', 'object'] else c for c in confounders if c in data.columns]
            if not conf_terms: 
                partial = raw
            else:
                try:
                    res_pc = smf.ols(f"{pc} ~ {' + '.join(conf_terms)}", data=data).fit().resid
                    res_tg = smf.ols(f"target_num ~ {' + '.join(conf_terms)}", data=data).fit().resid
                    partial = spearmanr(res_pc, res_tg)[0]
                except: partial = 0
            results[pc] = {'Raw': abs(raw), 'Partial': abs(partial)}
            
        pd.DataFrame(results).T.plot(kind='bar', figsize=(8, 4), colormap='tab20')
        plt.grid(alpha=0.1); plt.title(f"Correlation with {target}: Raw vs Controlled"); plt.show()

    def run_cascade_analysis(self, pcs_to_check=['PC1'], batch_method='combat'):
        print(f"--- [Analysis] Cascade Correction (Method: {batch_method}) ---")
        
        adata_temp = self.adata.copy()
        
        # Define groups of variables for stepwise removal
        # 1. Technical Bias (Continuous)
        tech_vars = [m for m in self.get_active_metrics() if 'platelet' not in m] # usually GC, Len, Counts
        # 2. Biological/Contamination (e.g., Platelet)
        bio_noise_vars = [m for m in self.get_active_metrics() if 'platelet' in m]
        # 3. Batch (Categorical)
        batch_vars = [self.cols['batch']] if self.cols['batch'] in adata_temp.obs.columns else []
        
        steps = [
            ('1. Raw', [], False),
            ('2. -Tech', tech_vars, False),
            ('3. -Platelet', bio_noise_vars, False),
            ('4. -Batch', batch_vars, True)
        ]
        
        # Evaluation Targets
        eval_vars = tech_vars + bio_noise_vars + batch_vars + [self.cols['phenotype']]
        eval_vars = list(set([v for v in eval_vars if v in adata_temp.obs.columns])) # unique & existing

        pc_results = {pc: {'val': [], 'var': []} for pc in pcs_to_check}
        
        fig_pca, axes_pca = plt.subplots(1, len(steps), figsize=(5*len(steps), 5))
        if len(steps) == 1: axes_pca = [axes_pca]
        axes_pca = axes_pca.flatten()
        
        for idx, (step_name, drop_vars, is_batch_step) in enumerate(steps):
            # Correction
            if drop_vars:
                try:
                    if is_batch_step and batch_method == 'combat':
                        print(f"   -> ComBat on {drop_vars}")
                        sc.pp.combat(adata_temp, key=drop_vars[0])
                    else:
                        print(f"   -> Regress out {drop_vars}")
                        sc.pp.regress_out(adata_temp, drop_vars)
                except Exception as e:
                    print(f"   [Warning] {step_name} failed: {e}")

            # Recalculate PCA
            sc.tl.pca(adata_temp, n_comps=5)
            
            # Plot
            sc.pl.pca(adata_temp, color=self.cols['phenotype'], ax=axes_pca[idx], show=False, 
                      title=step_name, legend_loc='none' if idx < len(steps)-1 else 'right margin')

            # Evaluate Association
            pc_mat = pd.DataFrame(adata_temp.obsm['X_pca'], index=adata_temp.obs_names, 
                                  columns=[f'PC{i+1}' for i in range(5)])
            data_step = pd.concat([pc_mat, adata_temp.obs[eval_vars]], axis=1)

            for pc in pcs_to_check:
                step_vals = {'Step': step_name}
                for v in eval_vars:
                    # ANOVA for categorical, Spearman for numeric
                    if data_step[v].dtype.name in ['category', 'object']:
                        try:
                            model = smf.ols(f"{pc} ~ C({v})", data=data_step).fit()
                            anova = sm.stats.anova_lm(model, typ=2)
                            val = anova.loc[f'C({v})', 'sum_sq'] / anova['sum_sq'].sum()
                        except: val = 0
                    else:
                        try: val = abs(spearmanr(data_step[pc], data_step[v])[0])
                        except: val = 0
                    step_vals[v] = val if not np.isnan(val) else 0
                
                pc_results[pc]['val'].append(step_vals)
                pc_idx = int(pc.replace('PC','')) - 1
                pc_results[pc]['var'].append({'Step': step_name, 'Var_Ratio': adata_temp.uns['pca']['variance_ratio'][pc_idx]})

        plt.tight_layout(); plt.show()

        # Trend Plot
        print("Displaying Metric Evolution...")
        fig, axes = plt.subplots(len(pcs_to_check), 2, figsize=(16, 6 * len(pcs_to_check)))
        if len(pcs_to_check) == 1: axes = axes.reshape(1, -1)
        
        for i, pc in enumerate(pcs_to_check):
            df_vals = pd.DataFrame(pc_results[pc]['val']).set_index('Step')
            df_var = pd.DataFrame(pc_results[pc]['var']).set_index('Step')
            
            sns.heatmap(df_vals.T, cmap='Reds', annot=True, fmt='.2f', vmin=0, vmax=1, ax=axes[i,0])
            axes[i,0].set_title(f"{pc} Association")
            
            ax_twin = axes[i,1].twinx()
            df_var.plot(kind='bar', y='Var_Ratio', ax=axes[i,1], color='lightgray', alpha=0.4, legend=False)
            
            # Plot lines for each metric
            colors = sns.color_palette("husl", len(eval_vars))
            for j, v in enumerate(eval_vars):
                 if v in df_vals.columns:
                    ax_twin.plot(range(len(df_vals)), df_vals[v], label=v, color=colors[j], marker='o', lw=2)
            
            ax_twin.set_ylim(0, 1.1)
            ax_twin.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[i,1].set_title(f"{pc} Trend")
            
        plt.tight_layout(); plt.show()