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

class RNASeqQCPipeline:
    def __init__(self, adata, 
                 gene_type_col='GeneType', 
                 target_types=['protein_coding'],
                 phenotype_col='Type', 
                 batch_col='Batch_Granular',
                 gc_col='GC_Percent',
                 len_col='Length',
                 platelet_col='is_platelet'):
        # 원본 데이터 보호를 위해 copy 사용
        self.adata = adata.copy()
        self.cols = {
            'gene_type': gene_type_col,
            'phenotype': phenotype_col,
            'batch': batch_col,
            'gc': gc_col,
            'len': len_col,
            'platelet': platelet_col
        }
        self.target_gene_types = target_types
        # 분석 대상 기본 지표 리스트
        self.base_metrics = ['gc_bias_score', 'len_bias_score', 'platelet_score', 'log1p_total_counts']

    # --- [1] Layer 및 Metric 관리 로직 ---
    def set_layer(self, layer_name=None):
        """분석 대상 레이어를 교체하고 관련 PCA 정보를 초기화합니다."""
        if layer_name is None:
            print("--- Using default X ---")
        elif layer_name in self.adata.layers:
            print(f"--- [Switching Layer] Using: {layer_name} ---")
            self.adata.X = self.adata.layers[layer_name].copy()
        else:
            raise ValueError(f"Layer '{layer_name}' not found in AnnData.")

        # 레이어 교체 시 기존 PCA 및 이웃 계산 정보 삭제 (꼬임 방지)
        for key in ['X_pca', 'pca', 'PCs']:
            if key in self.adata.obsm: del self.adata.obsm[key]
            if key in self.adata.uns: del self.adata.uns[key]
            if key in self.adata.varm: del self.adata.varm[key]
        print(f"PCA and related metadata reset for layer: {layer_name if layer_name else 'default'}")

    def get_metric_keys(self, use_original=True):
        """Original 혹은 Residual 접두사가 붙은 실제 obs 컬럼명을 반환합니다."""
        prefix = "orig_" if use_original else "res_"
        keys = []
        for m in self.base_metrics:
            full_key = f"{prefix}{m}" if m != 'log1p_total_counts' else m
            if full_key in self.adata.obs.columns:
                keys.append(full_key)
        return keys

    # --- [2] 정제 및 데이터 준비 ---
    def prepare_data(self):
        """필터링 및 기본 로그 변환 등 분석을 위한 최소한의 데이터 정제"""
        print("--- [Step 1] Preparing Metadata & Filtering ---")
        
        if self.cols['phenotype'] in self.adata.obs.columns:
            self.adata = self.adata[~self.adata.obs[self.cols['phenotype']].isna()].copy()
        
        if self.cols['batch'] in self.adata.obs.columns:
            if self.adata.obs[self.cols['batch']].isna().any():
                if self.adata.obs[self.cols['batch']].dtype.name == 'category':
                    self.adata.obs[self.cols['batch']] = self.adata.obs[self.cols['batch']].cat.add_categories(['Unknown'])
                self.adata.obs[self.cols['batch']].fillna('Unknown', inplace=True)

        if self.cols['gene_type'] in self.adata.var.columns:
            mask = self.adata.var[self.cols['gene_type']].isin(self.target_gene_types)
            self.adata = self.adata[:, mask].copy()
            print(f"Filtered for {self.target_gene_types}: {self.adata.n_vars} genes left.")

        if self.cols['len'] in self.adata.var.columns:
            self.adata.var['log10_Length'] = np.log10(self.adata.var[self.cols['len']] + 1)
        
        if 'total_counts' not in self.adata.obs:
            self.adata.obs['total_counts'] = np.ravel(self.adata.X.sum(axis=1))
        self.adata.obs['log1p_total_counts'] = np.log1p(self.adata.obs['total_counts'])
        print("Preparation Done.")

    # --- [3] Bias 계산 로직 (Original vs Residual) ---
    def calculate_bias_metrics(self, is_original=False):
        status = "Original" if is_original else "Residual"
        prefix = "orig_" if is_original else "res_"
        print(f"--- [Step] Calculating {status} Bias Metrics ---")
        # --- 내부 보조 함수 ---
        def _calculate_sample_bias(adata_subset, feature_col, mode='spearman'):
            X = adata_subset.X
            feat_vals = adata_subset.var[feature_col].values.astype(float)
            scores = []
            for i in range(X.shape[0]):
                # 1. Sparse/Dense 대응 및 1차원 배열 변환
                sample_expr = np.ravel(X[i, :])
                # 2. 유효 데이터 필터링 (발현량이 0보다 큰 유전자만 사용하여 노이즈 제거)
                mask = sample_expr > 0
                if np.sum(mask) < 50: # 유효 유전자가 너무 적으면 0점 처리
                    scores.append(0)
                    continue
                valid_expr = np.log1p(sample_expr[mask])
                valid_feat = feat_vals[mask]
                if mode == 'lowess':
                    # [GC Bias용] 국소 회귀를 통한 곡선의 변화폭(Range) 계산
                    # frac=0.3: 데이터의 30%를 창(window)으로 사용
                    smoothed = lowess(valid_expr, valid_feat, frac=0.3, it=0)
                    bias_val = np.ptp(smoothed[:, 1]) # max - min
                    scores.append(bias_val)
                else:
                    # [Length Bias용] 선형/단조 상관관계 계산
                    corr, _ = spearmanr(valid_expr, valid_feat)
                    scores.append(corr if not np.isnan(corr) else 0)
            return np.array(scores)

        coding_mask = self.adata.var[self.cols['gene_type']] == 'protein_coding'
        subset_adata = self.adata[:, coding_mask]
        # 2. GC Content Bias (LOESS 방식 적용)
        if self.cols['gc'] in self.adata.var.columns:
            print(f"  > Computing GC bias using LOESS...")
            self.adata.obs[f'{prefix}gc_bias_score'] = _calculate_sample_bias(
                subset_adata, self.cols['gc'], mode='lowess'
            )
        # 3. Gene Length Bias (Spearman 방식 유지 - 길이는 보통 단조 증가하므로)
        len_col = 'log10_Length' if 'log10_Length' in self.adata.var.columns else None
        if len_col:
            print(f"  > Computing Length bias using Spearman...")
            self.adata.obs[f'{prefix}len_bias_score'] = _calculate_sample_bias(
                subset_adata, len_col, mode='spearman'
            )

        # 4. Platelet Score (기존 Scanpy 로직)
        if self.cols['platelet'] in self.adata.var.columns:
            platelet_genes = self.adata.var_names[self.adata.var[self.cols['platelet']]].tolist()
            if platelet_genes:
                print(f"  > Computing Platelet gene scores...")
                sc.tl.score_genes(self.adata, 
                                  gene_list=platelet_genes, 
                                  score_name=f'{prefix}platelet_score')

        print(f"Done: {status} metrics saved with prefix '{prefix}'.")

    # --- [4] 진단 및 분석 시각화 ---
    def run_pca_diagnostics(self, use_original_metrics=True):
        if 'X_pca' not in self.adata.obsm: 
            sc.tl.pca(self.adata)
        
        print("--- [Step 2-1] Scree Plot & Bias Heatmap ---")
        fig_top, ax_top = plt.subplots(1, 2, figsize=(20, 5), gridspec_kw={'width_ratios': [3, 7]})
        # Scree Plot
        var_ratios = self.adata.uns['pca']['variance_ratio']
        ax_top[0].plot(range(1, len(var_ratios)+1), var_ratios, 'o-k', alpha=0.7)
        ax_top[0].grid(alpha=0.1)
        ax_top[0].set_title("Scree Plot (Variance Explained)")
        ax_top[0].set_xlabel("Principal Component")
        ax_top[0].set_ylabel("Variance Ratio")
        # Bias Metrics Heatmap (Z-score)
        plot_metrics = self.get_metric_keys(use_original_metrics)
        if plot_metrics:
            df_plot = self.adata.obs[plot_metrics].copy()
            # Z-score 정규화 (샘플 간 비교를 위해)
            df_scaled = (df_plot - df_plot.mean()) / df_plot.std()
            sns.heatmap(df_scaled.T, cmap='RdBu_r', center=0, ax=ax_top[1], cbar_kws={'label': 'Z-score'})
            ax_top[1].set_title("Sample-specific Bias Metrics (Z-score)")
            ax_top[1].set_xlabel("Samples")
        
        plt.tight_layout()
        plt.show()
        print("--- [Step 2-2] PCA Scatter Plots ---")
        
        pc1_var = self.adata.uns['pca']['variance_ratio'][0]
        pc2_var = self.adata.uns['pca']['variance_ratio'][1]
        plot_keys = plot_metrics + [self.cols['phenotype'], self.cols['batch']]
        
        n_cols = 3
        n_rows = math.ceil(len(plot_keys) / n_cols)
        fig_pca, axes_pca = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 4.5*n_rows))
        axes_pca = axes_pca.flatten()
        
        for i, key in enumerate(plot_keys):
            if key not in self.adata.obs.columns:
                continue
            is_numeric = ('score' in key or 'count' in key)
            n_categories = 0 if is_numeric else len(self.adata.obs[key].unique())
            if is_numeric:
                legend_loc = 'right margin'
                cmap = 'RdBu_r'
            else:
                cmap = None
                if n_categories > 10:
                    legend_loc = 'on data'
                else:
                    legend_loc = 'right margin'
            
            sc.pl.pca(
                self.adata, 
                color=key, 
                ax=axes_pca[i], 
                show=False, 
                cmap=cmap, 
                size=120,
                frameon=True,
                legend_loc=legend_loc,
                legend_fontsize='small',
                legend_fontoutline=2 if legend_loc == 'on data' else None 
            )
            
            axes_pca[i].set_xlabel(f"PC1 ({pc1_var:.1%})")
            axes_pca[i].set_ylabel(f"PC2 ({pc2_var:.1%})")
            
        for j in range(i+1, len(axes_pca)):
            axes_pca[j].axis('off')
            
        plt.tight_layout()
        plt.show()

    def analyze_pc_associations(self, n_pcs=5, use_original_metrics=True):
        if 'X_pca' not in self.adata.obsm: sc.tl.pca(self.adata)
        pc_df = pd.DataFrame(self.adata.obsm['X_pca'][:, :n_pcs], 
                             columns=[f'PC{i+1}' for i in range(n_pcs)], index=self.adata.obs_names)
        
        cont_vars = self.get_metric_keys(use_original_metrics)
        cat_vars = [self.cols['phenotype'], self.cols['batch']]
        assoc_matrix = pd.DataFrame(index=cont_vars + cat_vars, columns=pc_df.columns)

        for col in cont_vars:
            for pc in pc_df.columns:
                corr, _ = spearmanr(self.adata.obs[col], pc_df[pc])
                assoc_matrix.loc[col, pc] = abs(corr)
        for col in cat_vars:
            # 데이터가 없거나, 모든 값이 동일(Unique count < 2)하면 ANOVA 불가
            if col not in self.adata.obs.columns or self.adata.obs[col].nunique() < 2:
                print(f"Skipping ANOVA for {col}: only one group present.")
                assoc_matrix.loc[col, :] = 0  # 0으로 채우거나 NaN 처리
                continue
                
            for pc in pc_df.columns:
                temp = pd.concat([self.adata.obs[col], pc_df[pc]], axis=1).dropna()
                temp.columns = ['G', 'V']
                
                try:
                    model = smf.ols('V ~ C(G)', data=temp).fit()
                    anova = sm.stats.anova_lm(model, typ=2)
                    assoc_matrix.loc[col, pc] = anova.loc['C(G)', 'sum_sq'] / anova['sum_sq'].sum()
                except Exception as e:
                    assoc_matrix.loc[col, pc] = 0

        plt.figure(figsize=(8, 6))
        sns.heatmap(assoc_matrix.astype(float), annot=True, cmap='Reds', vmin=0, vmax=1, fmt='.2f')
        plt.title(f"PC - Variable Association ({'Original' if use_original_metrics else 'Residual'})")
        plt.show()
        
    def analyze_partial_correlation(self, confounders=['gc_bias_score', 'len_bias_score', 'log1p_total_counts', 'platelet_score'], n_pcs=5, use_original_metrics=True):
        """Confounder를 통제했을 때 Phenotype과의 상관관계 유지 여부 확인"""
        print(f"--- [Step 6] Partial Correlation Analysis ({'Original' if use_original_metrics else 'Residual'}) ---")
        prefix = "orig_" if use_original_metrics else "res_"
        
        processed_conf = []
        for c in confounders:
            full_key = f"{prefix}{c}" if c != 'log1p_total_counts' else c
            if full_key in self.adata.obs.columns:
                processed_conf.append(full_key)
        
        all_conf = processed_conf + [self.cols['batch']]
        target = self.cols['phenotype']
        
        if 'X_pca' not in self.adata.obsm: sc.tl.pca(self.adata)
        pc_df = pd.DataFrame(self.adata.obsm['X_pca'][:, :n_pcs], columns=[f'PC{i+1}' for i in range(n_pcs)], index=self.adata.obs_names)
        data = pd.concat([pc_df, self.adata.obs[[target] + [c for c in all_conf if c in self.adata.obs.columns]]], axis=1)
        
        if data[target].dtype == 'object' or data[target].dtype.name == 'category':
            data['target_num'] = pd.factorize(data[target])[0]
        else: data['target_num'] = data[target]

        results = {}
        for pc in pc_df.columns:
            raw = data[[pc, 'target_num']].corr().iloc[0, 1]
            conf_terms = [f"C({c})" if data[c].dtype.name in ['category', 'object'] else c for c in all_conf if c in data.columns]
            if not conf_terms: partial = raw
            else:
                try:
                    res_pc = smf.ols(f"{pc} ~ {' + '.join(conf_terms)}", data=data).fit().resid
                    res_tg = smf.ols(f"target_num ~ {' + '.join(conf_terms)}", data=data).fit().resid
                    partial = spearmanr(res_pc, res_tg)[0]
                except: partial = 0
            results[pc] = {'Raw': abs(raw), 'Partial': abs(partial)}
            
        pd.DataFrame(results).T.plot(kind='bar', figsize=(8, 4), colormap='tab20')
        plt.grid(alpha=0.1); plt.title(f"Phenotype ({target}) Correlation: Raw vs Controlled"); plt.show()
        
    def analyze_variance_decomposition(self, n_pcs=5, use_original_metrics=True):
        covs = self.get_metric_keys(use_original_metrics) + [self.cols['phenotype'], self.cols['batch']]
        if 'X_pca' not in self.adata.obsm: sc.tl.pca(self.adata)
        pc_df = pd.DataFrame(self.adata.obsm['X_pca'][:, :n_pcs], columns=[f'PC{i+1}' for i in range(n_pcs)])
        data = pd.concat([pc_df, self.adata.obs[covs].reset_index(drop=True)], axis=1)
        
        importance = pd.DataFrame(index=covs, columns=pc_df.columns)
        terms = [f"C({c})" if data[c].dtype == 'object' or data[c].dtype.name == 'category' else c for c in covs]

        for pc in pc_df.columns:
            full_r2 = smf.ols(f"{pc} ~ {' + '.join(terms)}", data=data).fit().rsquared
            for i, cov in enumerate(covs):
                red_terms = [t for j, t in enumerate(terms) if i != j]
                red_r2 = smf.ols(f"{pc} ~ {' + '.join(red_terms)}", data=data).fit().rsquared if red_terms else 0
                importance.loc[cov, pc] = max(0, full_r2 - red_r2)
                
        importance.T.plot(kind='bar', stacked=True, colormap='Spectral', figsize=(10, 5))
        plt.grid(alpha=0.1)
        plt.title(f"Variance Decomposition ({'Original' if use_original_metrics else 'Residual'})")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.show()

    # --- [5] Cascade Analysis (계단식 보정 분석) ---
    def run_cascade_analysis(self, pcs_to_check=['PC1'], use_original_metrics=True, batch_method='combat'):
        print(f"--- [Step 8] Cascade Correction Analysis (Batch Method: {batch_method.upper()}) ---")
        
        adata_temp = self.adata.copy()
        prefix = "orig_" if use_original_metrics else "res_"
        
        # 변수 설정
        tech_vars = [f'{prefix}gc_bias_score', f'{prefix}len_bias_score', 'log1p_total_counts']
        platelet_vars = [f'{prefix}platelet_score'] if f'{prefix}platelet_score' in adata_temp.obs.columns else []
        batch_vars = [self.cols['batch']]
        
        # 분석 대상 변수 리스트업
        extra_vars = platelet_vars + batch_vars + [self.cols['phenotype']]
        target_vars = [v for v in (tech_vars + extra_vars) if v in adata_temp.obs.columns]

        # Steps 정의: (Step 이름, 제거할 변수들, 'Batch 단계 여부')
        steps = [
            ('1. Raw', [], False),
            ('2. -Tech(GC/L/D)', [v for v in tech_vars if v in adata_temp.obs.columns], False),
            ('3. -Platelet', platelet_vars, False),
            ('4. -Batch', batch_vars, True)  # True: 이 단계가 배치 보정 단계임을 표시
        ]
        
        pc_results = {pc: {'val': [], 'var': []} for pc in pcs_to_check}
        
        # --- (1) PCA Plot 시각화 준비 ---
        fig_pca, axes_pca = plt.subplots(2, 2, figsize=(14, 11))
        axes_pca = axes_pca.flatten()
        
        for idx, (step_name, drop_vars, is_batch_step) in enumerate(steps):
            
            # --- [보정 로직 분기점] ---
            if drop_vars:
                try:
                    # Case A: Batch 단계이고, 사용자가 'combat'을 선택했을 때
                    if is_batch_step and batch_method == 'combat':
                        batch_key = drop_vars[0]
                        print(f"   -> Applying ComBat on '{batch_key}'")
                        # ComBat 실행 (Scanpy 내장 함수)
                        sc.pp.combat(adata_temp, key=batch_key)
                        
                    # Case B: 연속형 변수 단계거나, 사용자가 'regression'을 선택했을 때
                    else:
                        print(f"   -> Regressing out: {drop_vars}")
                        sc.pp.regress_out(adata_temp, drop_vars)
                except Exception as e:
                    print(f"   [Warning] Correction failed at {step_name}: {e}")
            
            # PCA 재계산 및 시각화 (기존 로직 동일)
            sc.tl.pca(adata_temp, n_comps=5)
            sc.pl.pca(adata_temp, color=self.cols['phenotype'], ax=axes_pca[idx], show=False, 
                      title=f"{step_name}", size=100)
            
            # --- 지표 계산 (기존 로직 동일) ---
            pc_mat = pd.DataFrame(adata_temp.obsm['X_pca'], index=adata_temp.obs_names, 
                                  columns=[f'PC{i+1}' for i in range(adata_temp.obsm['X_pca'].shape[1])])
            data_step = pd.concat([pc_mat, adata_temp.obs[target_vars]], axis=1)
            
            for pc in pcs_to_check:
                step_vals = {'Step': step_name}
                for v in target_vars:
                    # 범주형: ANOVA
                    if data_step[v].dtype.name in ['category', 'object']:
                        try:
                            model = smf.ols(f"{pc} ~ C({v})", data=data_step).fit()
                            anova_table = sm.stats.anova_lm(model, typ=2)
                            val = anova_table.loc[f'C({v})', 'sum_sq'] / anova_table['sum_sq'].sum()
                        except: val = 0
                    # 연속형: Spearman
                    else:
                        try:
                            val = abs(spearmanr(data_step[pc], data_step[v])[0])
                        except: val = 0
                    step_vals[v] = val
                
                pc_results[pc]['val'].append(step_vals)
                pc_idx = int(pc.replace('PC','')) - 1
                pc_results[pc]['var'].append({'Step': step_name, 
                                              'Var_Ratio': adata_temp.uns['pca']['variance_ratio'][pc_idx]})

        plt.tight_layout()
        plt.show()

        # --- (2) Metric Evolution 시각화 (Heatmap + Trend) ---
        print("Displaying Metric Evolution (Spearman/ANOVA)...")
        fig, axes = plt.subplots(len(pcs_to_check), 2, figsize=(16, 6 * len(pcs_to_check)))
        if len(pcs_to_check) == 1: axes = axes.reshape(1, -1)
        colors = sns.color_palette("husl", len(target_vars))
        for i, pc in enumerate(pcs_to_check):
            df_vals = pd.DataFrame(pc_results[pc]['val']).set_index('Step')
            df_var = pd.DataFrame(pc_results[pc]['var']).set_index('Step')
            
            # Heatmap
            sns.heatmap(df_vals.T, cmap='Reds', annot=True, fmt='.2f', vmin=0, vmax=1, ax=axes[i,0])
            axes[i,0].set_title(f"{pc} Association (Rho/Eta^2)")
            axes[i,0].set_xticklabels(axes[i,0].get_xticklabels(), rotation=90)         
            
            ax_twin = axes[i,1].twinx()
            df_var.plot(kind='bar', y='Var_Ratio', ax=axes[i,1], color='lightgray', alpha=0.4, legend=False)
            axes[i,1].set_ylabel("PC Variance Ratio")
            
            for j, v in enumerate(target_vars):
                if v in df_vals.columns:
                    ax_twin.plot(range(len(df_vals)), df_vals[v], label=v, color=colors[j], marker='o', lw=2)
            
            ax_twin.set_ylabel("Association Score (0-1)")
            ax_twin.set_ylim(0, 1.1)
            ax_twin.legend(loc='upper left', bbox_to_anchor=(1.15, 1))
            axes[i,1].set_title(f"{pc} Trend: Signal vs Noise")
            
        plt.tight_layout()
        plt.show()

    def run_all(self, use_original=True):
        self.prepare_data()
        self.calculate_bias_metrics(is_original=True)
        self.run_pca_diagnostics(use_original_metrics=use_original)
        self.analyze_pc_associations(use_original_metrics=use_original)
        self.analyze_partial_correlation(use_original_metrics=use_original) # 포함됨
        self.analyze_variance_decomposition(use_original_metrics=use_original)
        self.run_cascade_analysis(use_original_metrics=use_original)
        return self.adata