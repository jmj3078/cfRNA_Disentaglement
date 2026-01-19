# 1. 라이브러리 로드
library(RUVSeq)
library(EDASeq)
library(edgeR)
library(dplyr)
library(tidyr)
library(limma)
# -----------------------------------------------------------------------------
# STEP 1: 데이터 로딩 및 병합
# -----------------------------------------------------------------------------
counts_a <- read.table("/project/cfRNA_Disentaglement/Data/RPM/RPM_Lab/RPM10_12_13_14_16_raw.csv", header=TRUE, row.names=1, sep=',')
counts_b <- read.table("/project/cfRNA_Disentaglement/Data/RPM/Compgen_Lab/Seq15.csv", header=TRUE, row.names=1, sep=',')
meta     <- read.table("/project/cfRNA_Disentaglement/Data/RPM/RPM_Lab/Meta.csv", header=TRUE, row.names=1, sep=',')
annot    <- read.table("/project/cfRNA_Disentaglement/Data/GECODEv49_Annot.tsv", header=TRUE, row.names=1, sep='\t')
palangodb <- read.table("/project/cfRNA_Disentaglement/Data/PalangoDB_CellTypeMarkers.tsv", header=TRUE, sep='\t')

# Count Matrix 병합 및 정렬
counts <- merge(counts_a, counts_b, by = 0)
rownames(counts) <- counts$Row.names
counts$Row.names <- NULL
counts <- counts[, order(colnames(counts))]

# -----------------------------------------------------------------------------
# STEP 2: 전처리 (Metadata & Gene Filtering)
# -----------------------------------------------------------------------------
# 2.1 Metadata 정제 (Python 로직 반영)
meta <- meta %>%
  mutate(
    Batch_Granular = paste(Sample.source, Seq_ID, Batch_tube, Batch_centrifuge, Batch_RNAext, sep = "-"),
    Subtype = ifelse(Subtype == Type | Subtype == "(NA)", NA, Subtype),
    Responder = ifelse(Responder == 1, "ICI-Responder", "ICI-Nonresponder")
  ) %>%
  unite("Type_Granular", Type, Subtype, Responder, sep = "_", remove = FALSE, na.rm = TRUE)

# 2.2 샘플 동기화
common_samples <- intersect(colnames(counts), rownames(meta))
counts <- counts[, common_samples]
meta   <- meta[common_samples, ]
stopifnot(all(colnames(counts) == rownames(meta)))

# 2.3 Protein Coding 유전자 필터링
pc_genes  <- rownames(annot)[annot$GeneType == "protein_coding"]
counts_pc <- counts[rownames(counts) %in% pc_genes, ]
annot_pc  <- annot[rownames(counts_pc), ] # 순서 동기화

valid_len_gc <- !is.na(annot_pc$Length) & annot_pc$Length > 0 & !is.na(annot_pc$GC_Percent)
counts_pc    <- counts_pc[valid_len_gc, ]
annot_pc     <- annot_pc[valid_len_gc, ]

keep_genes <- rowSums(counts_pc) > 0
counts_pc  <- counts_pc[keep_genes, ]
annot_pc <- annot_pc[keep_genes, ]

# 2.4 Platelet Control Genes 추출
platelet_syms <- palangodb[palangodb$cell.type == "Platelets", "official.gene.symbol"]
platelet_ids  <- rownames(annot_pc)[annot_pc$GeneName %in% platelet_syms]
control_genes <- intersect(platelet_ids, rownames(counts_pc))

message(paste("Filtered PC Genes:", nrow(counts_pc), "| Platelet Genes:", length(control_genes)))

# -----------------------------------------------------------------------------
# STEP 3: 정규화 수행
# -----------------------------------------------------------------------------
results_list <- list()

clean_matrix <- function(mat, label = "Unknown") {
    mat[is.na(mat)] <- 0
    mat[is.infinite(mat)] <- 0
    return(mat)
}

# 3.1 Raw & Basic
results_list[["Raw"]] <- counts_pc

dge <- DGEList(counts = as.matrix(counts_pc))
dge <- calcNormFactors(dge, method = "TMM")
results_list[["TMM_log2"]] <- cpm(dge, normalized.lib.sizes = TRUE, log = TRUE, prior.count = 1)

# 3.2 Length-based (FPKM, TPM)
gene_len_kb <- annot_pc$Length / 1000
results_list[["FPKM_log2"]] <- clean_matrix(rpkm(as.matrix(counts_pc), gene.length = annot_pc$Length, log = TRUE, prior.count = 1))

rpk <- counts_pc / gene_len_kb
scale_factor <- colSums(rpk)
scale_factor[scale_factor == 0] <- 1
tpm_val <- t(t(rpk) * 1e6 / scale_factor)
results_list[["TPM_log2"]] <- clean_matrix(log2(tpm_val + 1))


# 3.3 EDASeq
sample_info <- data.frame(conditions = rep("All", ncol(counts_pc)), 
                          row.names = colnames(counts_pc))

eda_set <- newSeqExpressionSet(as.matrix(counts_pc),
                              featureData = data.frame(gc = annot_pc$GC_Percent, length = annot_pc$Length,
                                row.names = rownames(counts_pc)))

results_list[["EDA_GC_Only"]]  <- clean_matrix(log2(normCounts(withinLaneNormalization(eda_set, "gc", which="full")) + 1))
results_list[["EDA_Len_Only"]] <- clean_matrix(log2(normCounts(withinLaneNormalization(eda_set, "length", which="full")) + 1))
set_within <- withinLaneNormalization(withinLaneNormalization(eda_set, "gc", which="full"), "length", which="full")

results_list[["EDA_Within_Full"]] <- clean_matrix(log2(normCounts(set_within) + 1))
results_list[["EDA_Scale_Only"]] <- clean_matrix(log2(normCounts(betweenLaneNormalization(eda_set, which="upper")) + 1))
results_list[["EDA_Full_All"]]   <- clean_matrix(log2(normCounts(betweenLaneNormalization(set_within, which="upper")) + 1))

# 3.4 RUVg (Nature.)
input_matrix <- results_list[["TMM_log2"]]

for (k in c(1, 2, 3)) {
    message(paste0(">>> Processing RUVg correction with k = ", k, "..."))
    set_ruvg <- RUVg(input_matrix, control_genes, k = k, isLog = TRUE, center = TRUE)
    W_platelet <- set_ruvg$W
    key_ruvg <- paste0("RUVg_Platelet_k", k)
    results_list[[key_ruvg]] <- clean_matrix(set_ruvg$normalizedCounts)
    key_full <- paste0("Proposed_Full_k", k)
    results_list[[key_full]] <- clean_matrix(
        removeBatchEffect(results_list[["EDA_Full_All"]], covariates = W_platelet)
    )
    write.csv(W_platelet, file.path(out_dir, paste0("W_factor_k", k, ".csv")))
  }
# -----------------------------------------------------------------------------
# STEP 4: 데이터 저장
# -----------------------------------------------------------------------------
out_dir <- "/project/cfRNA_Disentaglement/Data/RPM/Processed/"
if(!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# 일괄 저장 루프
for(name in names(results_list)) {
  file_path <- paste0(out_dir, "Norm_", name, ".csv")
  write.csv(as.data.frame(results_list[[name]]), file = file_path, row.names = TRUE)
  message(paste("Saved:", name))
}

# 메타데이터 및 W 요인 별도 저장
write.csv(meta, paste0(out_dir, "Final_Metadata.csv"))
write.csv(annot_pc, paste0(out_dir, "Final_Gene_Annotation.csv"))

message("--- All normalization steps completed successfully ---")


# 