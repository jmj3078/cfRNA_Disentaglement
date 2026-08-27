setwd("/project/cfRNA_NormativeModeling/MixedEffectsModeling/OutriderComparison/held_out_comparison")

for (f in list.files(".", pattern = "^z_test_.*\\.rds$")) {
  z <- readRDS(f)
  write.csv(z, sub("\\.rds$", ".csv", f))
}

for (f in list.files(".", pattern = "^cv_fold[0-9]+_full\\.rds$")) {
  d <- readRDS(f)
  fi <- sub("cv_fold([0-9]+)_full\\.rds", "\\1", f)
  write.csv(d$mu, sprintf("cv_fold%s_mu.csv", fi))
  write.csv(d$y, sprintf("cv_fold%s_y.csv", fi))
  write.csv(data.frame(gene = d$genes, theta = d$theta), sprintf("cv_fold%s_theta.csv", fi), row.names = FALSE)
}
cat("done\n")
