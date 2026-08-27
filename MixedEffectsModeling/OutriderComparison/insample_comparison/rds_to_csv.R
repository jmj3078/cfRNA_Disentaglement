files <- list.files(".", pattern="^z_test_.*\\.rds$")
for (f in files) {
  z <- readRDS(f)
  out <- sub("\\.rds$", ".csv", f)
  write.csv(z, out)
}
