library("ggplot2")
library("gridExtra")

# colNames <- c("ands" = "ands", "neighbors" = "neighbors", "time" = "time", "unbal" = "unbal", "reduction" = "reduction", "lsh_method" = "lsh_method", "class" = "classification")
graficar <- function(pathInput, pathOuput, colNames) {
  data <- read.csv(pathInput)
  g1 <- gplotYaxis(colNames, colNames[["time"]])
  g2 <- gplotYaxis(colNames, colNames[["reduction"]])
  g3 <- gplotYaxis(colNames, colNames[["class"]])
  g <- grid.arrange(g1, g2, g3)
  ggsave(pathOuput, g)
}

gplotYaxis <- function(colNames, yAxis) {
  labs <- labs(color = "Distribution", linetype = "LSH method", x = "ands")
  distribLabels <- scale_color_manual(labels = c("Balanced", "Unbalanced"), values = c("firebrick2", "cyan4"))
  lshLabels <- scale_linetype_manual(labels = c("Euclidean", "Hyperplanes"), values = c("solid", "dashed"))
  pointSize <- 1.5

  ggplot(data = data, aes_string(x = colNames[["ands"]], y = yAxis, linetype = colNames[["lsh_method"]], colour = colNames[["unbal"]])) +
  geom_line() +
  geom_point(size = pointSize) +
  facet_grid(reformulate(colNames[["neighbors"]]), labeller = label_both) +
  labs +
  distribLabels +
  lshLabels;
}
