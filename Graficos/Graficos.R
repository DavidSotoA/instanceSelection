
graficar <-function(df, dir, title){
    vars <- c("reduction", "bucket", "time")
    ws <- unique((df[,"w"]))
    rows <- floor(length(ws)/2)
    cols <- rows
    while ((cols * rows) < length(ws)) {
      cols <- cols + 1
    }
    for (var in vars) {
      jpeg(paste(dir, '/',title, '_',var, '.jpg'))
      slicedfVar <- df[, c("w","ansd", "ors", var)]
      par(mfrow = c(rows, cols))
      for (w in ws) {
        slicedf <- slicedfVar[slicedfVar$w == w,]
	      x <- unique(as.vector(slicedf[, c("ands")]))
        w_title <- unique(as.vector(slicedf[, c("w")]))
        slicedf <- slicedf[, c("ors", "reduction")]
      }
    }
}



 ggplot(data = df, aes(x = ands, y = reduction, group = ors, colour= ors)) 
 + geom_line() + geom_point() 
 + facet_wrap(~w)
