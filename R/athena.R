

athena.data = read.csv("../data/UA_AthenaData.csv", header = TRUE, sep = ",")

# 6 minute timeframe
research.data = read.csv("../data/UA_ResearchData.csv", header=TRUE, sep = ",")
isLandline.data = split(research.data, research.data$isLandline)

# Split the data into groups by landline or cellphone
cellphone.data = isLandline.data$"FALSE"
landline.data = isLandline.data$"TRUE"

# aggregate the data so phone numbers are unique
CPcount = aggregate(cellphone.data$call_time ~ cellphone.data$LineNumber, cellphone.data, FUN = "length")
LLcount = aggregate(landline.data$call_time ~ landline.data$LineNumber, landline.data, FUN = "length")

## take the high-vol callers from research data and see if there are any behaviors in athena
# filter for the anomalies (more than 11 phone calls in 6 minutes)
# cut off at 0.99% (any # with calls >11 is a spam (cellphone), and >12 for landline)
CPanomaly = CPcount[CPcount$`cellphone.data$call_time` > 11,]
LLanomaly = LLcount[LLcount$`landline.data$call_time` > 12,]

# Match the phone numbers from the resarch.data with the athena data
CPathena = athena.data[CPanomaly$`cellphone.data$LineNumber`,]
LLathena = athena.data[LLanomaly$`landline.data$LineNumber`,]

# Strip off the categorial data and the cellphone numbers
cp.numeric = CPathena[, -c(1,2)]
ll.numeric = LLathena[, -c(1,2)]

### CELLPHONE
# Compute the Median Absolute Deviation (mad)
cp.mads = apply(cp.numeric, 2, mad)
ll.mads = apply(ll.numeric, 2, mad)

cp.y = names(cp.numeric[cp.mads == 0])
ll.y = names(ll.numeric[ll.mads == 0])
# If MAD=0, then var=0, so we remove this useless feature
cp.novariance = c()
ll.novariance = c()
for(i in 1:length(cp.mads)) {
  if(cp.mads[i] == 0) {
    cp.novariance = append(cp.novariance, i)
  }
}
for(i in 1:length(ll.mads)) {
  if(ll.mads[i] == 0) {
    ll.novariance = append(ll.novariance, i)
  }
}
# Remove columns with no variance
cp.numeric = cp.numeric[, -cp.novariance]
ll.numeric = ll.numeric[, -ll.novariance]
# Re-compute the MAD for trimmed data
cp.mads = apply(cp.numeric, 2, mad) 
ll.mads = apply(ll.numeric, 2, mad)


### Cellphone data
# detect outliers using a modified-z-score approach
cp.output = data.frame(cp.numeric) # Copy the data for rewriting
for(i in 1:ncol(cp.numeric)) {
  cp.col = cp.numeric[,i]
  cp.col.med = median(cp.col)
  cp.col.mad = cp.mads[i]
  for(j in 1:nrow(cp.numeric)) {
    cp.x = cp.numeric[j,i]
    cp.output[j,i] = (0.6745 * (cp.x - cp.col.med)) / cp.col.mad
  }
}

# Remove rows that contain outliers
cp.outliers = c()
for(i in 1:nrow(cp.output)) {
  for(j in 1:ncol(cp.output)) {
    if(abs(cp.output[i,j] > 3.5)) {
      cp.outliers = append(cp.outliers, i)
    }
  }
}
cp.outliers = unique(cp.outliers)
# cp.removed.outliers = cp.output[-cp.outliers,]
cp.outlier.data = cp.numeric[cp.outliers,]
cp.removed.outliers = cp.numeric[-cp.outliers,]
write.csv(cp.removed.outliers, file = "cp_transformed.csv", row.names = F)
write.csv(cp.outlier.data, file = "cp_outliers.csv", row.names = F)


### LandLine data
# detect outliers using a modified-z-score approach
ll.output = data.frame(ll.numeric) # Copy the data for rewriting
for(i in 1:ncol(ll.numeric)) {
  ll.col = ll.numeric[,i]
  ll.col.med = median(ll.col)
  ll.col.mad = ll.mads[i]
  for(j in 1:nrow(ll.numeric)) {
    ll.x = ll.numeric[j,i]
    ll.output[j,i] = (0.6745 * (ll.x - ll.col.med)) / ll.col.mad
  }
}

# Remove rows that contain outliers
ll.outliers = c()
for(i in 1:nrow(ll.output)) {
  for(j in 1:ncol(ll.output)) {
    if(abs(ll.output[i,j] > 3.5)) {
      ll.outliers = append(ll.outliers, i)
    }
  }
}
ll.outliers = unique(ll.outliers)
# ll.removed.outliers = ll.output[-ll.outliers,]
ll.outlier.data = ll.numeric[ll.outliers,]
ll.removed.outliers = ll.numeric[-ll.outliers,]
write.csv(ll.removed.outliers, file = "ll_transformed.csv", row.names = F)
write.csv(ll.outlier.data, file = "ll_outliers.csv", row.names = F)

#Produce histograms
for(i in 1:ncol(cp.removed.outliers)) {
  hist(cp.removed.outliers[,i], freq = F, breaks = 30, main = names(cp.removed.outliers)[i])
}

for(i in 1:ncol(ll.removed.outliers)) {
  hist(ll.removed.outliers[,i], freq = T, breaks = 30, main = names(ll.removed.outliers)[i])
}

# RUN THESE IF YOU NEED
#install.packages("devtools")
#library(devtools)
#install_github("vqv/ggbiplot")
#library(ggbiplot)

# Run the prcomp function on each of the datasets
cp.pca = prcomp(cp.removed.outliers, center = T, scale. = T)
ll.pca = prcomp(ll.removed.outliers, center = T, scale. = T)
# ggbiplot(cp.pca, choices = c(1,4))
# ggbiplot(ll.pca)
write.csv(cp.pca$x, file = "cp_pca.csv", row.names = F)
write.csv(ll.pca$x, file = "ll_pca.csv", row.names = F)