

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

# Compute the Median Absolute Deviation (mad)
mads = apply(cp.numeric, 2, mad)
y = names(cp.numeric[mads == 0])
# If MAD=0, then var=0, so we remove this useless feature
novariance = c()
for(i in 1:length(mads)) {
  if(mads[i] == 0) {
    novariance = append(novariance, i)
  }
}
cp.numeric = cp.numeric[, -novariance]
mads = apply(cp.numeric, 2, mad) # Re-compute the MAD

# detect outliers using a modified-z-score approach
output = data.frame(cp.numeric) # Copy the data for rewriting
for(i in 1:ncol(cp.numeric)) {
  col = cp.numeric[,i]
  col.med = median(col)
  col.mad = mads[i]
  for(j in 1:nrow(cp.numeric)) {
    x = cp.numeric[j,i]
    output[j,i] = (0.6745 * (x - col.med)) / col.mad
  }
}

# Remove rows that contain outliers
outliers = c()
for(i in 1:nrow(output)) {
  for(j in 1:ncol(output)) {
    if(abs(output[i,j] > 3.5)) {
      outliers = append(outliers, i)
    }
  }
}
removed.outliers = output[-outliers,]

write.csv(removed.outliers, file = "test_transform1.csv", row.names = F)

write.csv(CPathena, file = "cellphone_athena_anomaly.csv", row.names = F)
write.csv(LLathena, file = "landline_athena_anomaly.csv", row.names = F)

eps = 0.000001
avg.callduration.transform = log(CPathena$Total_BToday_AvgDuration + eps)
outlier.avgcalldur = which(avg.callduration.transform > 10)
CPathena = CPathena[-c(outlier.avgcalldur),]

n30daycalls.transform = log(CPathena$Total_N30Day_Calls + eps)
outlier.30daycalls = c(which(n30daycalls.transform > 11), which(n30daycalls.transform < 0))
CPathena = CPathena[-c(outlier.30daycalls),]

write.csv(CPathena, file = "cellphone_athena_anomaly_trim.csv", row.names = F)

# hist(athena.data$Total_B30Day_Calls, breaks = 30, freq = F, main = "Total_B30Day_Calls")

# func = function(x) { if(x < 1e3) { return(x)}}
# func1 = function(x) { return(as.numeric(x))}
# x = split(athena.data$Total_B30Day_Calls, )
# x = as.numeric(unlist(x))

PCA.transformed = read.csv("../data/PCA_trans.csv", header=TRUE, sep = ",")
hist(PCA.transformed$values, freq=F)
