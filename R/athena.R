

athena.data = read.csv("../data/UA_AthenaData.csv", header = TRUE, sep = ",")

# 6 minute timeframe
# cut off at 0.99% (any # with calls >11 is a spam (cellphone), and >12 for landline)
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
CPanomaly = CPcount[CPcount$`cellphone.data$call_time` > 11,]
LLanomaly = LLcount[LLcount$`landline.data$call_time` > 12,]

CPathena = athena.data[CPanomaly$`cellphone.data$LineNumber`,]
LLathena = athena.data[LLanomaly$`landline.data$LineNumber`,]

write.csv(CPathena, file = "cellphone_athena_anomaly.csv", row.names = F)
write.csv(LLathena, file = "landline_athena_anomaly.csv", row.names = F)

# hist(athena.data$Total_B30Day_Calls, breaks = 30, freq = F, main = "Total_B30Day_Calls")

# func = function(x) { if(x < 1e3) { return(x)}}
# func1 = function(x) { return(as.numeric(x))}
# x = split(athena.data$Total_B30Day_Calls, )
# x = as.numeric(unlist(x))

PCA.transformed = read.csv("../data/PCA_trans.csv", header=TRUE, sep = ",")
hist(PCA.transformed$values, freq=F)
