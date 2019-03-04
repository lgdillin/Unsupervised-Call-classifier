

athena.data = read.csv("../data/UA_AthenaData.csv", header = TRUE, sep = ",")

# 6 minute timeframe
# cut off at 0.99% (any # with calls >11 is a spam (cellphone), and >12 for landline)
research.data = read.csv("../data/UA_ResearchData.csv", header=TRUE, sep = ",")
isLandline.data = split(research.data, research.data$isLandline)

cellphone.data = isLandline.data$"FALSE"
landline.data = isLandline.data$"TRUE"

CPcount = aggregate(cellphone.data$call_time ~ cellphone.data$LineNumber, cellphone.data, FUN = "length")
LLcount = aggregate(landline.data$call_time ~ landline.data$LineNumber, landline.data, FUN = "length")

## take the high-vol callers from research data and see if there are any behaviors in athena


# CPanomaly = CPcount$`cellphone.data$call_time` > 11

# hist(athena.data$Total_B30Day_Calls, breaks = 30, freq = F, main = "Total_B30Day_Calls")

# func = function(x) { if(x < 1e3) { return(x)}}
# func1 = function(x) { return(as.numeric(x))}
# x = split(athena.data$Total_B30Day_Calls, )
# x = as.numeric(unlist(x))

hist(x, breaks = 30, freq = F, main = "Total_B30Day_Calls")
d=density(x)
plot(d)
