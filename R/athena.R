

athena.data = read.csv("../data/UA_AthenaData.csv", header = TRUE, sep = ",")
research.data = read.csv("../data/UA_ResearchData.csv", header=TRUE, sep = ",")
isLandline.data = split(research.data, research.data$isLandline)


# hist(athena.data$Total_B30Day_Calls, breaks = 30, freq = F, main = "Total_B30Day_Calls")

# func = function(x) { if(x < 1e3) { return(x)}}
# func1 = function(x) { return(as.numeric(x))}
# x = split(athena.data$Total_B30Day_Calls, )
# x = as.numeric(unlist(x))

hist(x, breaks = 30, freq = F, main = "Total_B30Day_Calls")
d=density(x)
plot(d)
