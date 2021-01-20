require('mlbench')

write.csv(mlbench.2dnormals(200), '2dnormals.csv', row.names=FALSE)
write.csv(mlbench.xor(100), 'xor.csv', row.names=FALSE)
write.csv(mlbench.circle(100), 'circle.csv', row.names=FALSE)
write.csv(mlbench.spirals(100,sd=0.05), 'spirals.csv', row.names=FALSE)
