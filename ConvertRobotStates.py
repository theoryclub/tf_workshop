def loadData(file):
    #read file
    csvFile=open(file)
    lines=csvFile.read().splitlines()
    csvFile.close()
    splitLines=[]
    for line in lines:
        split=line.split('\t')
        if split!=['']:
            if split[0]=='':
                split=split[1:]
            splitLines+=[split]
    return splitLines
   
splitLines=loadData('/home/willie/workspace/TensorFlowWorkshop/data/RobotActions')     

convertedLines=''
line=[0.0,0.0,0.0,0.0,0.0,0.0]
for splitLine in splitLines:
    if len(splitLine)==1:
        for ind in range(len(line)):
            convertedLines+=str(line[ind]/15.0)+','
        if 'normal' in splitLine[0]:
            convertedLines+=str(0.0)+'\n'
        elif 'collision' in splitLine[0]:
            convertedLines+=str(1.0)+'\n'
        elif 'obstruction' in splitLine[0]:
            convertedLines+=str(2.0)+'\n'
        line=[0.0,0.0,0.0,0.0,0.0,0.0]      
    else:
        for ind in range(0, len(splitLine)):
            line[ind]+=float(splitLine[ind])
  
text_file = open("/home/willie/workspace/TensorFlowWorkshop/data/RobotStates.csv", "w")
text_file.write(convertedLines)
text_file.close()          