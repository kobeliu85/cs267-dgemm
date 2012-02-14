

def create(type):
    
    
    import os.path
    files=os.listdir(os.getcwd())
    blasExt=[];
    for i in range(len(files)):

        pathSplit=os.path.splitext(files[i])
        name=pathSplit[0]
        ext=pathSplit[1]
        if name==('job-'+type):
            extLen=len(ext)
            letter=ext[0:2]
            blasExt.append(ext[2:extLen])

    blasExt.sort()
    extUse=blasExt[(len(blasExt)-1)]
    fileUse='job-'+type+letter+extUse
    fid=open(fileUse,'r')
    sizes=[]
    speed=[]
    read=1
    count=0
    while read:
        count=count+1
        line=fid.readline()
        if line[0:12]=='Description:':
            go=1
            title=line[13:len(line)]
            line=fid.readline()
            while go:
                line=fid.readline()
                if line[0:5]=='Size:':
                    ind=line.find('\t')
                    if not (ind==-1):
                        num=line[6:ind]
                    sizes.append(num)
                    indn=line.find('\n')
                    speed.append(line[(ind+10):indn])
                    
                else:
                    go=0
                    read=0
                    fid.close()
        if count>1000:
            read=0
    
    fwrite=open(type+'.dat','w')
    for i in range(len(sizes)):
        fwrite.write(sizes[i]+'\t'+speed[i]+'\n')
    fwrite.close()
    


names=['blas','naive','tuned']

for x in names:
    create(x)

