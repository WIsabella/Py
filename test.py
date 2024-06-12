list_score=[81,83,82,86,84,85]
list_name=['john','mary','lily','mark','June','April']

count=len(list_score)
for i in range(0,count):
    for j in range(i+1,count):
        if list_score[i]>list_score[j]:
            t=list_name[i]
            list_name[i]=list_name[j]
            list_name[j]=t
            n=list_score[i]
            list_score[i]=list_score[j]
            list_score[j]=n
for i in range(0,count):
    print("%s:%d"%(list_name[i],list_score[i]))
