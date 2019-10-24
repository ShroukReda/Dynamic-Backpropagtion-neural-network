import matplotlib.pyplot as plt
import numpy as np
import glob
import math
def Sigmoid(x):
    x=1/1+ math.exp(-x)
    return x
def Hyberpolic(x):
    x=math.tanh(x)
    return x
def loading():
    MydataSet = open("D:\CS\Semester2/Neural Networks\CI [CS-2019]\IrisData.txt", "r")
    MydatasetLine = MydataSet.readlines()
    i = 1
    C1 = [[None] * 4] * 51
    C2 = [[None] * 4] * 51
    C3 = [[None] * 4] * 51
    # ============================================================
    x11 = [None] * 51
    x21 = [None] * 51
    x31 = [None] * 51
    x41 = [None] * 51

    x12 = [None] * 51
    x22 = [None] * 51
    x32 = [None] * 51
    x42 = [None] * 51

    x13 = [None] * 51
    x23 = [None] * 51
    x33 = [None] * 51
    x43 = [None] * 51
    firstClass = 1
    secClass = 1
    thirdClass = 1
    while (i < MydatasetLine.__len__()):
        feature = MydatasetLine[i].split(',')
        x = i / 50
        if x <= 1:
            C1[firstClass] = feature
            firstClass += 1
        elif x > 1 and x <= 2:
            C2[secClass] = feature
            secClass += 1
        elif x > 2 and x <= 3:
            C3[thirdClass] = feature
            thirdClass += 1
        i = i + 1
    # ==================================================
    z = 1
    for j in range(1, 51):
        x11[z] = C1[j][0]
        x21[z] = C1[j][1]
        x31[z] = C1[j][2]
        x41[z] = C1[j][3]

        x12[z] = C2[j][0]
        x22[z] = C2[j][1]
        x32[z] = C2[j][2]
        x42[z] = C2[j][3]

        x13[z] = C3[j][0]
        x23[z] = C3[j][1]
        x33[z] = C3[j][2]
        x43[z] = C3[j][3]
        z = z + 1
    AllFeatures=[x11,x21,x31,x41,x12,x22,x32,x42,x13,x23,x33,x43] #feature then class
    MyClasses=[C1,C2,C3]
    for i in range(1, 51):
        MyClasses[0][i][4] = [1,0,0]
        MyClasses[1][i][4] = [0,1,0]
        MyClasses[2][i][4] = [0,0,1]
    return AllFeatures,MyClasses

def Train(Features,Classes,Epoch,Rate,NumOfHidd,NumofNeurons,Bias,Func):
    Epoch=int(Epoch)
    Rate=float(Rate)
    Dict={}
    n=0
    for i in range(1,int(NumOfHidd)+2):
            if i==1:
                InputHidden =[]
                for k in range(1,6): #input w hidden
                    a=np.random.rand(1,int(NumofNeurons[i-1]))
                    InputHidden.append(a)
                if Bias == "Without Bias":
                    InputHidden[4] = np.zeros((1,int(NumofNeurons[i-1]))) #bias
                Dict[n]=InputHidden
                n+=1
            elif i==int(NumOfHidd)+1: #hidden w output
                HiddOut=[]
                for z in range(1,int(NumofNeurons[i-2])+2):
                    a=np.random.rand(1,3)
                    HiddOut.append(a)
                if Bias == "Without Bias":
                    HiddOut[int(NumofNeurons[i - 2])] = np.zeros((1,3))
                Dict[n] = HiddOut
                n += 1
            else: #hidd w hidd
                weights=[]
                for j in range(1, int(NumofNeurons[i-2]) + 2):
                    a=np.random.rand(1,int(NumofNeurons[i-1]))
                    weights.append(a)
                if Bias == "Without Bias":
                    weights[int(NumofNeurons[i-2])]=np.zeros((1,int(NumofNeurons[i-1])))
                Dict[n]=weights
                n+=1
    while Epoch!=0:
        for p in range(1, 91):
            if p <= 30:
                Class = np.array([float(Features[0][p]), float(Features[1][p]), float(Features[2][p]), float(Features[3][p]), 1])
                L=0
            elif p>30 and p<=60:
                Class = np.array([float(Features[4][p-30]), float(Features[5][p-30]),float(Features[6][p-30]),float(Features[7][p-30]),1])
                L=1
            else:
                Class = np.array([float(Features[8][p-60]), float(Features[9][p-60]), float(Features[10][p-60]), float(Features[11][p-60]), 1])
                L=-1
            NetValues,Dic=Forward(Dict,Class,NumOfHidd,NumofNeurons,Bias,Func)
            Grads=Backward(NetValues,Classes,Class,NumOfHidd,NumofNeurons,Bias,Func,Dic,L)#backWard
            D=Update(Grads,Class,Dic,Rate,NetValues,NumOfHidd,NumofNeurons)#update weights
        Epoch=Epoch-1
    return Classes,NumOfHidd,NumofNeurons,Bias,Func,D

def Update(Grads,Class,Dic,Rate,Net,NumOfHidd,NumofNeurons):
    Rate=int(Rate)
    N=0
    v =int(NumofNeurons[N])  #neurons x awl hidden
    n = []
    c=0
    L=[]
    count=0
    length=len(Grads)
    for x in reversed(Grads):
        if count == (length - 3):
            L.append(n)
            break
        else:
            if c!=v:
                n.append(x) #awl hidd
                Grads.remove(x)
                c+=1
                count += 1
            else:
                N+=1 #tb2a tany hidd
                v = int(NumofNeurons[N]) #3dd lly x tany hidd
                L.append(n)
                n=[]
                n.append(x)
                Grads.remove(x)
                c=1
                count += 1
    q = 0
    o = 0
    for key, value in Dic.items():
        if key == 0: #weights mn l input to hidd
            I = len(L[0])-1 #num of neurons of first hidden law 2 => 0 , 1 G[1] 3l4an na md5lahom bel m42lb kano n0 n1 2rethom n1 n0
            for o in range(int(NumofNeurons[q])):
                for h in range(5): # 5 3l4an l bias
                    ss=value[h][0][o]
                    cc=(Class[h])
                    ll=L[0][I]
                    value[h][0][o] = ss+(cc*Rate*ll)
                I-=1 #tany neuron
            q += 1
        elif key == len(Dic) - 1: #mn l hidden to output
            d = 0
            for a in range(len(NumofNeurons)):
                d += int(NumofNeurons[a])
            for t in range(3):
                I = len(Grads) - 1
                f = d - int(NumofNeurons[len(NumofNeurons) - 1])
                for u in range(int(NumofNeurons[len(NumofNeurons) - 1])):
                    ss=value[u][0][t]
                    value[u][0][t]= ss+((Net[f])*Rate*Grads[I])
                    f += 1
                g = value[int(NumofNeurons[key - 1])][0][t]
                value[int(NumofNeurons[key - 1])][0][t]= g+(Rate*Grads[I])
                I -= 1
        else: #hidden w hidden
            for c in range(int(NumofNeurons[q])):
                k = key
                Sum = 0
                while k <= len(NumofNeurons) - 1:
                    Sum += int(NumofNeurons[k])
                    k += 1
                Sum += 3
                x = len(Net) - (Sum) - int(NumofNeurons[key - 1])
                I = len(L[key]) - 1
                for z in range(int(NumofNeurons[key - 1])):
                    ss=value[z][0][c]
                    value[z][0][c]= ss+((Net[x])*Rate*L[key][I])
                    x += 1
                g = value[int(NumofNeurons[key - 1])][0][c]
                value[int(NumofNeurons[key - 1])][0][c]=g+(Rate*L[key][I])
                I-=1
            q += 1
    return Dic

def Backward(Net,Targets,Class1,NumOfHidd,NumofNeurons,Bias,Func,Dic,L):
    if L < 0:
        L=2
    Gradinats=[]
    Y=[]
    OutIndex=(len(Net)-3)
    G1=(((Targets[L][1][4][0])-Net[OutIndex])*Net[OutIndex]*(1-Net[OutIndex]))
    Y.append(G1)
    OutIndex+=1
    G2 = (((Targets[L][1][4][1]) - Net[OutIndex]) * Net[OutIndex] * (1 - Net[OutIndex]))
    Y.append(G2)
    OutIndex+=1
    G3 = (((Targets[L][1][4][2]) - Net[OutIndex] )* Net[OutIndex] * (1 - Net[OutIndex]))
    Y.append(G3)
    i=int(NumOfHidd)
    n=1
    Ac=(int(NumofNeurons[len(NumofNeurons)-1])+3) #index awl neuron x l hidden layer lly 2bl l output mslun 2 2 w 3 out => 7-5=2
    Lastindex=len(Net)-Ac
    while(i>0):
        if i==int(NumOfHidd): #hidden before output
            for j in range(int(NumofNeurons[len(NumofNeurons)-n])):
                S = 0
                for k in range(3):
                    S+=(Y[k]*Dic[i][j][0][k]) #sum of w*G in the output neurons
                N=Net[Lastindex]
                G=S*N*(1-N)
                Y.append(G)
                Lastindex+=1
            i-=1
            x=n
            Ac+=int(NumofNeurons[len(NumofNeurons) - x])  #walahy ma fakra :"(
            Lastindex = len(Net) - Ac
            n+=1
        else: #other hiddens
            t=len(Y)
            for j in range(int(NumofNeurons[len(NumofNeurons) - n])):
                S=0
                z = (t - int(NumofNeurons[len(NumofNeurons) - x]))
                for k in range(int(NumofNeurons[len(NumofNeurons) - x])):
                    S += (Y[z] * Dic[i][j][0][k])
                    z+=1
                N = Net[Lastindex]
                G = S * N * (1 - N)
                Y.append(G)
                Lastindex+=1
            i-=1
            x = n
            Ac += int(NumofNeurons[len(NumofNeurons) - x])
            Lastindex = len(Net) - Ac
            n += 1
    Gradinats=Y
    return Gradinats

def Forward(Dict,Class1,NumOfHidd,NumofNeurons,Bias,Func):
    q = 0
    o = 0
    Nets=[]
    for key, value in Dict.items():
        if key==0: #firsr hidden
            for o in range(int(NumofNeurons[q])):
                v=0
                for h in range(5):
                    v+=((Class1[h])*(value[h][0][o])) #input*W
                if Func == "Sigmoid":
                    Yhat1 = Sigmoid(v)
                elif Func == "Hyperbolic Tangent":
                    Yhat1 = Hyberpolic(v)
                Nets.append(Yhat1)
            q+=1
        elif key == len(Dict)-1:
            d=0
            for a in range(len(NumofNeurons)):
                d+=int(NumofNeurons[a])
            for t in range(3):
                v=0
                f = d - int(NumofNeurons[len(NumofNeurons) - 1])
                for u in range(int(NumofNeurons[len(NumofNeurons)-1])):
                    v+=((Nets[f])*(value[u][0][t]))#nets from hidden *w
                    f+=1
                g = value[int(NumofNeurons[key -1])][0][t] #bias
                v+=g
                if Func == "Sigmoid":
                    Yhat1 = Sigmoid(v)
                elif Func == "Hyperbolic Tangent":
                    Yhat1 = Hyberpolic(v)
                Nets.append(Yhat1)
        else:
            d = len(Nets) - int(NumofNeurons[key - 1])
            for c in range(int(NumofNeurons[q])):
                v=0
                x = d
                for z in range(int(NumofNeurons[key-1])):
                    v+=((Nets[x])*(value[z][0][c]))
                    x+=1
                g = value[int(NumofNeurons[key -1])][0][c]
                v+=g
                if Func == "Sigmoid":
                    Yhat1 = Sigmoid(v)
                elif Func == "Hyperbolic Tangent":
                    Yhat1 = Hyberpolic(v)
                Nets.append(Yhat1)
            q+=1
    Check=Dict
    return Nets,Dict

def Test (Classes,NumOfHidd,NumofNeurons,Bias,Func,Features,DD):
    Right1=0
    wrong1=0
    Right2=0
    wrong2=0
    Right3=0
    wrong3=0
    for i in range(31, 91):
        if i >= 31 and i<51:
            Class = np.array([float(Features[0][i]), float(Features[1][i]), float(Features[2][i]), float(Features[3][i]), 1])
            L = 0
        elif i >= 51 and i <= 70:
            Class = np.array([float(Features[4][i - 20]), float(Features[5][i - 20]), float(Features[6][i - 20]),float(Features[7][i - 20]), 1])
            L = 1
        else:
            Class = np.array([float(Features[8][i - 40]), float(Features[9][i - 40]), float(Features[10][i - 40]),float(Features[11][i - 40]), 1])
            L = -1
        if L < 0:
            L = 2
        N,D=Forward(DD,Class,NumOfHidd,NumofNeurons,Bias,Func)
        x=len(N)-3
        Y1=N[x]
        x+=1
        Y2=N[x]
        x+=1
        Y3=N[x]
        m=max(Y1,Y2,Y3)
        c=1
        if m==Y1 and c==Classes[L][1][4][0]:
            Right1+=1
        if m==Y2 and c==Classes[L][1][4][1]:
            Right2+=1
        if m==Y3 and c==Classes[L][1][4][2]:
            Right3+=1
        if c==Classes[L][1][4][0] and m!=Y1 :
            wrong1+=1
        if c==Classes[L][1][4][1] and m!=Y2 :
            wrong2+=1
        if c==Classes[L][1][4][2] and m!=Y3 :
            wrong3+=1
    sum=Right1+Right2+Right3
    accuracy=(sum/60)*100
    M=ConMatrix(Right1,wrong1,Right2,wrong2,Right3,wrong3)
    return accuracy,M

def ConMatrix(R1,W1,R2,W2,R3,W3):
    C1=[R1,W2]
    C2=[W1,R2]
    C3 = [W3, R3]
    Mat=[C1,C2,C3]
    return Mat
