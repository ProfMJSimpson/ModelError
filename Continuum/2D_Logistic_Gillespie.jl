using Plots
using Random
using NLopt
using Distributions
using .Threads
using Roots, Loess, DifferentialEquations
using Interpolations,CubicSplines,DSP,Dierckx
gr()
function Continuum(T,C0,PP,PD)
    QE=zeros(T+1)
    QE[1]=C0
for i in 2:T+1
    QE[i]=(C0*(PD-PP)*exp(-i*(PD-PP)))/((PD-PP)+C0*PP*(1.0-exp(-i*(PD-PP)))) #This is the exact solution
    end
return QE
end





function Stochastic(LX,LY,Tend,PM,PP,PD,C0)
Q0=Int(round(C0*LX*LY))
A0=zeros(LX,LY)
t=0.0
count=0

while count < Q0
    II =rand(1:LX)
    JJ=rand(1:LY)
    if A0[II,JJ]==0.0
        count=count+1
        A0[II,JJ]=1.0
    end
end  
lge=Int(1e8)
Density_record = zeros(lge)
Time_record = zeros(lge)

count = 1
Density_record[count] = Q0/(LX*LY)
Time_record[count]=t

while t < Tend

    Q0=Int(sum(A0))
    a0=(PM+PP+PD)*Q0
    tau = log(1/rand())/a0 
    t = t + tau
    RR = a0*rand()

    II=0
    JJ=0
    find = 0
    while find < 1
    II =rand(1:LX)
    JJ=rand(1:LY)
        if A0[II,JJ] == 1.0
            find = 1
        end
    end   


    if  RR <= PM*Q0 #give an agent an opportunity to movement
        

        if  II>1 && II<LX && JJ > 1 && JJ < LY  #Internal
        R =rand()
           if A0[II-1,JJ] == 0.0 && R > 0 && R<=1/4 
            A0[II,JJ]=0.0
            A0[II-1,JJ]=1.0
            elseif A0[II+1,JJ] == 0.0 && R > 1/4 && R<=2/4 
            A0[II,JJ]=0.0
            A0[II+1,JJ]=1.0
            elseif A0[II,JJ-1] == 0.0 && R > 2/4 && R<=3/4 
            A0[II,JJ]=0.0
            A0[II,JJ-1]=1.0
            elseif A0[II,JJ+1] == 0.0 && R > 3/4 && R<=4/4 
            A0[II,JJ]=0.0
            A0[II,JJ+1]=1.0
            end
        
        elseif II>1 && II<LX && JJ == 1   #bottom row
        R =rand()
        if A0[II-1,JJ] == 0.0 && R > 0 && R<=1/4 
            A0[II,JJ]=0.0
            A0[II-1,JJ]=1.0
            elseif A0[II+1,JJ] == 0.0 && R > 1/4 && R<=2/4 
            A0[II,JJ]=0.0
            A0[II+1,JJ]=1.0
            elseif A0[II,JJ+1] == 0.0 && R > 3/4 && R<=4/4 
            A0[II,JJ]=0.0
            A0[II,JJ+1]=1.0
            end

        elseif  II>1 && II<LX && JJ ==LY  #top row 
        R =rand()
        if A0[II-1,JJ] == 0.0 && R > 0 && R<=1/4 
            A0[II,JJ]=0.0
            A0[II-1,JJ]=1.0
            elseif A0[II+1,JJ] == 0.0 && R > 1/4 && R<=2/4 
            A0[II,JJ]=0.0
            A0[II+1,JJ]=1.0
            elseif A0[II,JJ-1] == 0.0 && R > 2/4 && R<=3/4 
            A0[II,JJ]=0.0
            A0[II,JJ-1]=1.0
            end
        
        elseif  II==1 && JJ > 1 && JJ < LY  #Left column
        R =rand()
            if A0[II+1,JJ] == 0.0 && R > 1/4 && R<=2/4 
            A0[II,JJ]=0.0
            A0[II+1,JJ]=1.0
            elseif A0[II,JJ-1] == 0.0 && R > 2/4 && R<=3/4 
            A0[II,JJ]=0.0
            A0[II,JJ-1]=1.0
            elseif A0[II,JJ+1] == 0.0 && R > 3/4 && R<=4/4 
            A0[II,JJ]=0.0
            A0[II,JJ+1]=1.0
            end
        
        elseif  II == LX && JJ > 1 && JJ < LY  #right column 
        R =rand()
            if A0[II-1,JJ] == 0.0 && R > 0 && R<=1/4 
            A0[II,JJ]=0.0
            A0[II-1,JJ]=1.0
            elseif A0[II,JJ-1] == 0.0 && R > 2/4 && R<=3/4 
            A0[II,JJ]=0.0
            A0[II,JJ-1]=1.0
            elseif A0[II,JJ+1] == 0.0 && R > 3/4 && R<=4/4 
            A0[II,JJ]=0.0
            A0[II,JJ+1]=1.0
            end




        end




    elseif RR > PM*Q0 && RR <= (PM+PP)*Q0 # then let the agent attempt to proliferate

        
        if II>1 && II<LX && JJ > 1 && JJ < LY #Central
        R =rand()
            if A0[II-1,JJ] == 0.0 && R > 0 && R<=1/4 
            A0[II-1,JJ]=1.0
            elseif A0[II+1,JJ] == 0.0 && R > 1/4 && R<=2/4 
            A0[II+1,JJ]=1.0
            elseif A0[II,JJ-1] == 0.0 && R > 2/4 && R<=3/4 
            A0[II,JJ-1]=1.0
            elseif A0[II,JJ+1] == 0.0 && R > 3/4 && R<=4/4 
            A0[II,JJ+1]=1.0
            end

        elseif II>1 && II<LX && JJ == 1 #bottom row
        R =rand()
            if A0[II-1,JJ] == 0.0 && R > 0 && R<=1/4 
            A0[II-1,JJ]=1.0
            elseif A0[II+1,JJ] == 0.0 && R > 1/4 && R<=2/4 
            A0[II+1,JJ]=1.0
            elseif A0[II,JJ+1] == 0.0 && R > 3/4 && R<=4/4 
            A0[II,JJ+1]=1.0
            end

        elseif II>1 && II<LX && JJ == LY #top 
            R =rand()
            if A0[II-1,JJ] == 0.0 && R > 0 && R<=1/4 
            A0[II-1,JJ]=1.0
            elseif A0[II+1,JJ] == 0.0 && R > 1/4 && R<=2/4 
            A0[II+1,JJ]=1.0
            elseif A0[II,JJ-1] == 0.0 && R > 2/4 && R<=3/4 
            A0[II,JJ-1]=1.0
            end

        elseif II==1 && JJ > 1 && JJ < LY #left column
        R =rand()
            if A0[II+1,JJ] == 0.0 && R > 1/4 && R<=2/4 
            A0[II+1,JJ]=1.0
            elseif A0[II,JJ-1] == 0.0 && R > 2/4 && R<=3/4 
            A0[II,JJ-1]=1.0
            elseif A0[II,JJ+1] == 0.0 && R > 3/4 && R<=4/4 
            A0[II,JJ+1]=1.0
            end

        elseif II==LX && JJ > 1 && JJ < LY #right column
        R =rand()
        if A0[II-1,JJ] == 0.0 && R > 0 && R<=1/4 
            A0[II-1,JJ]=1.0
            elseif A0[II,JJ-1] == 0.0 && R > 2/4 && R<=3/4 
            A0[II,JJ-1]=1.0
            elseif A0[II,JJ+1] == 0.0 && R > 3/4 && R<=4/4 
            A0[II,JJ+1]=1.0
            end
                                
        end

    elseif RR > (PM+PP)*Q0 && RR <= (PM+PP+PD)*Q0 # then kill the agent
        A0[II,JJ]=0.0



    Q0=Int(sum(A0)) 
    end

#println(Q0)
Density_record[count+1] = Q0/(LX*LY)
Time_record[count+1]=t
count = count + 1
end

density=zeros(count+1)
time=zeros(count+1)

density = Density_record[1:count]
time = Time_record[1:count]

f1=LinearInterpolation(time,density)
Q=f1(0:1:T)

return Q

end

    
       
       




LX=100 #this is the side length of the domain - leave this - but you can make the results smoother by increasing LX
PM=1.0 #Leave this fixed - this is the movement probability
PP=1.0/10  #Vary this - this is the proliferation probability
PD=1.0/20 #Vary this - this is the proliferation probability
C0=0.1 #Vary this - this is the initial density
T=500 #This is an estimate of the maximum time we are probably interested in
QE=zeros(T+1)
QE[1]=C0
Q=zeros(T+1)

(Q)=Stochastic(LX,LX,T,PM,PP,PD,C0) #This is the call to the funtion that evaluates the stochastic model
(QE)=Continuum(T,C0,PP,PD) #This is the call to the funtion that evaluates the continuous model
p1=plot(0:1:T,Q,ylims=(0,1.2),lw=4,label=false)
p1=plot!(0:1:T,QE,ylims=(0,1.2),ls=:dash,lw=4,xlabel='t',ylabel='C',label=false)
display(p1)
savefig(p1,"figure1.pdf")




