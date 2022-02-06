using Plots
using Random
using NLopt
using Distributions
using .Threads
using Roots, Loess, DifferentialEquations
using Interpolations,CubicSplines,DSP,Dierckx
gr()






function Stochastic(LX,LY,Tend,PM,PP,PD,C0)
Q0=Int(round(C0*LX*LY))
A0=zeros(LX,LY)
t=0.0
count=0
Tplot=[200,400,600,800,1000]

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




#Plot the distribution of agents at t=0
     
Q0=Int(sum(A0))
pos0=zeros(Q0,2)
pos1=zeros(Q0,2)
pos2=zeros(Q0,2)
pos3=zeros(Q0,2)
pos4=zeros(Q0,2)
pos5=zeros(Q0,2)

agent=0
for i in 1:LX
   for j in 1:LY
       if A0[i,j] == 1.0
       agent = agent+1
       pos0[agent,1]=i
       pos0[agent,2]=j
       end 
   end
 end



















 KK=1
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


Density_record[count+1] = Q0/(LX*LY)
Time_record[count+1]=t
count = count + 1


if KK == 1 && t > Tplot[1]

    Q0=Int(sum(A0))
    pos1=zeros(Q0,2)

    agent=0
    for i in 1:LX
       for j in 1:LY
           if A0[i,j] == 1.0
           agent = agent+1
           pos1[agent,1]=i
           pos1[agent,2]=j
           end 
       end
     end
KK=KK+1

elseif KK == 2 && t > Tplot[2]

    Q0=Int(sum(A0))
    pos2=zeros(Q0,2)

    agent=0
    for i in 1:LX
       for j in 1:LY
           if A0[i,j] == 1.0
           agent = agent+1
           pos2[agent,1]=i
           pos2[agent,2]=j
           end 
       end
     end
KK=KK+1

elseif KK == 3 && t > Tplot[3]
    agent=0

    Q0=Int(sum(A0))
    pos3=zeros(Q0,2)

    for i in 1:LX
       for j in 1:LY
           if A0[i,j] == 1.0
           agent = agent+1
           pos3[agent,1]=i
           pos3[agent,2]=j
           end 
       end
     end
KK=KK+1

elseif KK == 4 && t > Tplot[4]
    agent=0

    Q0=Int(sum(A0))
    pos4=zeros(Q0,2)
    for i in 1:LX
       for j in 1:LY
           if A0[i,j] == 1.0
           agent = agent+1
           pos4[agent,1]=i
           pos4[agent,2]=j
           end 
       end
     end
KK=KK+1

elseif KK == 5 && t > Tplot[5]
    agent=0

    Q0=Int(sum(A0))
    pos5=zeros(Q0,2)
    for i in 1:LX
       for j in 1:LY
           if A0[i,j] == 1.0
           agent = agent+1
           pos5[agent,1]=i
           pos5[agent,2]=j
           end 
       end
     end
end






















end

density=zeros(count+1)
time=zeros(count+1)

density = Density_record[1:count]
time = Time_record[1:count]

f1=LinearInterpolation(time,density)
Q=f1(0:1:T)

return Q,pos0,pos1,pos2,pos3,pos4,pos5

end

    
       
       




LX=100 #this is the side length of the domain - leave this - but you can make the results smoother by increasing LX
PM=1.0 #Leave this fixed - this is the movement probability
PP=1.0/100  #Vary this - this is the proliferation probability
PD=1.0/200 #Vary this - this is the proliferation probability
C0=0.1 #Vary this - this is the initial density
T=1000 #This is an estimate of the maximum time we are probably interested in
Q=zeros(T+1)

(Q,pos0,pos1,pos2,pos3,pos4,pos5)=Stochastic(LX,LX,T,PM,PP,PD,C0) #This is the call to the funtion that evaluates the stochastic model
p1=plot(0:1:T,Q,ylims=(0,1.2),ls=:dash,lw=4,xlabel='t',ylabel='C',label=false)
display(p1)
savefig(p1,"figure1.pdf")




q0=scatter(pos0[:,1],pos0[:,2],markersize=1.5,markershape=:circle, markercolor=:red,xlims=(0,LX),ylims=(0,LX),aspect_ratio=:equal,legend=false)
q1=scatter(pos1[:,1],pos1[:,2],markersize=1.5,markershape=:circle, markercolor=:red,xlims=(0,LX),ylims=(0,LX),aspect_ratio=:equal,legend=false)
q2=scatter(pos2[:,1],pos2[:,2],markersize=1.5,markershape=:circle, markercolor=:red,xlims=(0,LX),ylims=(0,LX),aspect_ratio=:equal,legend=false)
q3=scatter(pos3[:,1],pos3[:,2],markersize=1.5,markershape=:circle, markercolor=:red,xlims=(0,LX),ylims=(0,LX),aspect_ratio=:equal,legend=false)
q4=scatter(pos4[:,1],pos4[:,2],markersize=1.5,markershape=:circle, markercolor=:red,xlims=(0,LX),ylims=(0,LX),aspect_ratio=:equal,legend=false)
q5=scatter(pos5[:,1],pos5[:,2],markersize=1.5,markershape=:circle, markercolor=:red,xlims=(0,LX),ylims=(0,LX),aspect_ratio=:equal,legend=false)

q6=plot(q0,q1,q2,q3,q4,q5,layout=(2,3))
display(q6)
savefig(q6,"Dot2.pdf") 