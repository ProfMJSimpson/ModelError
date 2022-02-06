using Plots
using Random
using NLopt
using Distributions
using .Threads
using Roots, Loess, DifferentialEquations
using Interpolations,CubicSplines,DSP,Dierckx
gr()
global iter=0

function Stochastic(LX,LY,Tend,Tplot,PM,PP)
    density=zeros(4,LX) 
    A0=zeros(LX,LY)
    t=0.0
    
    for i in 90:110
        for j in 1:LY
        A0[i,j]=1.0
        end
    end

#Plot the distribution of agents at t=0
     
     Q0=Int(sum(A0))
     pos0=zeros(Q0,2)
     pos1=zeros(Q0,2)
     pos2=zeros(Q0,2)
     pos3=zeros(Q0,2)
     pos4=zeros(Q0,2)
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
      

    Q0=Int(sum(A0))
    KK=1
    while t < Tend
    
        Q0=Int(sum(A0))
        a0=(PM+PP)*Q0
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
    
        Q0=Int(sum(A0)) 
        end
    
    if KK == 1 && t > Tplot[1]
    

    for i in 1:LX
        for j in 1:LY
        density[KK,i]=density[KK,i]+A0[i,j]
        end
    end

    KK=KK+1
    print("Output 1")


    
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


    elseif KK==2 && t > Tplot[2]
    

    for i in 1:LX
        for j in 1:LY
        density[KK,i]=density[KK,i]+A0[i,j]
        end
    end
    KK=KK+1
    print("Output 2")

    
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

    elseif KK==3 && t > Tplot[3]
    

    for i in 1:LX
        for j in 1:LY
        density[KK,i]=density[KK,i]+A0[i,j]
        end
    end
    KK=KK+1
    print("Output 3")

    
     Q0=Int(sum(A0))
     pos3=zeros(Q0,2)
     agent=0
    for i in 1:LX
        for j in 1:LY
            if A0[i,j] == 1.0
            agent = agent+1
            pos3[agent,1]=i
            pos3[agent,2]=j
            end 
        end
      end

    elseif KK==4 && t > Tplot[4]

        for i in 1:LX
            for j in 1:LY
            density[KK,i]=density[KK,i]+A0[i,j]
            end
        end
        print("Output  4")

        Q0=Int(sum(A0))
        pos4=zeros(Q0,2)
        agent=0
       for i in 1:LX
           for j in 1:LY
               if A0[i,j] == 1.0
               agent = agent+1
               pos4[agent,1]=i
               pos4[agent,2]=j
               end 
           end
         end

    end

    


    end
    

      

    
    density=density/LY

    
    return density,pos0,pos1,pos2,pos3,pos4
    
    end
     
     
     
     
     
     
     
 
     
    
























LX=200
LY=50
T=1:800
Tplot=[200,400,600,800]
PM=1/1
PP=1/100
data=zeros(4,LX) 
xxloc=1:1:LX
ttloc=1:1:maximum(T)

(data,pos0,pos1,pos2,pos3,pos4)=Stochastic(LX,LY,maximum(T),Tplot,PM,PP)



p1=plot(xxloc,data[1,:],lw=4,legend=false,xlims=(0,LX),ylims=(0,1.2))
p2=plot(xxloc,data[2,:],lw=4,legend=false,xlims=(0,LX),ylims=(0,1.2))
p3=plot(xxloc,data[3,:],lw=4,legend=false,xlims=(0,LX),ylims=(0,1.2))
p4=plot(xxloc,data[4,:],lw=4,legend=false,xlims=(0,LX),ylims=(0,1.2))


    
p5=plot(p1,p2,p3,p4,layout=(4,1))
display(p5)
savefig(p5,"Continuum_Discrete.pdf") 


q0=scatter(pos0[:,1],pos0[:,2],markersize=2,markershape=:circle, markercolor=:red,xlims=(0,LX),ylims=(0,LY),legend=false)
q1=scatter(pos1[:,1],pos1[:,2],markersize=2,markershape=:circle, markercolor=:red,xlims=(0,LX),ylims=(0,LY),legend=false)
q2=scatter(pos2[:,1],pos2[:,2],markersize=2,markershape=:circle, markercolor=:red,xlims=(0,LX),ylims=(0,LY),legend=false)
q3=scatter(pos3[:,1],pos3[:,2],markersize=2,markershape=:circle, markercolor=:red,xlims=(0,LX),ylims=(0,LY),legend=false)
q4=scatter(pos4[:,1],pos4[:,2],markersize=2,markershape=:circle, markercolor=:red,xlims=(0,LX),ylims=(0,LY),legend=false)

q5=plot(q0,q1,q2,q3,q4,layout=(5,1))
display(q5)
savefig(q5,"Dot.pdf") 