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

    elseif KK==2 && t > Tplot[2]
    

    for i in 1:LX
        for j in 1:LY
        density[KK,i]=density[KK,i]+A0[i,j]
        end
    end
    KK=KK+1
    print("Output 2")


    elseif KK==3 && t > Tplot[3]
    

    for i in 1:LX
        for j in 1:LY
        density[KK,i]=density[KK,i]+A0[i,j]
        end
    end
    KK=KK+1
    print("Output 3")


    elseif KK==4 && t > Tplot[4]

        for i in 1:LX
            for j in 1:LY
            density[KK,i]=density[KK,i]+A0[i,j]
            end
        end
        print("Output  4")
    end

    


    end
    
    
    density=density/LY
    return density
    
    end
     
     
     
     
     
     
     
 
     
    















function diff!(du,u,p,t)
    D,r,dx,N=p
    
     for i in 2:N-1
        du[i]=D*(u[i+1]-2*u[i]+u[i-1])/dx^2 + r*u[i]*(1-u[i])
    end
    du[1]=0.0
    du[N]=0.0
    du[1]=du[2]
    du[N]=du[N-1]

    end



function pdesolver(L,dx,N,T,u0,D,r)
p=(D,r,dx,N)
tspan=(0.0,maximum(T))
prob=ODEProblem(diff!,u0,tspan,p)
sol=solve(prob,saveat=T);
   
   for i in 1:length(sol[:,])
   uc[i,:]=sol[:,i]
   end
    
return uc
end





LX=200
LY=500
T=1:80
Tplot=[20,40,60,80]
PM=1/1
PP=1/1
data=zeros(4,LX) 
xxloc=1:1:LX
ttloc=1:1:maximum(T)

(data)=Stochastic(LX,LY,maximum(T),Tplot,PM,PP)







dx=1.0
N=Int(LX/dx)+1
D = PM/4.0
Lambda = PP
u0=zeros(N)
xloc=zeros(N)
uc=zeros(Int(maximum(T)),N)


for i in 1:N
xloc[i] = 0+(i-1)*dx
    if xloc[i]<=110 && xloc[i]>=90
        u0[i]=1.0
    end
end

    
    

uc=pdesolver(LX,dx,N,T,u0,D,Lambda) 


#Interpolate numerical results
f1=LinearInterpolation(xloc,uc[Tplot[1],:]);
f2=LinearInterpolation(xloc,uc[Tplot[2],:]);
f3=LinearInterpolation(xloc,uc[Tplot[3],:]);
f4=LinearInterpolation(xloc,uc[Tplot[4],:]);    
p1=plot(xxloc,data[1,:],lw=4)
p1=plot!(xxloc,f1(xxloc),linewidth=4,ls=:dash,xlims=(0,LX),ylims=(0,1.2),legend=false)
p2=plot(xxloc,data[2,:],lw=4)
p2=plot!(xxloc,f2(xxloc),linewidth=4,ls=:dash,xlims=(0,LX),ylims=(0,1.2),legend=false)
p3=plot(xxloc,data[3,:],lw=4)
p3=plot!(xxloc,f3(xxloc),linewidth=4,ls=:dash,xlims=(0,LX),ylims=(0,1.2),legend=false)
p4=plot(xxloc,data[4,:],lw=4)
p4=plot!(xxloc,f4(xxloc),linewidth=4,ls=:dash,xlims=(0,LX),ylims=(0,1.2),legend=false)



    
p5=plot(p1,p2,p3,p4,layout=(4,1))
display(p5)
savefig(p5,"Continuum_Discrete.pdf") 

