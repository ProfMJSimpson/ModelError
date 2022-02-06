using Plots
using Random
using NLopt, LinearAlgebra
using Distributions
using .Threads
using PosDefManifold
using Roots, Loess, DifferentialEquations
using Interpolations,CubicSplines,DSP,Dierckx
gr()
function Stochastic(LX,LY,Tend,PM,PP)
    density=zeros(Tend,LX) 
    A0=zeros(LX,LY)
    t=0.0
    deltaT=1.0;
    
    for i in 91:111
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
    
    if t > (KK*deltaT)
    
    for i in 1:LX
        for j in 1:LY
        density[KK,i]=density[KK,i]+A0[i,j]
        end
    end

    KK=KK+1
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

function ErrorSample(PM,PP,Sigma_et,Sigma_tt,mu_e,Sigma_ee,T,u0,NN,xloc,LX,dx)
 
local  Nodes= Int(LX/dx)+1;
local uc=pdesolver(LX,dx,Nodes,T,u0,0.25*PM,PP)
local f1=LinearInterpolation(T,uc[:,130]);
local DC = f1(T)

local pp=[PM; PP];

mu_egt=mu_e+Sigma_et*inv(Sigma_tt)*(pp-mu_t);
Sigma_egt=Sigma_ee-Sigma_et*inv(Sigma_tt)*(pp*DC')/(length(pp)*length(DC'));
SSigma_egt=(Sigma_egt+Sigma_egt')/2
SSigma_egt=SSigma_egt-1.5*minimum(eigvals(SSigma_egt))*I
dc=MvNormal(zeros(length(mu_e)), SSigma_egt)
e = rand(dc, NN);
return mu_egt,SSigma_egt,e
end


LX=200 #this is the side length of the domain - leave this - but you can make the results smoother by increasing LX
LY=400 #Leave this fixed - this is the movement probability
PM=0.50
PP=2.00   #Vary this - this is the proliferation probability
T=1:60 #This is an estimate of the maximum time we are probably interested in
data=zeros(Int(maximum(T)),LX) 
xxloc=1:1:LX
P=[0.0; 0.0] #This is the vector of unknowns
DS=zeros(LX)
DC=zeros(LX)
(data)=Stochastic(LX,LY,maximum(T),PM,PP)
DS = data[:,130]



dx=1.0
Nodes= Int(LX/dx)+1;
u0=zeros(Nodes)
xloc=zeros(Nodes)
uc=zeros(Int(maximum(T)),Nodes)
for i in 1:Nodes
xloc[i] = 0+(i-1)*dx
    if xloc[i]<=110 && xloc[i]>=90
        u0[i]=1.0
    end
end
uc=pdesolver(LX,dx,Nodes,T,u0,0.25*PM,PP)
f1=LinearInterpolation(T,uc[:,130]);
DC = f1(T)


p1=scatter(T,DS,ylims=(0,1.2),label="slow/stochastic")
p1=plot!(T,DC,linewidth=4,ylims=(0,1.2),xlims=(0,Int(maximum(T))),label="fast/continuum")
display(p1)

N=300
x1=zeros(N);
x2=zeros(N);
for i in 1:N
x1[i]=rand(Uniform(0.3,0.7))
x2[i]=rand(Uniform(1.8,2.2))
end




p2=scatter(x1,x2,markerstrokewidth=0,markersize=3,xlims=(0.0,1.0),ylims=(1.5,2.5),xticks=(0.0:0.2:1.0),yticks=(1.5:0.2:2.5),aspect_ratio=:equal,linewidth=0,label=false,xlabel="Pm",ylabel="Pp")
display(p2)
savefig(p2,"Sample.pdf")

Y=zeros(Int(maximum(T))+length(P),N); #Store the data,

for i in 1:N
println(i)


local (data)=Stochastic(LX,LY,maximum(T),x1[i],x2[i])
local DS = data[:,130]
local uc=pdesolver(LX,dx,Nodes,T,u0,0.25*x1[i],x2[i])
local f1=LinearInterpolation(T,uc[:,130]);
local DC = f1(T)


        for k in 1:Int(maximum(T))
        Y[k,i] = DS[k]-DC[k]
        end

        Y[Int(maximum(T))+1,i]=x1[i]
        Y[Int(maximum(T))+2,i]=x2[i]

    end

mu=mean(Y,dims=2); #Calculate the row mean
Y .-=mu; #Subtract off the row mean 
Sigma=(Y*Y')/(N-1); #Calculate the covariance matrix can divide by N or N-1.

# # #OK let's slice the covariance matrix
Sigma_ee=Sigma[1:Int(maximum(T)),1:Int(maximum(T))];
Sigma_et=Sigma[1:Int(maximum(T)),Int(maximum(T))+1:Int(maximum(T))+length(P)];
Sigma_tt=Sigma[1+Int(maximum(T)):Int(maximum(T))+length(P),Int(maximum(T))+1:Int(maximum(T))+length(P)];
mu_e=mu[1:Int(maximum(T))];
mu_t=mu[Int(maximum(T))+1:Int(maximum(T))+length(P)];


PM=0.6
PP=1.9
NN=50

(mu_egt,Sigma_egt,e)= ErrorSample(PM,PP,Sigma_et,Sigma_tt,mu_e,Sigma_ee,T,u0,NN,xloc,LX,dx);
# plot(e,legend=false,xlabel="time",ylabel="error")    
# plot(mu_egt)


(data)=Stochastic(LX,LY,maximum(T),PM,PP)
DS=data[:,130]

uc=pdesolver(LX,dx,Nodes,T,u0,0.25*PM,PP)
f1=LinearInterpolation(T,uc[:,130]);
DC = f1(T)



p3=plot(T,DS,ylims=(0,1.2),lw=4,label=false)
p3=plot!(T,DC,ylims=(0,1.2),lw=4,ls=:dash,label=false)
p3=plot!(T,DC+mu_egt,ylims=(0,1.2),lw=4,color=:red,ls=:dot,xlabel='t',ylabel='C',label=false)
display(p3)
savefig(p3,"BAE_Mean3.pdf")


p4=plot(T,DS,lw=4,label=false,xlabel='t',ylabel='Q')
for ii in 1:NN
local p4=plot!(T,DC+mu_egt+e[:,ii],la=0.25,lc=:green,label=false)
end
display(p4)
savefig(p4,"BAE_Sample3.pdf")

#Now let's write some code to find the MLE and profile for when we have more than two parameters




function error(a)
    ee=0
    s=1/200;
    uc=pdesolver(LX,dx,Nodes,T,u0,0.25*a[1],a[2])
    f1=LinearInterpolation(T,uc[:,130]);
    DC = f1(T)
    
    (mu_egt,Sigma_egt)= ErrorSample(a[1],a[2],Sigma_et,Sigma_tt,mu_e,Sigma_ee,T,u0,NN,xloc,LX,dx);
        
        #dc=MvNormal(zeros(length(mu_egt)),s*I);
        #ee = loglikelihood(dc,DC-DS)
        
        #dcb=MvNormal(zeros(length(mu_egt)),Sigma_egt);
        #ee = loglikelihood(dcb,DC+mu_egt-DS)
        
        dcbb=MvNormal(zeros(length(mu_egt)),Sigma_egt+s*I);
        ee = loglikelihood(dcbb,DC+mu_egt-DS)
    return ee
end





function optimise(fun,θ₀,lb,ub;
    dv = false,
    method = dv ? :LD_LBFGS : :LN_BOBYQA,
)

if dv || String(method)[2] == 'D'
    tomax = fun
else
    tomax = (θ,∂θ) -> fun(θ)
end

opt = Opt(method,length(θ₀))
opt.max_objective = tomax
opt.lower_bounds = lb       # Lower bound
opt.upper_bounds = ub       # Upper bound
opt.local_optimizer = Opt(:LN_BOBYQA, length(θ₀))
res = optimize(opt,θ₀)
return res[[2,1]]

end

θG = [PM,PP]
lb=[0.05, 0.05]
ub=[10.0,10.0]
(xopt,fopt)  = optimise(error,θG,lb,ub);



















#Let's plot the likelihood - both with and without the BAE
xxloc=0.20:0.02:1.00
yyloc=1.00:0.02:3.00
ll=zeros(length(xxloc),length(yyloc))
llb=zeros(length(xxloc),length(yyloc))
llbb=zeros(length(xxloc),length(yyloc))
s=0.05

for ii in 1:length(xxloc)
     for jj in 1:length(yyloc)
          uc=pdesolver(LX,dx,Nodes,T,u0,0.25*xxloc[ii],yyloc[jj])
         f1=LinearInterpolation(T,uc[:,130]);
         DC = f1(T)
         (mu_egt,Sigma_egt)= ErrorSample(xxloc[ii],yyloc[jj],Sigma_et,Sigma_tt,mu_e,Sigma_ee,T,u0,NN,xloc,LX,dx)
          
         dc=MvNormal(zeros(length(mu_egt)),s*I);
         dcb=MvNormal(zeros(length(mu_egt)),Sigma_egt);
         dcbb=MvNormal(zeros(length(mu_egt)),Sigma_egt+s*I);

         ll[ii,jj]=ll[ii,jj]+loglikelihood(dc,DC-DS) 
         llb[ii,jj]=llb[ii,jj]+loglikelihood(dcb,DC+mu_egt-DS) 
         llbb[ii,jj]=llbb[ii,jj]+loglikelihood(dcbb,DC+mu_egt-DS) 

          
     end
 end


#Here I'd like to plot two heat maps with a consistent colorbar
clims=extrema([llbb.-maximum(llbb);llb.-maximum(llb); ll.-maximum(ll)])
q1=contourf(xxloc,yyloc,ll'.-maximum(ll),linewidth=0,xlabel="PM",ylabel="PP",c=:viridis, colorbar=false)
q1=contour!(xxloc,yyloc,ll'.-maximum(ll),levels=[-3.0], c=:black, linewidth=3)
q1=scatter!([PM],[PP],markersize=3,markershape=:circle, markercolor=:blue,legend=false)
q2=contourf(xxloc,yyloc,llb'.-maximum(llb),linewidth=0,xlabel="PM",ylabel="PP",c=:viridis, colorbar=false)
q2=contour!(xxloc,yyloc,llb'.-maximum(llb),levels=[-3.0], c=:black, linewidth=3)
q2=scatter!([PM],[PP],markersize=3,markershape=:circle, markercolor=:blue, legend=false)
q3=contourf(xxloc,yyloc,llbb'.-maximum(llbb),linewidth=0,xlabel="PM",ylabel="PP",c=:viridis, colorbar=false)
q3=contour!(xxloc,yyloc,llbb'.-maximum(llbb),levels=[-3.0], c=:black, linewidth=3)
q3=scatter!([PM],[PP],markersize=3,markershape=:circle, markercolor=:blue, legend=false)



h2 = scatter([0,1], [0,1], zcolor=[0,3], xlims=(1,1.1), clims=clims, label="", c=:viridis, framestyle=:none, right_margin=5Plots.mm)

l = @layout [grid(1, 3) a{0.035w}]
p_all = plot(q1, q2, q3, h2, layout=l, link=:all)
display(p_all)
savefig(p_all, "Likelihood.pdf")



















nptss = 50
a1min=0.20
a1max=3.00
a1range=LinRange(a1min,a1max,nptss)
a2range=zeros(nptss)
nrange=zeros(nptss)
lhooda1=zeros(nptss)
llhooda1=zeros(nptss)

for i in 1:nptss
function fun1(aa)
return error([a1range[i],aa[1]])
end

local lb1=[0.0]
local ub1=[10.0]
local θG1=[1.0]



local (xo,fo)=optimise(fun1,θG1,lb1,ub1)
nrange[i]=xo[1]
lhooda1[i]=fo[1]
end

llhooda1=lhooda1.-maximum(lhooda1)


#To profile a2 we will specify fixed values of the nuisance parameter a1 and then calculate a2 that optimises the function

a2min=0.50
a2max=3.00
a2range=LinRange(a2min,a2max,nptss)
nrange=zeros(nptss)
lhooda2=zeros(nptss)
llhooda2=zeros(nptss)


for i in 1:nptss

function fun2(aa)
return error([aa[1],a2range[i]])
end

local lb1=[0.0]
local ub1=[10.0]
local θG1=[1.0]


local (xo,fo)=optimise(fun2,θG1,lb1,ub1)
nrange[i]=xo[1]
lhooda2[i]=fo[1]
end

llhooda2=lhooda2.-maximum(lhooda2)

s1=plot(a1range,llhooda1,xlim=(a1min,a1max),ylim=(-3,0),xlabel="PM",ylabel="llp")
s1=hline!([-1.92])
s1=vline!([PM])
s2=plot(a2range,llhooda2,xlim=(a2min,a2max),ylim=(-3,0),xlabel="PP",ylabel="llp")
s2=hline!([-1.92])
s2=vline!([PP])
s3=plot(s1,s2,layout=(1,2),legend=false)
display(s3)
savefig(s3, "Profiles1.pdf")


f1=LinearInterpolation(a1range,llhooda1);
aa= 0.50;
bb=1.50;
while (bb-aa) >= 0.000001
cc = (aa+bb)/2.0
    if sign(f1(cc)+1.92) == sign(f1(aa)+1.92)
        aa=cc
    else
        bb=cc
    end
end


f2=LinearInterpolation(a2range,llhooda2);
aa= 2.00;
bb=3.00;
while (bb-aa) >= 0.0000001
cc = (aa+bb)/2.0
    if sign(f2(cc)+1.92) == sign(f2(aa)+1.92)
        aa=cc
    else
        bb=cc
    end
end