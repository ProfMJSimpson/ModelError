using Plots
using Random
using NLopt, LinearAlgebra
using Distributions
using .Threads
using PosDefManifold
using Roots, Loess, DifferentialEquations
using Interpolations,CubicSplines,DSP,Dierckx
using StatsPlots
using LatinHypercubeSampling
plotlyjs()
gr()
function Continuum(T,C0,PP,PD)
    QE=zeros(length(T))
    KK = (PP-PD)/PP
    rr=(PP-PD)
for i in 1:length(T)
    tt=T[i]
    QE[i]=C0*KK/(C0+(KK-C0)*exp(-rr*tt)) #This is the exact solution
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
    Q=f1(0:1:Tend)
    
    return Q
    
    end





function ErrorSample(PP,PD,Sigma_et,Sigma_tt,mu_e,Sigma_ee,T,C0,NN)
local QE=Continuum(T,C0,PP,PD);
local pp=[PP; PD];
mu_egt=mu_e+Sigma_et*inv(Sigma_tt)*(pp-mu_t);
Sigma_egt=Sigma_ee-Sigma_et*inv(Sigma_tt)*(pp*QE')/(length(pp)*length(QE'));
SSigma_egt=(Sigma_egt+Sigma_egt')/2
SSigma_egt=SSigma_egt-2.0*minimum(eigvals(SSigma_egt))*I
dc=MvNormal(zeros(length(mu_e)), SSigma_egt)
e = rand(dc, NN);
return mu_egt,SSigma_egt
end


LX=100 #this is the side length of the domain - leave this - but you can make the results smoother by increasing LX
PM=1.0 #Leave this fixed - this is the movement probability
PP=0.8 #Vary this - this is the proliferation probability
PD=0.2 #vary this - this is the death probability
C0=0.05 #Vary this - this is the initial density
T=0:1:30 #This is an estimate of the maximum time we are probably interested in
P=[0.0; 0.0] #This is the vector of unknowns
QS=zeros(length(T))
QE=zeros(length(T))
(QS)=Stochastic(LX,LX,maximum(T),PM,PP,PD,C0)
(QE)=Continuum(T,C0,PP,PD) #This is the call to the funtion that evaluates the continuous model
p1=scatter(T,QS,ylims=(0,1.2),label="slow/stochastic")
p1=plot!(T,QE,ylims=(0,1.2),lw=4,label="fast/continuum",xlabel='t',ylabel='Q')
display(p1)


N=500
x1=zeros(N);
x2=zeros(N);

#for i in 1:N
#x1[i]=rand(Uniform(0.6,1.0))
#x2[i]=rand(Uniform(0.0,0.4))
#end

#for i in 1:N
#x1[i]=rand(Truncated(Normal(0.8,0.1),0.0,10.0))
#x2[i]=rand(Truncated(Normal(0.2,0.1),0.0,10.0))
#end


plan = randomLHC(N,2);
scaledplan = scaleLHC(plan,[(0.6,1.0),(0.0,0.4)]);3
x1[:]=scaledplan[:,1];
x2[:]=scaledplan[:,2];


p2=scatter(x1,x2,markerstrokewidth=0,markersize=3,xlims=(0,1.2),ylims=(0,0.6),xticks=(0:0.2:1.2),yticks=(0:0.2:0.6),aspect_ratio=:equal,linewidth=0,label=false,xlabel="Pp",ylabel="Pd")
display(p2)
savefig(p2,"Sample.pdf")

Y=zeros(length(T)+length(P),N) #Store the data,

 for i in 1:N
println(i)
local QS=Stochastic(LX,LX,maximum(T),PM,x1[i],x2[i],C0) #This is the call to the funtion that evaluates the stochastic model
local QE=Continuum(T,C0,x1[i],x2[i]) #This is the call to the funtion that evaluates the continuous model

     for k in 1:length(T)
     Y[k,i] = QS[k]-QE[k]
     end

     Y[length(T)+1,i]=x1[i]
     Y[length(T)+2,i]=x2[i]

 end

mu=mean(Y,dims=2); #Calculate the row mean
Y .-=mu; #Subtract off the row mean 
Sigma=(Y*Y')/(N-1); #Calculate the covariance matrix can divide by N or N-1.

#OK let's slice the covariance matrix
Sigma_ee=Sigma[1:length(T),1:length(T)];
Sigma_et=Sigma[1:length(T),length(T)+1:length(T)+length(P)];
Sigma_tt=Sigma[1+length(T):length(T)+length(P),length(T)+1:length(T)+length(P)];
mu_e=mu[1:length(T)];
mu_t=mu[length(T)+1:length(T)+length(P)];

PM=1.0
PP=0.8
PD=0.2
NN=1

(mu_egt,Sigma_egt)= ErrorSample(PP,PD,Sigma_et,Sigma_tt,mu_e,Sigma_ee,T,C0,NN);
#plot(e,legend=false,xlabel="time",ylabel="error")    

(QS)=Stochastic(LX,LX,maximum(T),PM,PP,PD,C0) #This is the call to the funtion that evaluates the stochastic model
(QE)=Continuum(T,C0,PP,PD) #This is the call to the funtion that evaluates the continuous model
p3=plot(T,QS,ylims=(0,1.2),lw=4,label=false)
p3=plot!(T,QE,ylims=(0,1.2),lw=4,ls=:dash,label=false)
p3=plot!(T,QE+mu_egt,ylims=(0,1.2),lw=4,color = [:red],ls=:dashdot,xlabel='t',ylabel='C',label=false)


display(p3)
savefig(p3,"BAE_Mean1.pdf")

#p4=plot(T,QS,ylims=(0,1.2),lw=4,label=false)
#for ii in 1:NN
#p4=plot!(T,QE+mu_egt+e[:,ii],la=0.25,lc=:green,ylims=(0,1.2),xlabel='t',ylabel='C',label=false)
#end
#display(p4)
#savefig(p4,"BAE_Sample3.pdf")


#Now let's write some code to find the MLE and profile for when we have more than two parameters

function error(a)
    ee=0
    s=1/200;
    local (QE)=Continuum(T,C0,a[1],a[2])
    local (mu_egt,Sigma_egt)= ErrorSample(a[1],a[2],Sigma_et,Sigma_tt,mu_e,Sigma_ee,T,C0,NN)
        
        #dc=MvNormal(zeros(length(mu_egt)),s*I);
        #ee = loglikelihood(dc,QE-QS)
        
        #dcb=MvNormal(zeros(length(mu_egt)),Sigma_egt);
        #ee = loglikelihood(dcb,QE+mu_egt-QS)
        
        dcbb=MvNormal(zeros(length(mu_egt)),Sigma_egt+s*I);
        ee = loglikelihood(dcbb,QE+mu_egt-QS)
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

θG = [PP,PD]
lb=[0.0, 0.0]
ub=[10.0,10.0]
(xopt,fopt)  = optimise(error,θG,lb,ub);





#Let's plot the likelihood - both with and without the BAE
 xxloc=0.60:0.005:1.00
 yyloc=0.00:0.005:0.40
ll=zeros(length(xxloc),length(yyloc))
llb=zeros(length(xxloc),length(yyloc))
llbb=zeros(length(xxloc),length(yyloc))
s=1.0/200;

for ii in 1:length(xxloc)
    for jj in 1:length(yyloc)
         local (QE)=Continuum(T,C0,xxloc[ii],yyloc[jj])
         local (mu_egt,Sigma_egt)= ErrorSample(xxloc[ii],yyloc[jj],Sigma_et,Sigma_tt,mu_e,Sigma_ee,T,C0,NN)
       
         dc=MvNormal(zeros(length(mu_egt)),s*I);
         dcb=MvNormal(zeros(length(mu_egt)),Sigma_egt);
         dcbb=MvNormal(zeros(length(mu_egt)),Sigma_egt+s*I);

         ll[ii,jj]=ll[ii,jj]+loglikelihood(dc,QE-QS) 
         llb[ii,jj]=llb[ii,jj]+loglikelihood(dcb,QE+mu_egt-QS) 
         llbb[ii,jj]=llbb[ii,jj]+loglikelihood(dcbb,QE+mu_egt-QS) 
    end
end




 
#Here I'd like to plot two heat maps with a consistent colorbar
clims=extrema([llbb.-maximum(llbb);llb.-maximum(llb); ll.-maximum(ll)])
q1=contourf(xxloc,yyloc,ll'.-maximum(ll),linewidth=0,xlabel="PP",ylabel="PD",c=:viridis, colorbar=false)
q1=contour!(xxloc,yyloc,ll'.-maximum(ll),levels=[-3.0], c=:black, linewidth=3)
q1=scatter!([PP],[PD],markersize=3,markershape=:circle, markercolor=:blue,legend=false)
q2=contourf(xxloc,yyloc,llb'.-maximum(llb),linewidth=0,xlabel="PP",ylabel="PD",c=:viridis, colorbar=false)
q2=contour!(xxloc,yyloc,llb'.-maximum(llb),levels=[-3.0], c=:black, linewidth=3)
q2=scatter!([PP],[PD],markersize=3,markershape=:circle, markercolor=:blue, legend=false)
q3=contourf(xxloc,yyloc,llbb'.-maximum(llbb),linewidth=0,xlabel="PP",ylabel="PD",c=:viridis, colorbar=false)
q3=contour!(xxloc,yyloc,llbb'.-maximum(llbb),levels=[-3.0], c=:black, linewidth=3)
q3=scatter!([PP],[PD],markersize=3,markershape=:circle, markercolor=:blue, legend=false)



h2 = scatter([0,1], [0,1], zcolor=[0,3], xlims=(1,1.1), clims=clims, label="", c=:viridis, framestyle=:none, right_margin=5Plots.mm)

l = @layout [grid(1, 3) a{0.035w}]
p_all = plot(q1, q2, q3, h2, layout=l, link=:all)
display(p_all)
savefig(p_all, "Likelihood.pdf")


nptss = 100
a1min=0.50
a1max=1.00
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
local ub1=[1.0]
if i==1
local θG1=[0.2]
elseif i>1
local θG1=[0.2]
end


local (xo,fo)=optimise(fun1,θG1,lb1,ub1)
nrange[i]=xo[1]
lhooda1[i]=fo[1]
end

llhooda1=lhooda1.-maximum(lhooda1)


#To profile a2 we will specify fixed values of the nuisance parameter a1 and then calculate a2 that optimises the function
nptss = 100
a2min=0.10
a2max=0.40
a2range=LinRange(a2min,a2max,nptss)
nrange=zeros(nptss)
lhooda2=zeros(nptss)
llhooda2=zeros(nptss)


for i in 1:nptss

function fun2(aa)
return error([aa[1],a2range[i]])
end

local lb1=[0.0]
local ub1=[1.0]
if i==1
local θG1=[0.50]
elseif i>1
local θG1=[0.50]
end


local (xo,fo)=optimise(fun2,θG1,lb1,ub1)
nrange[i]=xo[1]
lhooda2[i]=fo[1]
end

llhooda2=lhooda2.-maximum(lhooda2)

s1=plot(a1range,llhooda1,xlim=(a1min,a1max),ylim=(-3,0),xlabel="PP",ylabel="llp")
s1=hline!([-1.92])
s1=vline!([PP])
s2=plot(a2range,llhooda2,xlim=(a2min,a2max),ylim=(-3,0),xlabel="PD",ylabel="llp")
s2=hline!([-1.92])
s2=vline!([PD])
s3=plot(s1,s2,layout=(1,2),legend=false)
display(s3)
savefig(s3, "Profiles3.pdf")

#Now let's interpolate the data
f1=LinearInterpolation(a1range,llhooda1);
aa= 0.5;
bb=0.7;
while (bb-aa) >= 0.000001
cc = (aa+bb)/2.0
    if sign(f1(cc)+1.92) == sign(f1(aa)+1.92)
        aa=cc
    else
        bb=cc
    end
end


f2=LinearInterpolation(a2range,llhooda2);
aa= 0.20;
bb=0.4;
while (bb-aa) >= 0.0000001
cc = (aa+bb)/2.0
    if sign(f2(cc)+1.92) == sign(f2(aa)+1.92)
        aa=cc
    else
        bb=cc
    end
end

