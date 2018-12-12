#Pkg.add("CSV")
#Pkg.add("MLKernels")
#Pkg.add("DataFrames")
#Pkg.add("PyPlot")
addprocs(3)
using CSV
@everywhere using MLKernels
@everywhere using DataFrames
using DataStructures
using PyPlot
using Distributions
using DistributedArrays

df=CSV.read("data.csv")
df=convert(Array,df)
for i in 1:1:size(df,1)
    df[i,:]=df[i,:].-mean(df[i,:])
    df[i,:]=normalize(df[i,:]) 
end

k=10:10:100

@everywhere sigma=1
@everywhere K=GaussianKernel(1/(0.75)^2)#h
function f(K, matrix, sigma)
    I=I=eye(size(matrix,1))
    return 1/2*log(det(I+sigma^(-2)*kernelmatrix(K,matrix)))
end

#greedy
F_A=[]
func_eval_a=[]
rounds_a=[]
for n in 1:1:length(k)
    A=Array{Float64}(0,size(df,2))
    df_1=df
    func_eval=0
    rounds=0
for i in 1:1:k[n]
    rounds += 1
    inc=[]
    f_A=f(K,A,sigma)
    func_eval += 1
    for j in 1:1:size(df_1,1)
        f_Aj=f(K,vcat(A,df_1[j,:]'), sigma)
        func_eval += 1
        inc=push!(inc,f_Aj-f_A)
    end
    a = df_1[indmax(inc),:]
    df_1 = df_1[1:end .!=indmax(inc), :]
    A=vcat(A,a')
end
F_A=push!(F_A,f(K,A,sigma))
func_eval_a=push!(func_eval_a,func_eval)
rounds_a=push!(rounds_a,rounds)
print(n)
end
figure(1)
title("Utility of the solution set and k")
plot(k, F_A, color="green", label="simple Greedy")
ylabel("Utility")
xlabel("k")
figure(2)
title("Oracle Evaluations and k")
yscale("log")
plot(k, func_eval_a, color="green", label="simple Greedy")
ylabel("Oracle Evaluations")
xlabel("k")
figure(3)
title("Adaptive rounds and k")
plot(k, rounds_a, color="green", label="simple Greedy")
ylabel("Adaptive rounds")
xlabel("k")

#lazy greedy
h=binary_maxheap(Tuple{Float64,Int})
F_A_2=[]
func_eval_a2=[]
rounds_a2=[]
for n in 1:1:length(k)
    A=Array{Float64}(0,size(df,2))
    df_2=df
    func_eval_2=0
    rounds=0
for i in 1:1:k[n]
    rounds += 1
    f_A=f(K,A,sigma)
    func_eval_2 += 1
    stop=0
    if(i==1)
        for j in 1:1:size(df_2,1)
            f_Aj=f(K,vcat(A,df_2[j,:]'), sigma)
            func_eval_2 += 1
            push!(h,(f_Aj-f_A,j))
        end
        a=df_2[top(h)[2],:]
        pop!(h)
    else
        while(stop!=1)
            top_heap=top(h)
            pop!(h)
            f_Aj=f(K,vcat(A,df_2[top_heap[2],:]'), sigma)
            func_eval_2 += 1
            if((f_Aj-f_A)>=top(h)[1])
                a=df_2[top_heap[2],:]
                stop=1
            else
                push!(h,((f_Aj-f_A),top_heap[2]))
            end
        end
    end
    A=vcat(A,a')
end
F_A_2=push!(F_A_2,f(K,A,sigma))
func_eval_a2=push!(func_eval_a2,func_eval_2)
rounds_a2=push!(rounds_a2,rounds)
print(n)
end
figure(1)
plot(k, F_A_2, color="blue", label="lazy greedy")
figure(2)
plot(k, func_eval_a2, color="blue", label="lazy greedy")
figure(3)
plot(k, rounds_a2, color="blue", label="lazy greedy")


#stochastic greedy
epsilon=0.4
F_A_3=[]
func_eval_a3=[]
rounds_a3=[]
for n in 1:1:length(k)
    A=Array{Float64}(0,size(df,2))
    df_3=df
    siz=floor(Int,size(df_3,1)*log(1/epsilon)/k[n])
    func_eval_3=0
    rounds=0
for i in 1:1:k[n]
    rounds += 1
    inc=[]
    f_A=f(K,A,sigma)
    func_eval_3 += 1
    
    indices=rand(1:1:size(df_3,1), siz)
    for j in 1:1:size(indices,1)
        f_Aj=f(K,vcat(A,df_3[indices[j],:]'), sigma)
        func_eval_3 += 1
        inc=push!(inc,f_Aj-f_A)
    end
    a = df_3[indices[indmax(inc)],:]
    df_3 = df_3[1:end .!=indices[indmax(inc)], :]
    A=vcat(A,a')
end
F_A_3=push!(F_A_3,f(K,A,sigma))
func_eval_a3=push!(func_eval_a3,func_eval_3)
rounds_a3=push!(rounds_a3,rounds)
print(n)
end
figure(1)
plot(k, F_A_3, color="orange", label="stochastic greedy eps=0.4")
legend(loc="upper left")
gcf()
figure(2)
plot(k, func_eval_a3, color="orange", label="stochastic greedy eps=0.4")
legend(loc="upper left")
gcf()
figure(3)
plot(k, rounds_a3, color="orange", label="stochastic greedy eps=0.4")
legend(loc="upper left")
gcf()

#threshold sampling
Del=0.1
Ep=0.4
F_A_43=[]
func_eval_a43=[]
rounds_a43=[]
include("functions.jl")

for n in 1:1:length(k)
    A=Array{Float64}(0,size(df,2))
    df_4=df
    func_eval_43=0
    rounds=0
    A,func_eval_43,rounds=sub_pre(df_4,k[n],Ep,Del,K,sigma,func_eval_43,rounds)
    push!(F_A_43,f(K,A,sigma))
    func_eval_a43=push!(func_eval_a43,func_eval_43)
    rounds_a43=push!(rounds_a43,rounds)
    print(n)
end
figure(1)
plot(k, F_A_43, color="cyan", label="Threshold-Sampling eps=0.4")
legend(loc="upper left")
gcf()
figure(2)
plot(k, func_eval_a43, color="cyan", label="Threshold-Sampling eps=0.4")
legend(loc="upper left")
gcf()
figure(3)
plot(k, rounds_a43, color="cyan", label="Threshold-Sampling eps=0.4")
legend(loc="upper left")
gcf()