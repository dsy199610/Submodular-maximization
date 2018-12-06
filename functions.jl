@everywhere function f(K, matrix, sigma)
    I=I=eye(size(matrix,1))
    return 1/2*log(det(I+sigma^(-2)*kernelmatrix(K,matrix)))
end

@everywhere function reduced_mean(ep,del,t,S,Tao,K,sigma,df)
    X=[]
    m=16*ceil(Int,log(2/del)/(ep^2))
    indices_t=rand(1:1:size(df,1), t-1)
    T=df[indices_t,:]
    df=df[setdiff(1:size(df,1), indices_t), :]
    for i in 1:1:m
        x=rand(1:1:size(df,1), 1)
        f_ST=f(K,vcat(S,T), sigma)
        f_STj=f(K,vcat(S,T,df[x,:]), sigma)
        if((f_STj-f_ST)>=Tao)
            X=push!(X,1)
        else
            X=push!(X,0)
        end
    end
    
    if(mean(X)<=1-1.5*ep)
        return true
    else
        return false
    end
end


@everywhere function threshold_sam(df,k,Tao,Ep,Del,K,sigma,func_eval)
    ep=Ep/3
    r=ceil(Int,log((1-ep)^(-1), 2*size(df,1)/Del))
    m=ceil(Int,log(k)/ep)
    del=Del/(2*r*(m+1))
    S=Array{Float64}(0,size(df,2))

    for i in 1:1:r
        inc=[]
        for j in 1:1:size(df,1)#filtering
            f_S=f(K,S,sigma)
            f_Sj=f(K,vcat(S,df[j,:]'),sigma)
            func_eval += 2
            inc=push!(inc,f_Sj-f_S)
        end
        df=df[setdiff(1:size(df,1), find(inc -> inc.<Tao,inc)), :]
        
        if(size(df,1)==0)
            break
        end
        t=0
        for p in 0:1:m
        t=min(size(df,1),floor(Int,(1+ep)^(p)))
            if(reduced_mean(ep,del,t,S,Tao,K,sigma,df))
                break
            end
        func_eval += 2*16*ceil(Int,log(2/del)/(ep^2))
        end
        idx=rand(1:1:size(df,1), min(t,k-size(S,1)))
        T=df[idx,:]
        df=df[setdiff(1:size(df,1), idx), :]
        S=vcat(S,T)

        if(size(S,1)==k)
            break
        end
    end
    return S, func_eval
end

@everywhere function compare3(matrix1,matrix2)
    if(f(K,matrix1,sigma)>f(K,matrix2,sigma))
        return matrix1
    else
        return matrix2
    end
end

function exhaustive_maxi(df,k,Ep,Del,K,sigma,L,U,func_eval)
    inc=[]
    for i in 1:1:size(df,1)
        f_i=f(K,df[i,:]',sigma)
        push!(inc,f_i)
    end
    #delta_s=maximum(inc)
    delta_s=L
    r=ceil(Int,2*log(k)/Ep)
    m=ceil(Int,log(4)/Ep)
    del=Del/(r*(m+1))
    
    for p in workers()[1]:workers()[3]
    remotecall_fetch(()->delta_s,p)
    remotecall_fetch(()->r,p)
    remotecall_fetch(()->m,p)
    remotecall_fetch(()->del,p)
    remotecall_fetch(()->k,p)
    remotecall_fetch(()->Ep,p)
    remotecall_fetch(()->Del,p)
    remotecall_fetch(()->df,p)
    remotecall_fetch(()->func_eval,p)
    end
    
    for i in 0:r
        if (1+Ep)^(i)*delta_s>=U
            r=i
            break
        end
    end

    #R=SharedArray{Float64}(0,size(df,2))
    R=@sync @parallel compare3 for i in 0:1:r
        Tao=(1+Ep)^(i)*delta_s/k
        S=Array{Float64}(0,size(df,2))
        for j in 0:1:m
            if(((1-Ep)^(j)*Tao)<Tao/4)
                break
            end
            T,func_eval=threshold_sam(df,k-size(S,1),(1-Ep)^(j)*Tao, Ep,del, K, sigma,func_eval)
            S=vcat(S,T)
            if(size(S,1)==k)
                break
            end
        end
        S
    end
    func_eval += 2*r
    return R,func_eval
end

function sub_pre(df,k,Ep,Del,K,sigma,func_eval)
    inc=[]
    for i in 1:1:size(df,1)
        f_i=f(K,df[i,:]',sigma)
        func_eval += 1
        push!(inc,f_i)
    end
    delta_s=maximum(inc)
    L=delta_s; U=k*delta_s; R=U/L; R_s=2*10^6/(Del^2)
    while U/L >= R_s
        l=log(R)^2
        p=1/log(R)
        m=ceil(Int,log(2,R))
        del=Del/(2*(m+1)*log(R))
        idx=rand(1:1:size(df,1), 1/l)
        N=df[idx,:]
        for r in workers()[1]:workers()[3]
            remotecall_fetch(()->L,r)
            remotecall_fetch(()->k,r)
            remotecall_fetch(()->N,r)
            remotecall_fetch(()->del,r)
            remotecall_fetch(()->p,r)
            remotecall_fetch(()->U,r)
            remotecall_fetch(()->R,r)
            remotecall_fetch(()->R_s,r)
            remotecall_fetch(()->m,r)
            remotecall_fetch(()->delta_s,r)
            remotecall_fetch(()->l,r)
            remotecall_fetch(()->func_eval,r)
        end
        record=SharedArray{Float64}(3,m+1)
        @sync @parallel for i in 0:1:m
            tao_i=2^(i)*(L/k)
            record[1,i]=tao_i
            S_i,func_eval=threshold_sam(N,k,tao_i, 1-p,del, K, sigma,func_eval)
            record[2,i]=size(S_i,1)
            record[3,i]=f(K,S_i,sigma)
        end
        if length(record[2,:].>=k)==0 && length(record[3,:].>k*record[1,:])==0
            L=(delta_s+L)/2
            U=(4*l/del)*(delta_s+L)
        elseif length(record[2,:].!=k)==0 && length(record[3,:].<=k*record[1,:])==0
            L=(p/2)*(delta_s+U)
            U=(2*l/del)*(delta_s+U)
        else 
            i_s=0
            for i in 0:m
                tao_i=2^(i)/k
                S_i,func_eval=threshold_sam(N,k,tao_i, 1-p,del, K, sigma,func_eval)
                S_ii,func_eval=threshold_sam(N,k,tao_i*2, 1-p,del, K, sigma,func_eval)
                if size(S_i,1)<k && f(K,S_i,sigma)<=k*tao_i && f(K,S_ii,sigma)>k*tao_i*2 && size(S_ii,1)==k
                    func_eval += 2
                    i_s=i
                    break
                end
            end
            L=(p/2)*(delta_s+2^(i_s+1))
            U=(2*l/del)*(delta_s+2^(i_s+1))
        end
    end
    print("hh")
    return exhaustive_maxi(df,k,Ep,Del,K,sigma,L,U,func_eval)
end
