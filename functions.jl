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


@everywhere function threshold_sam(df,k,Tao,Ep,Del,K,sigma,func_eval,rounds)
    ep=Ep/3
    r=ceil(Int,log((1-ep)^(-1), 2*size(df,1)/Del))
    m=ceil(Int,log(k)/ep)
    del=Del/(2*r*(m+1))
    S=Array{Float64}(0,size(df,2))
    for i in 1:1:r
        rounds += 1
        inc=[]
        f_S=f(K,S,sigma)
        for j in 1:1:size(df,1)#filtering
            f_Sj=f(K,vcat(S,df[j,:]'),sigma)
            func_eval += 1
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
    return S, func_eval,rounds
end

@everywhere function compare3(matrix1,matrix2)
    if(f(K,matrix1,sigma)>f(K,matrix2,sigma))
        return matrix1
    else
        return matrix2
    end
end
@everywhere function max_rounds(round1,round2)
    if(round1>=round2)
        return round1
    else
        return round2
    end
end
@everywhere type set_rounds
    set ::Array{Union{Float64, Missings.Missing},2}
    rounds ::Int64
    eval ::Int64
  end
@everywhere join(a::set_rounds, b::set_rounds) = set_rounds(compare3(a.set,b.set), max_rounds(a.rounds,b.rounds),a.eval+b.eval)

function exhaustive_maxi(df,k,Ep,Del,K,sigma,L,U,func_eval,rounds)
    inc=[]
    for i in 1:1:size(df,1)
        f_i=f(K,df[i,:]',sigma)
        func_eval += 1
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
    remotecall_fetch(()->rounds,p)
    end
    
    for i in 0:r
        if (1+Ep)^(i)*delta_s>=U
            r=i
            break
        end
    end
    #R=SharedArray{Float64}(0,size(df,2))
    
    set_rounds_=@sync @parallel (join) for i in 0:1:r
        Tao=(1+Ep)^(i)*delta_s/k
        S=Array{Float64}(0,size(df,2))
        for j in 0:1:m
            if(((1-Ep)^(j)*Tao)<Tao/4)
                break
            end
            rounds=0
            T,func_eval,rounds=threshold_sam(df,k-size(S,1),(1-Ep)^(j)*Tao, Ep,del, K, sigma,func_eval,rounds)
            print("lalalalalalalal",rounds)
            S=vcat(S,T)
            if(size(S,1)==k)
                break
            end
        end
        set_rounds(S,rounds,func_eval)
    end
    print("maxrounds",set_rounds_.rounds)
    func_eval = func_eval+set_rounds_.eval+2*r
    return set_rounds_.set,func_eval,set_rounds_.rounds
end

function sub_pre(df,k,Ep,Del,K,sigma,func_eval,rounds)
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
        set_rounds_ = @sync @parallel (join) for i in 0:1:m
            tao_i=2^(i)*(L/k)
            record[1,i]=tao_i
            S_i,func_eval,rounds=threshold_sam(N,k,tao_i, 1-p,del, K, sigma,func_eval,rounds)
            record[2,i]=size(S_i,1)
            record[3,i]=f(K,S_i,sigma)
            set_rounds(S_i,rounds,func_eval)
        end
        rounds=set_rounds_.rounds
        func_eval=set_rounds_.eval
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
                S_i,func_eval,rounds=threshold_sam(N,k,tao_i, 1-p,del, K, sigma,func_eval,rounds)
                S_ii,func_eval,rounds=threshold_sam(N,k,tao_i*2, 1-p,del, K, sigma,func_eval,rounds)
                if size(S_i,1)<k && f(K,S_i,sigma)<=k*tao_i && f(K,S_ii,sigma)>k*tao_i*2 && size(S_ii,1)==k
                    func_eval += 2
                    i_s=i
                    break
                end
            end
            L=(p/2)*(delta_s+2^(i_s+1))
            U=(2*l/del)*(delta_s+2^(i_s+1))
        end
        print("valid")
    end
    return exhaustive_maxi(df,k,Ep,Del,K,sigma,L,U,func_eval,rounds)
end
