"""
    Map the pairwise force to a 2d kernel so at best we get one gpu thread
    calculating the force (or the abscence of it) between one pair of particles,
    with all pairs being done simultaneously!
"""
function ForceKernel(Y,F,R,k,N,t)
    index_x = (blockIdx().x-1)*blockDim().x + threadIdx().x # starts at 1 for blockidx !
    stride_x = gridDim().x*blockDim().x

    index_y = (blockIdx().y-1)*blockDim().y + threadIdx().y
    stride_y = gridDim().y*blockDim().y

    for i in index_x:stride_x:N
        for j in index_y:stride_y:N
            if (i == j)
                continue
            end
            r = 2*R # min dist by radii
            @inbounds rx = Y[t-1,j,1]-Y[t-1,i,1] # sep vector x
            @inbounds ry = Y[t-1,j,2]-Y[t-1,i,2] # sep vector y
            d2 = rx*rx+ry*ry
            if (d2 < r*r) # distance test without sqrt
                # apply a force to handle overlap
                d = CUDA.sqrt(d2) # need it now
                mag = -k*(r-d)

                F[i,1] += mag*rx/d
                F[i,2] += mag*ry/d
            end
        end
    end
end

"""
    Do an Euler-Maruyama step with the calculated forces and constants, 1d kernel
    so at best one thread for each particle.
"""
function StepKernel(Y,F,V,v0,Dr,Dt,dt,RAND,N,t)
    index_x = (blockIdx().x-1)*blockDim().x + threadIdx().x # starts at 1 for blockidx !
    stride_x = gridDim().x*blockDim().x

    for i in index_x:stride_x:N
        # angular diffusion
        @inbounds Y[t,i,3] = Y[t-1,i,3] + RAND[t-1,i,3]*CUDA.sqrt(2.0*Dr*dt)
        DT = CUDA.sqrt(2.0*Dt*dt)
        # save increments for boundary kernel
        @inbounds V[t,i,1] = dt * (v0*CUDA.cos(Y[t-1,i,3]) + F[i,1] + RAND[t-1,i,1]*DT)
        @inbounds V[t,i,2] = dt * (v0*CUDA.sin(Y[t-1,i,3]) + F[i,2] + RAND[t-1,i,2]*DT)
        # update (subject to boundary kernel)
        @inbounds Y[t,i,1] = Y[t-1,i,1] + V[t,i,1]
        @inbounds Y[t,i,2] = Y[t-1,i,2] + V[t,i,2]

        # reset force while we are here
        F[i,1] = 0.0
        F[i,2] = 0.0
    end
end

"""
    Keep particles in the box [0,L]x[0,L] by ``elastic collision''
"""
function BoundsKernel(Y,F,R,V,L,N,t)
    index_x = (blockIdx().x-1)*blockDim().x + threadIdx().x # starts at 1 for blockidx !
    stride_x = gridDim().x*blockDim().x

    for i in index_x:stride_x:N
        ux = 0. |> Float32 # be careful this is needed  for atan2...
        uy = 0. |> Float32
        flag = false
        ang = 0. |> Float32
        @inbounds vx = V[t,i,1]
        @inbounds vy = V[t,i,2]

        @inbounds OUTX = Y[t,i,1] < R || Y[t,i,1] > L-R
        @inbounds OUTY = Y[t,i,2] < R || Y[t,i,2] > L-R

        if (OUTX)
            ux = -1.0*vx |> Float32
            ang = CUDA.atan2(vy,ux)
            flag = true
        end

        if (OUTY)
            uy = -1.0*vy |> Float32
            if (flag)
                ang = CUDA.atan2(uy,ux)
            else
                ang = CUDA.atan2(uy,vx)
            end
            flag = true
        end

        if (flag)
            @inbounds Y[t,i,1] += ux
            @inbounds Y[t,i,2] += uy
            @inbounds Y[t,i,3] = ang
        end
    end
end

"""
    Encapsulate GPU kernels in a function, computes data in batches to
        balance host/device memory shunting with memory on device.
"""
function GPUSteps(X,steps::Int=1;R=1.0,v0=1.0,dt=0.00166,Dt=0.0,Dr=0.005,L=10,k=300.,threads=256)
    Y = zeros(1+steps,size(X,1),size(X,2))
    Y[1,:,:] = X
    Y = Y|>cu
    V = zeros(size(Y))|>cu
    # forces
    F = zeros(size(Y,2),2)|>cu

    # curand not quite here yet?? https://discourse.julialang.org/t/how-to-generate-a-random-number-in-cuda-kernel-function/50364
    # will have to precompute random variables
    RAND = zeros(steps,size(X,1),3)
    if Dt > 0
        RAND[:,:,1:2] = randn(steps,size(X,1),2)
    end

    if Dr > 0
        RAND[:,:,3] = randn(steps,size(X,1))
    end
    RAND=RAND|>cu

    for i in 2:steps+1

        CUDA.@sync begin
            @cuda threads=threads blocks=ceil(Int,size(X,1)/threads) ForceKernel(Y,F,R,k,size(X,1),i)
        end

        CUDA.@sync begin
            @cuda threads=threads blocks=ceil(Int,size(X,1)/threads) StepKernel(Y,F,V,v0,Dr,Dt,dt,RAND,size(X,1),i)
        end

        CUDA.@sync begin
            @cuda threads=threads blocks=ceil(Int,size(X,1)/threads) BoundsKernel(Y,F,R,V,L,size(X,1),i)
        end

    end

    return Y[2:end,:,:] |> Array
end
