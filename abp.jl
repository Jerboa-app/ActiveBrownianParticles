"""
Naive approach to calculating forces (scales very poorly, O(n^2)).
    Takes as input an array of floats X with dimensions (n,d) for n particles
    in d dimensions

    Returns a vector of forces F (n,d) when collisions are detected.
"""
function Fcol(X,radius=1.0)
    F = zeros(size(X,1),2)
    r2 = 4.0*radius*radius # distance threshold
    for i in 1:size(X,1)
        for j in i+1:size(X,1) # use symmetry
            if j != i
                rij = X[j,1:2] .- X[i,1:2] # relative dist from i to j
                d2 = rij[1]*rij[1]+rij[2]*rij[2] # can avoid one sqrt
                if d2 < r2
                    d = sqrt(d2)
                    F[i,:] += (d-2*radius)*rij/d
                    F[j,:] -= (d-2*radius)*rij/d # use symmetry
                end
            end
        end
    end
    return F
end

"""
    Takes as input the positions of N particles X ~ (N,2), and returns
    a new positions in the next time frame.
"""
function Step(X;radius=1.0,v0=1.0,dt=0.001,Dr=0.1,Dt=0.0,L=10,k=300.)
    Y = copy(X)
    F = Fcol(X,radius) # get forces
    for i in 1:size(X,1)
        d = [cos(X[i,3]),sin(X[i,3])] # i's directions vector
        Y[i,3] += sqrt(2.0*Dr/dt)*randn()*dt # increment of rotational noise
        # self propulsion + forces + spatial diffusion
        Y[i,1:2] += dt*(v0*d + k*F[i,:] + sqrt(2.0*Dt/dt)*randn(2))
    end
    # periodic box
    Y[:,1:2] = mod.(Y[:,1:2],L)
    return Y
end
