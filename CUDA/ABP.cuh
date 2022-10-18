#ifndef ABP_CUH
#define ABP_CUH

/*
   True if two ellipses overlap, see "Excitations of ellipsoid packings near jamming, 2009, Europhysics Letters" 
    ArXiv: https://arxiv.org/abs/0904.1558 
*/
__global__ void ForcesEllipse(float * X, float * theta, float * ABs, float * F, int P, float dt,float k, float L){
  int ROW = blockIdx.x*blockDim.x + threadIdx.x;
  int COL = blockIdx.y*blockDim.y + threadIdx.y;
	float a1 = ABs[ROW*2];
	float b1 = ABs[ROW*2+1];
	float a2 = ABs[COL*2];
	float b2 = ABs[COL*2+1];
	float chi = ( std::pow(a1/b1,2) -1 )/( std::pow(a2/b2,2) +1);
	float sigma_0 = a1+a2;
	if (ROW < P && COL < P){
		float Rx = X[COL*4] - X[ROW*4];
		float Ry = X[COL*4+1] - X[ROW*4+1];

		// if periodic, use nearest image method
		if (L>0){

			if (Rx > L*0.5)
				Rx -= L;
			else if (Rx <= -L*0.5)
				Rx += L;

			if (Ry > L*0.5)
				Ry -= L;
			else if (Ry <= -L*0.5)
				Ry += L;

		}

		float dist = sqrt(Rx*Rx+Ry*Ry);

		float theta_a = theta[ROW];
		float theta_b = theta[COL];
		float r1 = Rx/dist;
		float r2 = Ry/dist;
		float sigma_ij = sigma_0/sqrt(1.0-(chi/2.0)*(pow(r1*cos(theta_a)+r2*sin(theta_a) + r1*cos(theta_b)+r2*sin(theta_b),2.0) / (1+chi*(cos(theta_a)*cos(theta_b) + sin(theta_a)*sin(theta_b)))+pow(r1*cos(theta_a)+r2*sin(theta_a) - r1*cos(theta_b)-r2*sin(theta_b),2.0) / (1-chi*(cos(theta_a)*cos(theta_b) + sin(theta_a)*sin(theta_b)))));
		if (dist <= sigma_ij){
			F[ROW*2] -= k*(sigma_ij - dist) * r1 / (sigma_0*sigma_0);
			F[ROW*2+1] -= k*(sigma_ij - dist) * r2 / (sigma_0*sigma_0);
		}
	}
}


__global__ void TorqueEllipse(float * X, float * theta, float * ABs, float * T, int P, float dt,float k, float L, float dtheta=0.01){
  int ROW = blockIdx.x*blockDim.x + threadIdx.x;
  int COL = blockIdx.y*blockDim.y + threadIdx.y;
	float a1 = ABs[ROW*2];
	float b1 = ABs[ROW*2+1];
	float a2 = ABs[COL*2];
	float b2 = ABs[COL*2+1];
	float chi = ( std::pow(a1/b1,2) -1 )/( std::pow(a2/b2,2) +1);
	float sigma_0 = a1+a2;
  if (ROW < P && COL < P){
		float Rx = X[COL*4] - X[ROW*4];
		float Ry = X[COL*4+1] - X[ROW*4+1];

		// periodic
		if (L>0){

			if (Rx > L*0.5)
				Rx -= L;
			else if (Rx <= -L*0.5)
				Rx += L;

			if (Ry > L*0.5)
				Ry -= L;
			else if (Ry <= -L*0.5)
				Ry += L;

		}

		float dist = sqrt(Rx*Rx+Ry*Ry);
		float theta_a = theta[ROW];
		float theta_b = theta[COL];
		float r1 = Rx/dist;
		float r2 = Ry/dist;
		float sigma_ij = sigma_0/sqrt(1.0-(chi/2.0)*(
	    pow(r1*cos(theta_a)+r2*sin(theta_a) + r1*cos(theta_b)+r2*sin(theta_b),2.0) / (1+chi*(cos(theta_a)*cos(theta_b) + sin(theta_a)*sin(theta_b)))+
	    pow(r1*cos(theta_a)+r2*sin(theta_a) - r1*cos(theta_b)-r2*sin(theta_b),2.0) / (1-chi*(cos(theta_a)*cos(theta_b) + sin(theta_a)*sin(theta_b)))
		));
		if (dist <= sigma_ij){
			theta_a = theta_a + dtheta;
			// basic differential, In Julia can use Dual numbers for exact differential, I have not implemented
			// a dual number for CUDA, but that should not be too hard
			float dsigma_ij = sigma_0/sqrt(1.0-(chi/2.0)*(
				    pow(r1*cos(theta_a)+r2*sin(theta_a) + r1*cos(theta_b)+r2*sin(theta_b),2.0) / (1+chi*(cos(theta_a)*cos(theta_b) + sin(theta_a)*sin(theta_b)))+
				    pow(r1*cos(theta_a)+r2*sin(theta_a) - r1*cos(theta_b)-r2*sin(theta_b),2.0) / (1-chi*(cos(theta_a)*cos(theta_b) + sin(theta_a)*sin(theta_b)))
					));
			T[ROW] -= ( 0.5*k*pow( (dsigma_ij-dist)/sigma_0 ,2.0) - 0.5*k*pow( (sigma_ij-dist)/sigma_0 ,2.0) )/dtheta;
		}
  }
}

// for circular particles
__global__ void Forces(float * X, float * F, float * ABs, int P, float dt,float k, bool periodic, float L){
  int ROW = blockIdx.x*blockDim.x + threadIdx.x;
  int COL = blockIdx.y*blockDim.y + threadIdx.y;
  if (ROW < P && COL < P){
    // collision forces
    if (ROW != COL){
      float Rx = X[COL*4] - X[ROW*4];
      float Ry = X[COL*4+1] - X[ROW*4+1];

			// if periodic use nearest image
			if (periodic){

				if (Rx > L*0.5)
					Rx -= L;
				else if (Rx <= -L*0.5)
					Rx += L;

				if (Ry > L*0.5)
					Ry -= L;
				else if (Ry <= -L*0.5)
					Ry += L;

			}

      float dist = sqrt(Rx*Rx+Ry*Ry);
			float rc = ABs[ROW*2]+ABs[COL*2];
      if (dist < rc){
        F[ROW*2] += -1.0*k*(rc-dist)*(Rx/dist);
        F[ROW*2+1] += -1.0*k*(rc-dist)*(Ry/dist);
      }
    }
  }
}


__global__ void GenerateNRandomUniforms(float* numbers, unsigned long seed, float min, float max, int N) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < N) {

        curandState state;

        curand_init(seed, i, 0, &state);

        numbers[i] = min + curand_uniform(&state)*max;
    }
}

// only works with circles atm
__device__ void elastic_boundaries(float * X, float * theta, float * ABs, int ROW, float L){
	float Vx = X[ROW*4] - X[ROW*4+2];
	float Vy = X[ROW*4+1] - X[ROW*4+3];
	float Ux = 0.0;
	float Uy = 0.0;
	float ang = 0.0;
	bool flag = false;
//	if (a == b){
		// circle
	float a = ABs[ROW*2];
	if (X[ROW*4] < a || X[ROW*4] > L-a){
		Ux = -1.0*Vx;
		ang = atan2(Vy,Ux);
		flag = true;
	}

	if (X[ROW*4+1] < a || X[ROW*4+1] > L-a){
		Uy = -1.0*Vy;
		if (flag == false){
			ang = atan2(Uy,Vx);
		}
		else{
			ang = atan2(Uy,Ux);
		}
		flag = true;
	}

	if (flag){
		X[ROW*4] += Ux;
		X[ROW*4+1] += Uy;
		theta[ROW] = ang;
	}
}

// for a Vicsek model simulation
__device__ float alignementTorque(float * X, float * theta, float rcrit, float kappa, bool periodic, float L, int ROW, int COL){
	if (ROW != COL){
		float Rx = X[COL*4] - X[ROW*4];
		float Ry = X[COL*4+1] - X[ROW*4+1];

		if (periodic){

			if (Rx > L*0.5)
				Rx -= L;
			else if (Rx <= -L*0.5)
				Rx += L;

			if (Ry > L*0.5)
				Ry -= L;
			else if (Ry <= -L*0.5)
				Ry += L;

		}

		float dist = Rx*Rx+Ry*Ry;
		if (dist < rcrit*rcrit){
			// return torque
			float cross = cos(theta[ROW])*sin(theta[COL])-sin(theta[ROW])*cos(theta[COL]); // ni X nj
			return cross*kappa;
		}
	}
	return 0;
}

// basic Euler-Maruyama scheme, can have a 2nd order method, G-JK but not implemented here
__global__ void Step(float * X, float * theta, float * F, float * ABs, float * T, float k, float mu, float mur, float Dr, float Dt, float dt, float v0, float rcrit, float kappa, bool periodic, float L, float P){
  // calculate force terms (inc random) and forward propulsion
  // X      1d flattened array of positions
  // theta  1d array of angles
  // r      particle radius
  // k      harmonic force constant
  // P      is the number of particles
  int ROW = blockIdx.x*blockDim.x + threadIdx.x;
  int COL = blockIdx.y*blockDim.y + threadIdx.y;
  if (ROW < P && COL < P){
    curandState state;
    curand_init(clock64(), ROW, 0, &state);
		// store last position
		X[ROW*4+2] = X[ROW*4];
		X[ROW*4+3] = X[ROW*4+1];
		// self propulsion
		X[ROW*4] += v0*cos(theta[ROW])*dt;
		X[ROW*4+1] += v0*sin(theta[ROW])*dt;
		// force
		X[ROW*4] += mu*dt*F[ROW*2];
		X[ROW*4+1] += mu*dt*F[ROW*2+1];
		// torque
		theta[ROW] += mur*dt*T[ROW];
		if (rcrit > 0.0){
			theta[ROW] += alignementTorque(X,theta,rcrit,kappa,periodic,L,ROW,COL);
		}
		// reset force and torque
		F[ROW*2] = 0.0;
		F[ROW*2+1] = 0.0;
		T[ROW] = 0.0;
    // random component on angle
    theta[ROW] += curand_normal(&state)*sqrt(2.0*Dr*dt);
		theta[ROW] = fmodf(theta[ROW],2*M_PI); // keep modded
    // random force
    X[ROW*4] += curand_normal(&state)*sqrt(2.0*Dt*dt);
    X[ROW*4+1] += curand_normal(&state)*sqrt(2.0*Dt*dt);

		if (isinf(L) == false){
			if (periodic){
				if (X[ROW*4] < 0.0){
					X[ROW*4] = L+X[ROW*4];
				}
				else if (X[ROW*4] > L){
					X[ROW*4] = X[ROW*4]-L;
				}
				if (X[ROW*4+1] < 0.0){
					X[ROW*4+1] = L+X[ROW*4+1];
				}
				else if (X[ROW*4+1] > L){
					X[ROW*4+1] = X[ROW*4+1]-L;
				}
			}
			else{
				// ellastic boundary conditions
				elastic_boundaries(X,theta,ABs,ROW,L);
			}
		}
		// if size of box L == inf, assume no boundary condition
  }
}

void TakeSteps(float * X, float * theta, float * ABs, float * Trajectories,
               int N, int total_steps, int StepsBetweenSaves, float dt,
							 float k, float mu, float mur, float Dt, float Dr, float v0, float rcrit, float kappa, float L, bool periodic, bool ellipse){
  float * d_X;
  float * d_theta;
	float * d_F;
	float * d_T;
	float * d_ABs;

	float * F = new float [N*2];
	for (int i = 0; i < N; i++){
		F[i*2] = 0.0;
		F[i*2+1] = 0.0;
	}

  size_t memX = N*4*sizeof(float);
	size_t memF = N*2*sizeof(float);
  size_t memT = N*sizeof(float);

  cudaMalloc(&d_X,memX);
  cudaMalloc(&d_theta,memT);
	cudaMalloc(&d_F,memF);
	cudaMalloc(&d_T,memT);
	cudaMalloc(&d_ABs,memF);

  cudaMemcpy(d_X,X,memX,cudaMemcpyHostToDevice);
  cudaMemcpy(d_theta,theta,memT,cudaMemcpyHostToDevice);
	cudaMemcpy(d_F,F,memF,cudaMemcpyHostToDevice);
	cudaMemcpy(d_ABs,ABs,memF,cudaMemcpyHostToDevice);

	dim3 ForcesthreadsPerBlock(8, 8);
  int bx = (N + ForcesthreadsPerBlock.x - 1)/ForcesthreadsPerBlock.x;
  int by = (N + ForcesthreadsPerBlock.y - 1)/ForcesthreadsPerBlock.y;
  dim3 ForcesblocksPerGrid(bx,by);

	int blockSize;   // The launch configurator returned block size
	int minGridSize; // The minimum grid size needed to achieve the
									 // maximum occupancy for a full device launch
	int gridSize;    // The actual grid size needed, based on input size

	cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize,
																			Step, 0, N);
	// Round up according to array size
	gridSize = (N + blockSize - 1) / blockSize;

	int Position = 0;
	int barWidth = 70.0;
	clock_t start;
	start = clock();
	if (silent == 0){
		std::cout << "Total Time Steps: " << total_steps/StepsBetweenSaves << std::endl;
	}
	for (int s = 0; s < total_steps/StepsBetweenSaves; s++){
		for (int t = 0; t < StepsBetweenSaves; t++){
			if (ellipse){
				ForcesEllipse<<<ForcesblocksPerGrid,ForcesthreadsPerBlock>>>(d_X,d_theta,d_ABs,d_F,N,dt,k,L);
				cudaDeviceSynchronize();
				TorqueEllipse<<<ForcesblocksPerGrid,ForcesthreadsPerBlock>>>(d_X,d_theta,d_ABs,d_T,N,dt,k,L,0.001);
			}
			else{
				Forces<<<ForcesblocksPerGrid,ForcesthreadsPerBlock>>>(d_X,d_F,d_ABs,N,dt,k,periodic,L);
			}
			cudaDeviceSynchronize();
			Step<<<gridSize,blockSize>>>(d_X,d_theta,d_F,d_ABs,d_T,k,mu,mur,Dr,Dt,dt,v0,rcrit,kappa,periodic,L,N);
			cudaDeviceSynchronize();

		}
		cudaMemcpy(X,d_X,N*4*sizeof(float),cudaMemcpyDeviceToHost);
		cudaMemcpy(theta,d_theta,N*sizeof(float),cudaMemcpyDeviceToHost);
		// update Trajectories
		for (int i = 0; i < N; i++){
			Trajectories[s*N*4 + 4*i + 0] = X[i*4];
			Trajectories[s*N*4 + 4*i + 1] = X[i*4+1];
			Trajectories[s*N*4 + 4*i + 2] = theta[i];
		}
		if (silent == 0){
			std::cout << "[";
		}
		Position = barWidth*float(s)/float(total_steps/StepsBetweenSaves);
		if (silent == 0){
			for (int i = 0; i < barWidth; i++){
				if (i < Position) std::cout << "=";
				else if (i == Position) std::cout << ">";
				else std::cout << " ";
			}
		}
		float time = (clock()-start)/(float)CLOCKS_PER_SEC;
		float rounded_down = floorf(time * 100) / 100;
		if (silent == 0){
			std::cout << "]" << int(100*float(s)/float(total_steps/StepsBetweenSaves)) << " % | " << rounded_down << "s\r";
			std::cout.flush();
		}
	}
	if (silent == 0){
		std::cout.flush();
		std::cout << std::endl;
	}

  cudaFree(d_X);
  cudaFree(d_theta);
	cudaFree(d_F);
	cudaFree(d_T);

	free(F);
  return;
}

#endif
