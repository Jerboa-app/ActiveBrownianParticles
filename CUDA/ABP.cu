#include <iostream>
#include <cmath>
#include <ctime>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <random>


int N = 100;
float a = 1.0; // diameter
float b = 1.0; // diameter
float k = 10.0;
float mu = 1.0;
float mu_r = 1.0;
float dt = 1.0/300.0;
float Dr = 0.0;
float Dt = 0.0;
float v0 = 1.0;
float L = 1./0.;
float TotalTime = 30.0;
float packing = 0.5;
float rcrit = 0.0;
float kappa = 0.0;
int StepsBetweenSaves = 30;
int silent = 0;
bool periodic = false;
bool randomSizes = false;

unsigned long seed = 123456789;

#include "ABP.cuh"

/*
   True if two ellipses overlap, see "Excitations of ellipsoid packings near jamming, 2009, Europhysics Letters" 
    ArXiv: https://arxiv.org/abs/0904.1558 
*/
bool ellipseEllipseContact(
	float x1, float x2, float y1, float y2,
	float a1, float a2, float b1, float b2,
	float theta_a, float theta_b){
	float Rx = x2-x1;
	float Ry = y2-y1;
	float dist = sqrt(Rx*Rx+Ry*Ry);
	float chi = ( std::pow(a1/b1,2) -1 )/( std::pow(a2/b2,2) +1);
	float sigma_0 = a1+a2;
	float r1 = Rx/dist;
	float r2 = Ry/dist;
	float sigma_ij = sigma_0/sqrt(1.0-(chi/2.0)*(pow(r1*cos(theta_a)+r2*sin(theta_a) + r1*cos(theta_b)+r2*sin(theta_b),2.0) / (1+chi*(cos(theta_a)*cos(theta_b) + sin(theta_a)*sin(theta_b)))+pow(r1*cos(theta_a)+r2*sin(theta_a) - r1*cos(theta_b)-r2*sin(theta_b),2.0) / (1-chi*(cos(theta_a)*cos(theta_b) + sin(theta_a)*sin(theta_b)))));
	if (dist <= sigma_ij){
		return true;
	}
	return false;
}

bool toClose(float x, float y, float a, float b, float theta, float * ABs, float * X, float * O, int N){
	for (int i = 0; i < N; i++){
		if (ellipseEllipseContact(x,X[i*2],y,X[i*2+1],a,ABs[i*2],b,ABs[i*2+1],theta,O[i])){
			return true;
		}
	}
	return false;
}

void help(){
  std::cout << "Options are: \n";
  std::cout << "-N                                           (int) Number of particles, DEFAULT 100\n";
  std::cout << "-T                                           (float) Max time, DEFAULT = 30.0, [seconds]\n";
  std::cout << "-dt                                          (float) Time step, DEFAULT = 1.0/300.0,[seconds]\n";
  std::cout << "-mur                                         (float) rotational mobility coefficient, DEFAULT = 0.0001\n";
  std::cout << "-a                                           (float) Particle major axis, DEFAULT = 1.0, this defines the length scale\n";
  std::cout << "-b                                           (float) Particle minor axis, DEFAULT = 1.0, this defines the length scale\n";
  std::cout << "Note: (a,b) = (2,2) implies a unit circle\n";
  std::cout << "-k                                           (float) Spring constant, DEFAULT = 10.0\n";
  std::cout << "-mu                                          (float) accel = f(v,x,t) + mu * sum(collision_forces(x,t)), DEFAULT = 1.0\n";
  std::cout << "-Dr                                          (float) rotational diffusion, DEFAULT = 0.0,[rad]^2[s]^-1\n";
  std::cout << "-Dt                                          (float) translational diffusion, DEFAULT = 0.00, [r]^2[s]^-1\n";
  std::cout << "-v                                           (float) v0, DEFAULT = 10.0 ,[r][s]^-1\n";
	std::cout << "-rcrit                                       (float) DEFAULT = 0, off, alignment interaction cutoff (Vicsek like)\n";
	std::cout << "-kappa                                       (float) DEFAULT = 0, off, alignment interaction strength (Vicsek like)\n";
  std::cout << "--initial-packing-fraction                   (float) density of random intial condition, DEFAULT = 0.5\n";
  std::cout << "--box-length                                 (float) length of periodic box, DEFAULT inf => no box\n";
	std::cout << "-periodic																		 (bool) DEFAULT = false, if true periodic boundary conditions, otherwise elastic collision";
	std::cout << "--random-sizes                               (bool) DEFULT = false\n";
  std::cout << "--save-every                                 (int) save state every --save-every time steps, DEFAULT = 10\n";
  std::cout << "--random-seed                                (unsigned long) DEFAULT = 31415926535897\n";
  std::cout << "-silent                                      suppress cout DEFAULT = 0 (don't suppress)\n";
}

int main(int argc, char ** argv){
  if ( (argc+1) % 2 == 0 && argc >= 1){
    // should have -OptionName Option pairs, + the program name
    for (int i = 1; i+1 < argc; i+=2){
      std::string OptionName = argv[i];
      std::string Option = argv[i+1];
      if (OptionName == "-h"){
        help();
        return 0;
      }
      else if (OptionName == "-N"){
        N = std::stoi(Option);
      }
      else if (OptionName == "-T"){
        TotalTime = std::stod(Option);
      }
      else if (OptionName == "-dt"){
        dt = std::stod(Option);
      }
      else if (OptionName == "-mur"){
        mu_r = std::stod(Option);
      }
      else if (OptionName == "-a"){
        a = std::stod(Option);
      }
      else if (OptionName == "-b"){
        b = std::stod(Option);
      }
      else if (OptionName == "-k"){
        k = std::stod(Option);
      }
      else if (OptionName == "-mu"){
        mu = std::stod(Option);
      }
      else if (OptionName == "-Dr"){
        Dr = std::stod(Option);
      }
      else if (OptionName == "-Dt"){
        Dt = std::stod(Option);
      }
      else if (OptionName == "-v"){
        v0 = std::stod(Option);
      }
			else if (OptionName == "-rcrit"){
				rcrit = std::stod(Option);
			}
			else if (OptionName == "-kappa"){
				kappa = std::stod(Option);
			}
      else if (OptionName == "--initial-packing-fraction"){
        packing = std::stod(Option);
      }
      else if (OptionName == "--box-length"){
        L = std::stod(Option);
      }
			else if (OptionName == "--random-sizes"){
        randomSizes = bool(std::stoi(Option));
      }
			else if (OptionName == "-periodic"){
				periodic = bool(std::stoi(Option));
			}
      else if (OptionName == "--save-every"){
        StepsBetweenSaves = std::stoi(Option);
      }
      else if (OptionName == "--random-seed"){
        seed = std::stoi(Option);
      }
      else if (OptionName == "-silent"){
        silent = std::stoi(Option);
      }
    }
  }
  else{
    std::cout << "Incomplete options\n";
    std::cout << "Options should be given in pairs, e.g -N 100\n";
    help();
    return 0;
  }
  if (silent == 0){
    std::cout << "#######################################\n";
    std::cout << "Parameters Set: \n";
    std::cout << "N                                      " << N << std::endl;
    std::cout << "T                                      " << TotalTime << std::endl;
    std::cout << "dt                                     " << dt << std::endl;
    std::cout << "a                                      " << a << std::endl;
    std::cout << "b                                      " << b << std::endl;
    std::cout << "force strength                         " << k << std::endl;
    std::cout << "mobility                               " << mu << std::endl;
    std::cout << "rotational mobility                    " << mu_r << std::endl;
    std::cout << "rotation-diffusion coefficient         " << Dr << std::endl;
    std::cout << "translation diffusion coefficient      " << Dt << std::endl;
    std::cout << "self propulsion speed                  " << v0 << std::endl;
		std::cout << "Alignment [cutoff, strength]           " << rcrit << ", " << kappa << std::endl;
    std::cout << "intial packing-fraction                " << packing << std::endl;
    std::cout << "box length                             " << L << std::endl;
		std::cout << "random sizes                           " << randomSizes << std::endl;
		std::cout << "periodic															 " << periodic << std::endl;
    std::cout << "save every                             " << StepsBetweenSaves << std::endl;
    std::cout << "random seed                            " << seed << std::endl;
    std::cout << "#######################################\n";
  }

  float * X; // Positions
  float * O; // orientations, theta
  float * Trajectories; // will store the answers
	float * ABs;

  X = new float [N*4];
  O = new float [N];
  ABs = new float [N*2];

  int total_steps = int(ceil(TotalTime/dt));
  Trajectories = new float [total_steps/StepsBetweenSaves*N*4]; // x,y,o,density for each N and t

  std::default_random_engine generator(seed);
  std::uniform_real_distribution<double> uniform_real(0.0, 1.0);
  std::normal_distribution<double> normal(0.0, 1.0);

  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  std::cout << "Found: " << deviceCount << " cuda devices\n";
  if (deviceCount == 0){std::cout << "" 
	  << "\n\n###################################\n\n"
	  <<     "ERROR no cuda enabled devices found\n\n"
	  <<     "###################################\n\n";}
  double r2 = a*a;
  if (a != b){
	r2 = 4.0*a*b;
  }
  // work out size of box
  double l = L;
  if (periodic){
  	if (packing == 0){
	    l = L;
	  }
	  else if (std::isnan(L) || std::isinf(L)){
	    l = sqrt((N*M_PI*r2)/packing);
			std::cout << l << std::endl;
	  }
	  else if (packing == 0 && std::isnan(L)){
	    l = std::sqrt((N*M_PI*r2)/0.5);
			std::cout << l << std::endl;
	  }
	}
  for (int i = 0; i < N; i++){
    // initialise positions with packing fraction = packing
		float ai = a;
		float bi = b;
		if (randomSizes){
			ai = uniform_real(generator)*(a-0.1*a)+0.1*a;
			bi = ai; //= uniform_real(generator)*(a-0.1*a)+0.1*a;
		}
		if (bi > ai){
			float tmp = ai;
			ai = bi;
			bi = tmp;
		}

    X[i*4] = uniform_real(generator)*(l-ai)+ai;
    X[i*4+1] = uniform_real(generator)*(l-ai)+ai;
		X[i*4+2] = 0.0;
		X[i*4+3] = 0.0;
    // random normal oritentations
    O[i] = normal(generator)*2.0*M_PI;

		ABs[i*2] = ai;
		ABs[i*2+1] = bi;
    // add particles so no two overlap, can be slow!!
		while (i > 0 && toClose(X[i*4],X[i*4+1],ai,bi,O[i],ABs,X,O,i-1)){
			float ai = a;
			float bi = b;
			if (randomSizes){
				ai = uniform_real(generator)*(a-0.1*a)+0.1*a;
				bi = ai;//= uniform_real(generator)*(a-0.1*a)+0.1*a;
			}
			if (bi > ai){
				float tmp = ai;
				ai = bi;
				bi = tmp;
			}

			X[i*4] = uniform_real(generator)*(l-ai)+ai;
			X[i*4+1] = uniform_real(generator)*(l-ai)+ai;
			X[i*4+2] = 0.0;
			X[i*4+3] = 0.0;
			// random normal oritentations
			O[i] = normal(generator)*2.0*M_PI;

			ABs[i*2] = ai;
			ABs[i*2+1] = bi;
		}
    Trajectories[0*N*4 + 4*i + 0] = X[i*4];
    Trajectories[0*N*4 + 4*i + 1] = X[i*4+1];
    Trajectories[0*N*4 + 4*i + 2] = O[i];
    Trajectories[0*N*4 + 4*i + 3] = 0.0;
  }
	std::cout << N << ", " << l << "\n";
  // hand over to CUDA
  TakeSteps(X,O,ABs,Trajectories,N,total_steps,StepsBetweenSaves,dt,k,mu,mu_r,
    Dt,Dr,v0,rcrit,kappa,l,periodic,a!=b);

  if (silent == 0){
    std::cout << "Simulation done, saving data...\n";
  }
  //set up output to save data.
  std::ostringstream namestring;
  namestring << "trajectories.txt";
  std::string str1 = namestring.str();
  std::ofstream output(str1.c_str());

  clock_t start;
	start = clock();

  for (int t = 0; t < total_steps/StepsBetweenSaves; t++){
    for (int i = 0; i < N; i++){
      output << Trajectories[t*N*4 + 4*i + 0] << ", ";
      output << Trajectories[t*N*4 + 4*i + 1] << ", ";
      output << Trajectories[t*N*4 + 4*i + 2] << ", ";
      output << Trajectories[t*N*4 + 4*i + 3];
      output << std::endl;
    }
		output << std::endl;
  }
	output.close();
  output.open("sizes.txt");

  for (int i = 0; i < N; i++){
		output << ABs[i*2] << ", " << ABs[i*2+1] << std::endl;
	}

	output.close();
  float time = (clock()-start)/(float)CLOCKS_PER_SEC;
  float rounded_down = floorf(time * 100) / 100;

  if (silent == 0){
    std::cout << "Saving data took: " << rounded_down << " s\n";
  }

  std::free(X);
  std::free(O);
	std::free(ABs);
  std::free(Trajectories);
  return 0;
}
