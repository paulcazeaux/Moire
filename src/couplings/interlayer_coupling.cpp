#include "interlayer_coupling.h"

// assumes sheet 1 to sheet 2 orientation, and sheet 2 is rotated by theta

const double pi2o3 	= M_PI*2/3;
const double pi1o3 	= M_PI/3;
const double pi2	= M_PI/2;
const double pi6 	= M_PI/6;
const double r_cut_graphene = 8;
const double r_cut2_graphene = 7;

// assuming (x,y) aims from sheet 1 to sheet 2 site
// theta rotates sheet counter-clockwise

Interlayer_coupling::Interlayer_coupling() {
}

Interlayer_coupling::~Interlayer_coupling() {
}

double interlayer_term(double x1_in, double y1_in, double z1_in, double x2_in, double y2_in, double z2_in, int orbit1, int orbit2, double theta1, double theta2, int mat1, int mat2) {

	if ((mat1 == 0 && mat2 == 0) || (mat1 == 5 && mat2 == 5)) {
		return inter_graphene(x1_in,y1_in,x2_in,y2_in,orbit1,orbit2,theta1,theta2);
	}
	
	if (mat1 > 0 && mat1 < 5 && mat2 > 0 && mat2 < 5){
		return inter_tmdc(x1_in,y1_in,z1_in,x2_in,y2_in,z2_in,orbit1,orbit2,mat1,mat2);
	}

	return 0;

}

void load_fftw_complex(std::vector< std::vector<fftw_complex*> > &out, std::string file) {

	std::ifstream fin(file.c_str());
	int num_orb_1;
	int num_orb_2;
	
	fin >> num_orb_1;
	fin >> num_orb_2;
	
	for (int o1 = 0; o1 < num_orb_1; ++o1){
		
		std::vector<fftw_complex*> temp_vec;
		
		for (int o2 = 0; o2 < num_orb_2; ++o2){
		
			int x_size;
			int y_size;
			
			int file_o1;
			int file_o2;
			
			int file_type;
			
			fin >> file_o1;
			fin >> file_o2;
			fin >> x_size;
			fin >> y_size;
			fin >> file_type;
			
			if (file_o1 != o1){
				printf("Warning! orbit mismatch in *fft.dat file \n");
				break;
			}
			if (file_o2 != o2){
				printf("Warning! orbit mismatch in *fft.dat file \n");
				break;
			}
			if (file_type != 0){
				printf("Warning! orbit mismatch in *fft.dat file \n");
				break;
			}
			
			// debugging print statement
			//printf("load: [x_size, y_size] = [%d, %d] \n",x_size, y_size);
			
			
			fftw_complex* temp_out;
			temp_out = (fftw_complex*) fftw_malloc(x_size*y_size*2*sizeof(fftw_complex));
	
			for (int i = 0; i < 2*x_size; i++)
				for (int j = 0; j < y_size; j++)
					fin >> temp_out[j + i*y_size][0];
			
			// following prints loaded matrix to terminal, for debugging
			/*
			for (int i = 0; i < 2*x_size; i++){
				for (int j = 0; j < y_size; j++){
					printf("%lf, ",temp_out[j + i*y_size][0]);
				}
				printf(" \n");
			}
			*/
			
			// Now get complex data for this orbit pairing
			
			fin >> file_o1;
			fin >> file_o2;
			fin >> x_size;
			fin >> y_size;
			fin >> file_type;
			
			if (file_o1 != o1){
				printf("Warning! orbit mismatch in *fft.dat file \n");
				break;
			}
			if (file_o2 != o2){
				printf("Warning! orbit mismatch in *fft.dat file \n");
				break;
			}
			if (file_type != 1){
				printf("Warning! orbit mismatch in *fft.dat file \n");
				break;
			}

			for (int i = 0; i < 2*x_size; i++)
				for (int j = 0; j < y_size; j++)
					fin >> temp_out[j + i*y_size][1];
			
			temp_vec.push_back(temp_out);
					
			
		}
		
		out.push_back(temp_vec);
		
	}

	fin.close();
}

double interp_4point(double x, double y, double v1, double v2, double v3, double v4) {
   double value = v1*(1-x)*(1-y) + v2*x*(1-y)+ v3*(1-x)*y + v4*x*y;
   return value;
}

double Interlayer_coupling::interp_fft(double x_input, double y_input, int o1, int o2, int entry) {

	// (x_input, y_input) is location at which you want to know the (interpolated) value of the fftw_complex data
	// o1, o2 are two two orbitals whose interlayer coupling you are computing
	// entry is 0 for real and 1 for imaginary part
	
	fftw_complex* data;
	data = fftw_data[o1][o2];
	
	int x_s = length_x*L_x;
	int y_s = length_y*L_y;
	double x = x_input*L_x/M_PI+x_s;
	double y = y_input*L_y/M_PI;
	
	if (y < 0)
		y = -y; // we use the y-symmetry in a FFT of purely real input data
	if (x < 0 || x > 2*x_s - 1 || y > y_s - 1) // here we do "- 1" to prevent wrap-around errors (i.e. interpolating at [2*x_s,y] would sample [0,y+1] for right-hand points!!
		return 0;
		
	int x_int = int(x);
	int y_int = int(y);

	double value = interp_4point(x-x_int,y-y_int, data[y_int + x_int*y_s][entry], data[y_int + (x_int+1)*y_s][entry], data[y_int+1 + x_int*y_s][entry],data[y_int+1+(x_int+1)*y_s][entry]);
	return value;

}

// verbose version of above call, for debugging purposes
double Interlayer_coupling::interp_fft_v(double x_input, double y_input, int o1, int o2, int entry) {

	// (x_input, y_input) is location at which you want to know the (interpolated) value of the fftw_complex data
	// o1, o2 are two two orbitals whose interlayer coupling you are computing
	// entry is 0 for real and 1 for imaginary part
	
	fftw_complex* data;
	data = fftw_data[o1][o2];
	
	int x_s = length_x*L_x;
	int y_s = length_y*L_y;
	double x = x_input*L_x/M_PI+x_s;
	double y = y_input*L_y/M_PI;
	
	printf("[x_s, y_s] = [%d, %d] \n",x_s,y_s);
	printf("[x,y] = [%lf, %lf] \n",x,y);
	
	if (y < 0)
		y = -y; // we use the y-symmetry in a FFT of purely real input data
	if (x < 0 || x > 2*x_s - 1 || y > y_s - 1) // here we do "- 1" to prevent wrap-around errors (i.e. interpolating at [2*x_s,y] would sample [0,y+1] for right-hand points!!
		return 0;
		
	
		
	int x_int = int(x);
	int y_int = int(y);
	
	printf("[x_int, y_int] = [%d, %d] \n",x_int, y_int);
	printf("[data] = [%lf, %lf, %lf, %lf] \n",data[y_int + x_int*y_s][entry], data[y_int + (x_int+1)*y_s][entry], data[y_int+1 + x_int*y_s][entry],data[y_int+1+(x_int+1)*y_s][entry]);

	double value = interp_4point(x-x_int,y-y_int, data[y_int + x_int*y_s][entry], data[y_int + (x_int+1)*y_s][entry], data[y_int+1 + x_int*y_s][entry],data[y_int+1+(x_int+1)*y_s][entry]);
	return value;

}

void Interlayer_coupling::fft_setup(int L_x_in, int L_y_in, int length_x_in, int length_y_in, std::string fftw_file_in){

	L_x = L_x_in;
	L_y = L_y_in;
	length_x = length_x_in;
	length_y = length_y_in;
	
	load_fftw_complex(fftw_data, fftw_file_in);
}

double inter_graphene(double x1_in, double y1_in, double x2_in, double y2_in, int orbit1, int orbit2, double theta1, double theta2)  {

/*
double x1 = cos(-theta1)*x1_in - sin(-theta1)*y1_in;
double y1 = sin(-theta1)*x1_in + cos(-theta1)*y1_in;

double x2 = cos(-theta1)*x2_in - sin(-theta1)*y2_in;
double y2 = sin(-theta1)*x2_in + cos(-theta1)*y2_in;
*/

double x = x2_in - x1_in;
double y = y2_in - y1_in; 

double theta12 = 0;
double theta21 = 0;

double r = sqrt(x*x+y*y);
double t = 0;

//printf("input for inter term: x = %lf, y = %lf, r  = %lf, orbit1 = %d, orbit 2 = %d, theta1 = %lf, theta2 = %lf \n",x,y,r,orbit1,orbit2,theta1,theta2);

// deal with r = 0 case first (causes problems with theta computation)
if (r == 0.0)
	return 0.3155;

if (r < r_cut_graphene){

	double ac = acos(x/r);
	if ( y < 0 )
		ac = 2*M_PI - ac;

	// theta21 (angle to bond on sheet 1)

	theta21 = ac - theta1;
	if (orbit1 == 1){
		theta21 = theta21 + pi6;
	}
	if (orbit1 == 0){
		theta21 = theta21 - pi6;
	}

	// theta12 (angle to bond on sheet 2)
	
	theta12 = ac - theta2 + M_PI;
	if (orbit2 == 1) {
		theta12 = theta12 + pi6;
	}
	if (orbit2 == 0) {
		theta12 = theta12 - pi6;
	}

	//printf("r = %lf, ac = %lf, orbit2 = %d, theta12 = %lf, orbit1 = %d, theta21 = %lf \n", r, ac, orbit2, theta12, orbit1, theta21);


	double V0 = .3155*exp(-1.7543*(r/2.46)*(r/2.46))*cos(2.001*r/2.46);
	double V3 = -.0688*(r/2.46)*(r/2.46)*exp(-3.4692*(r/2.46 - .5212)*(r/2.46-.5212));
	double V6 = -.0083*exp(-2.8764*(r/2.46-1.5206)*(r/2.46-1.5206))*sin(1.5731*r/2.46);
	
	t = V0+V3*(cos(3*theta12)+cos(3*theta21)) + V6*(cos(6*theta12)+cos(6*theta21));
	if (r > r_cut2_graphene)
	{
		double inside_cut = r_cut2_graphene - r_cut_graphene;
		double inside_cut2 = r - r_cut_graphene;
		t = t*exp(1/(inside_cut*inside_cut)-1/(inside_cut2*inside_cut2));
	}

}

//printf("coupling = %f, from input = %f, %f, %d, %d, %f, %f \n", t, x, y, orbit1, orbit2, theta1, theta2);
//printf("%f inter-term computed. \n",t);

return t;
}

double inter_tmdc(double x1_in, double y1_in, double z1_in, double x2_in, double y2_in, double z2_in, int orbit1, int orbit2, int mat1, int mat2) {
	double delta_z = z1_in - z2_in;
	
	double nu_sigma;
	double R_sigma;
	double eta_sigma;
	
	double nu_pi;
	double R_pi;
	double eta_pi;
	
	double c;
	double d_xx;
	
	// mat == 1,4: MoS2,WS2
	// This is for S-S interaction
	if ((mat1 == 1 || mat1 == 4) && ( mat2 == 1 || mat2 == 4)){
		nu_sigma 	=  2.627;
		R_sigma 	=  3.128;
		eta_sigma 	=  3.859;
		
		nu_pi 		= -0.708;
		R_pi 		=  2.923;
		eta_pi 		=  5.724;
		
		c 			=  12.29;
		d_xx		=  3.130;
	}
	// mat == 2,3: WSe2,MoSe2 bilayer
	// This is for Se-Se interaction
	
	else if ((mat1 == 2 || mat1 == 3) && (mat2 == 2 || mat2 == 3)){
		nu_sigma 	=  2.559;
		R_sigma 	=  3.337;
		eta_sigma 	=  4.114;
		
		nu_pi 		= -1.006;
		R_pi 		=  2.927;
		eta_pi 		=  5.185;
		
		c 			=  12.96;
		d_xx		=  3.350;
	}

	double XX_sep = (c/2.0) - d_xx;
	int p_i = 0;
	int p_j = 0;
	
	double r_vec[3] = {x2_in - x1_in, y2_in - y1_in, z2_in - z1_in};
	double r_sq = r_vec[0]*r_vec[0] + r_vec[1]*r_vec[1] + r_vec[2]*r_vec[2];
	double r = sqrt(r_sq);
	
	if (r > 7){
		return 0;
	}
	
	double temp_t = 0;
	
	if (delta_z < 0) {
		delta_z = -delta_z;
	}
	
	double delta = 0.05;
	if ((delta_z > XX_sep - delta) && (delta_z < XX_sep + delta)) {
	
		if (orbit1 == 5 || orbit1 == 8) {
			p_i = 0;
		}
		if (orbit1 == 6 || orbit1 == 9) {
			p_i = 1;
		}
		if (orbit1 == 7 || orbit1 == 10) {
			p_i = 2;
		}
		
		if (orbit2 == 5 || orbit2 == 8) {
			p_j = 0;
		}
		if (orbit2 == 6 || orbit2 == 9) {
			p_j = 1;
		}
		if (orbit2 == 7 || orbit2 == 10) {
			p_j = 2;
		}
		
		double V_sigma = nu_sigma*exp(-pow(r/R_sigma, eta_sigma));
		double V_pi    =    nu_pi*exp(-pow(r/R_pi,    eta_pi   ));
		
		temp_t += (V_sigma - V_pi)*((r_vec[p_i]*r_vec[p_j])/r_sq);
		if (p_i == p_j){
			temp_t += V_pi;
		}
		
		return temp_t;
		
	}

	return 0;
	
}
