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

double inter_graphene(double x, double y, int orbit1, int orbit2, double theta1, double theta2)
{

double theta12 = 0;
double theta21 = 0;

double r = sqrt(x*x+y*y);
double t = 0;

// deal with r = 0 case first (causes problems with theta computation)
if (x == 0.0 && y == 0.0)
	return 0.3155;

if (r < r_cut_graphene)
{
double ac = acos(x/r);

if ( y < 0 )
	ac = 2*M_PI - ac;
	
double theta21 = ac - theta1;
double theta12 = ac - theta2 + M_PI;

// theta21 (angle to bond on sheet 1)

if (orbit1 == 1) {

	while (theta21 >= pi2){
		theta21 -= pi2o3;
		}
	while (theta21 < -pi6){
		theta21 += pi2o3;
		}
	theta21 = theta21 + pi6;
	theta21 = pi2o3 - theta21;
}	else {
	
	while (theta21 >= M_PI - pi6){
		theta21 -= pi2o3;
		}
	while (theta21 < pi6){
		theta21 += pi2o3;
		}

	theta21 = theta21 - pi6;
	theta21 = pi2o3 - theta21;
}

// theta12 (angle to bond on sheet 2)

if (orbit2 == 1) {

	while (theta12 >= pi2){
		theta12 -= pi2o3;
		}
	while (theta12 < -pi6){
		theta12 += pi2o3;
		}
	theta12 = theta12 + pi6;
	theta12 = pi2o3 - theta12;
}	else {
	
	while (theta12 >= M_PI - pi6){
		theta12 -= pi2o3;
		}
	while (theta12 < pi6){
		theta12 += pi2o3;
		}

	theta12 = theta12 - pi6;
	theta12 = pi2o3 - theta12;
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

//printf("interlayer_coupling input = %f, %f, %d, %d, %f \n", x, y, orbit1, orbit2, theta);
//printf("%f inter-term computed. \n",t);

return t;
}
