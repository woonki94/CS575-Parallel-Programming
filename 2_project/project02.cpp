#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

const float GRAIN_GROWS_PER_MONTH =	       12.0;
const float ONE_DEER_EATS_PER_MONTH =		1.0;
const int ONE_WOLF_EATS_PER_MONTH = 		2; 
const float AVG_PRECIP_PER_MONTH =		7.0;	// average
const float AMP_PRECIP_PER_MONTH =		6.0;	// plus or minus
const float RANDOM_PRECIP =			2.0;	// plus or minus noise
const float AVG_TEMP =				60.0;	// average
const float AMP_TEMP =				20.0;	// plus or minus
const float RANDOM_TEMP =			10.0;	// plus or minus noise
const float MIDTEMP =				40.0;
const float MIDPRECIP =				10.0;
const int CURRENT_YEAR =            2024;

int NowYear;
int NowMonth;
float NowPrecip; //inches of rain per month
float NowTemp;
float NowHeight;
int NowNumDeer; 
int NowNumWolf;

omp_lock_t	Lock;
volatile int	NumInThreadTeam;
volatile int	NumAtBarrier;
volatile int	NumGone;

void	InitBarrier( int );
void	WaitBarrier( );

//Random number generator.
float
Ranf( float low, float high )
{
    float r = (float) rand();               // 0 - RAND_MAX
    float t = r  /  (float) RAND_MAX;       // 0. - 1.

    return   low  +  t * ( high - low );
}

void
TimeOfDaySeed( )
{
	struct tm y2k = { 0 };
	y2k.tm_hour = 0;   y2k.tm_min = 0; y2k.tm_sec = 0;
	y2k.tm_year = 100; y2k.tm_mon = 0; y2k.tm_mday = 1;

	time_t  timer;
	time( &timer );
	double seconds = difftime( timer, mktime(&y2k) );
	unsigned int seed = (unsigned int)( 1000.*seconds );    // milliseconds
	srand( seed );
}

float
SQR( float x )
{
    return x*x;
}


void
Wolf()
{

     while( NowYear < 2030 )
        {
        int NextWolfNum = NowNumWolf;
        //wolf eats 2 deers per month. 
        //if there is plenty of deer to eat for every wolf, number of wolf will increase by one.
        //else if there is not enough deer, number of wolf will decrease by one.
        int WolfCarryingCapacity = NowNumDeer/2; 
        if(NextWolfNum < WolfCarryingCapacity) 
            NextWolfNum++;
        else if((NextWolfNum > WolfCarryingCapacity) && NextWolfNum >0)
            NextWolfNum--;

        WaitBarrier();

        NowNumWolf = NextWolfNum;

        WaitBarrier();  

        WaitBarrier();


    }

}


void
Deer()
{
    while( NowYear < 2030 )
        {
        // compute a temporary next-value for this quantity
        // based on the current state of the simulation:
        int nextNumDeer = NowNumDeer;
        int carryingCapacity = (int)( NowHeight );
        // fprintf(stderr, "carryingCapacity : %8d\n", carryingCapacity );

        if( nextNumDeer < carryingCapacity ){
            nextNumDeer++;
            //fprintf(stderr, "increament of deer");
        }
        else if(nextNumDeer > carryingCapacity)
                nextNumDeer--;
        
        nextNumDeer -= ONE_WOLF_EATS_PER_MONTH * NowNumWolf;

        if(nextNumDeer < 0)
            nextNumDeer = 0;

        WaitBarrier();

        // fprintf(stderr, "nextNum of deer:%8d\n", nextNumDeer);
        NowNumDeer = nextNumDeer;   

        WaitBarrier();

        WaitBarrier();


    }

    
}

void
Grain()
{

     while( NowYear < 2030 )
        {
        // compute a temporary next-value for this quantity
        // based on the current state of the simulation:
        float ang = (  30.*(float)NowMonth + 15.  ) * ( M_PI / 180. );	// angle of earth around the sun

        float temp = AVG_TEMP - AMP_TEMP * cos( ang );
        NowTemp = temp + Ranf( -RANDOM_TEMP, RANDOM_TEMP );

        float precip = AVG_PRECIP_PER_MONTH + AMP_PRECIP_PER_MONTH * sin( ang );
        NowPrecip = precip + Ranf( -RANDOM_PRECIP, RANDOM_PRECIP );

        if( NowPrecip < 0. )
            NowPrecip = 0.;

        float tempFactor = exp(   -SQR(  ( NowTemp - MIDTEMP ) / 10.  )   );
        float precipFactor = exp(   -SQR(  ( NowPrecip - MIDPRECIP ) / 10.  )   );
        float nextHeight = NowHeight;

        nextHeight += tempFactor * precipFactor * GRAIN_GROWS_PER_MONTH;
        nextHeight -= (float)NowNumDeer * ONE_DEER_EATS_PER_MONTH;


        WaitBarrier();

        if(nextHeight > 0.)
            NowHeight = nextHeight;
        else
            NowHeight = 0.;
                
                
        WaitBarrier();  

        WaitBarrier();

    }   

}

void
Watcher()
{
    
    int tempMonth;
    int tempYear;
    float tempTemp;
    float tempPrecip;
    unsigned int seed = 0;

    while( NowYear < 2030 )
    {

        // DoneComputing barrier:
        WaitBarrier();
        // DoneAssigning barrier:
        WaitBarrier();      

        float celTemp = (5./9.)*(NowTemp-32);
        float cmPrecip = NowPrecip *2.54;
        int accMonth = (NowYear - CURRENT_YEAR) * 12 + NowMonth;

        fprintf(stderr, "%2d,%6.2f,%6.2f,%6.2f,%2d,%2d\n",
        accMonth, celTemp, cmPrecip, NowHeight, NowNumDeer,NowNumWolf);

        //increasing year and setting month to 0
        if (++NowMonth == 12)
        {
            NowYear++;
            NowMonth = 0;
        }
   
        // DonePrinting barrier:
        WaitBarrier();      
    }
    
}

int
main( int argc, char *argv[ ] )
{
    NowMonth =    0;
    NowYear  = CURRENT_YEAR;//2024

    // starting state (feel free to change this if you want):
    NowNumDeer = 30;
    NowHeight =  5.;
    NowNumWolf = 1;


    omp_set_num_threads(4);
    InitBarrier(4);
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            Deer();
        }
        #pragma omp section
        {
            Grain();
        }
        #pragma omp section
        {
            Wolf();
        }
        #pragma omp section
        {
            Watcher();
        }
    }

    return 0;

}

void
InitBarrier( int n )
{
        NumInThreadTeam = n;
        NumAtBarrier = 0;
	omp_init_lock( &Lock );
}


void
WaitBarrier( )
{
    omp_set_lock( &Lock );
    {
            NumAtBarrier++;
            if( NumAtBarrier == NumInThreadTeam )
            {
                NumGone = 0;
                NumAtBarrier = 0;
                // let all other threads get back to what they were doing
                // before this one unlocks, knowing that they might immediately
                // call WaitBarrier( ) again:
                while( NumGone != NumInThreadTeam-1 );
                omp_unset_lock( &Lock );
                return;
            }
    }
    omp_unset_lock( &Lock );

    while( NumAtBarrier != 0 );	// this waits for the nth thread to arrive

    #pragma omp atomic
    NumGone++;			// this flags how many threads have returned
}