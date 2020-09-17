typedef double db;
int sgn( db x) {return x < - EPS ? - 1 : x > EPS ;}

db Uniform(db x=.0)
{
	return x+(db)(1.0+rand())/(RAND_MAX+2.0);
}

db Lap( db miu = .0, db lambda = 1.0) 
{
	db U = Uniform(- 0.5) ;
	return miu - lambda * sgn(U) * log( fabs(1.0 - 2* fabs(U)) ) ;
}