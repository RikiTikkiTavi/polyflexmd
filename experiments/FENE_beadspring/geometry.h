#ifndef GEOMETRY_H
#define GEOMETRY_H
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <math.h>


template<class T> inline T sqr(const T& x){return x*x;}

enum   Direction { X,Y,Z };

inline Direction next(const Direction d)
{
  switch(d)
  {
  case X: return Y;  
  case Y: return Z;  
  case Z: return X;
  }
}

/*
inline ostream& operator<<(ostream& os,const Direction d)
{
  switch(d)
  {
  case X: return os << "X";  
  case Y: return os << "Y";  
  case Z: return os << "Z";
  }  
}
*/
 //istream& operator>>(istream& is,Direction& d);


template <class Base> class Point
{
  Base x;
  Base y;
  Base z;
public:
  Point() {  x = y = z = 0; }
  Point(int n) {  x = y = z = 0; }
  Point(const Base xx,const Base yy,const Base zz)
  { x = xx; y = yy; z = zz; }

void operator=(const Point& pp)
{
  x = pp.x;
  y = pp.y;
  z = pp.z;
}

 const bool operator==(const Point& pp) const {
     return x==pp.x && y==pp.y && z==pp.z;
 }

Base operator()(Direction d) const
{
  switch(d)
  {
  case X: return x;
  case Y: return y;
  case Z: return z;    
  }
  
}

Base &operator[] (int i) {
  switch(i) {
  case 0: return x;
  case 1: return y;
  case 2: return z;
  default:
//    cout << "Wrong index in Point: " << i << endl;
    exit(1);
  }
  return x;     // Is a dummy return value
}


void init(Base c) {
  x = y = z = c; 
}

void normalize()   
{
  Base r = sqrt(x*x+y*y+z*z);
  x /= r;
  y /= r;
  z /= r;
}


double norm() { return sqrt((double )(*this).dot(*this)); }

const Point& operator*=(const Base r) 
{
  x *= r;
  y *= r;
  z *= r;
  return *this;
}

const Point& operator-=(const Point& p) 
{
  x -= p.x;
  y -= p.y;
  z -= p.z;
  return *this;
}

const Point& operator+=(const Point& p) 
{
  x += p.x;
  y += p.y;
  z += p.z;
  return *this;
}

Base   operator*(const Point& p) const { return x*p.x + y * p.y + z *p.z; }
Point  operator+(const Point& p) const { return Point(x+p.x,y+p.y,z+p.z);  }
Point  operator-(const Point& p) const { return Point(x-p.x,y-p.y,z-p.z);  }
Base   dist2(const Point& p) const
       { return sqr(x-p.x) + sqr(y-p.y) + sqr(z-p.z); }

Base dot(const Point &w) {
  return (x*w.x + y*w.y + z*w.z);
}

friend Base cosTheta(Point v1, Point v2) {
    double cosT =  v1.dot(v2)/(v1.norm()*v2.norm());
    if (cosT > 1.0) cosT = 1.0;
    if (cosT < -1.0) cosT = -1.0;
    return (cosT);
}


friend Point operator * (Base c, const Point &w) {
  return Point(w.x*c, w.y*c, w.z*c);
}   

friend Point operator * (const Point &w, Base c) {
  return Point(w.x*c, w.y*c, w.z*c);
}  
  
friend Point operator / (const Point &w, Base c) {
  return Point(w.x/c, w.y/c, w.z/c);
}  


//b.b
bool operator<(const Point& p)const {
    if(x!=p.x) return x<p.x;
    if(y!=p.y) return y<p.y;
    return z<p.z;
}

bool operator>(const Point& p) const {
    return p<(*this);
}

Point Outer(const Point& p) const { // this X p
    return Point(y*p.z-z*p.y, z*p.x-x*p.z, x*p.y-y*p.x);
}


Base norm2() { return ((*this).dot(*this)); }


const Point& operator/=(Base p) 
{
  x /= p;
  y /= p;
  z /= p;
  return *this;
}

    
};


/*
template <class Base> istream& operator>>(istream& is,Point<Base>& p)
{
  Base x,y,z;
  is >> x >> y >> z;
  p = Point<Base>(x,y,z);
  return is;
}
*/


 //template <class Base> inline
/* 
ostream& operator<<(ostream& os,const Point<Base>& p)  
{
  return os << setw(8) << p(X) << " " << setw(8) << p(Y) 
	    << " " << setw(8) << p(Z);
}
*/


typedef Point<int>    GridPoint;
typedef Point<double> Coord;

class Edge {
 public:
    Coord orig;
    Coord dest;
    Edge (Coord &_orig, Coord &_dest) {
	orig = _orig;
	dest = _dest;
    }
    Edge (void) {}
    ~Edge() {}

    double distance(Coord p, double &t) {
    // evaluates closest distance from edge to p
        Coord diff = dest-orig;
        t = ((p-orig)*diff)/(diff*diff);
        if (t < 0.0) t = 0.0;
        else if (t > 1.0) t = 1.0;
        Coord closestP = orig + t*(dest-orig);
	return sqrt(closestP.dist2(p));
    }

     double distance2(Coord p) {
    // evaluates closest distance from edge to p
	double t;
        Coord diff = dest-orig;
        t = ((p-orig)*diff)/(diff*diff);
        if (t < 0.0) t = 0.0;
        else if (t > 1.0) t = 1.0;
        Coord closestP = orig + t*(dest-orig);
	return closestP.dist2(p);
    }


    double distanceB(Coord p, double &t) {
    // evaluates closest distance from edge to p
        Coord diff = dest-orig;
        t = ((p-orig)*diff)/(diff*diff);
        Coord closestP = orig + t*(dest-orig);
	return sqrt(closestP.dist2(p));
    }
    Coord point(double t) {
	return (orig + t*(dest-orig));
    }
    double angle(Edge e) {
        Coord b1 = this->dest - this->orig;
        Coord b2 = e.dest - e.orig;
        double cosT = cosTheta(b1, b2);
        return acos(cosT);
    }
    double distance(Edge e) {
    // evaluates closest distance between 2 edges
        double t1, t2, d, d2, tdum;
	Coord a1 = this->orig;
        Coord b1 = this->dest;
        Coord a2 = e.orig;
        Coord b2 = e.dest; 
        double b2ma2sq = (b2-a2)*(b2-a2);
        double b1ma1sq = (b1-a1)*(b1-a1);
        double b2ma2b1ma1 = (b2-a2)*(b1-a1);
        double b2ma2b1ma1sq = b2ma2b1ma1*b2ma2b1ma1;
        double a2ma1b1ma1 = (a2-a1)*(b1-a1);
        double a2ma1b2ma2 = (a2-a1)*(b2-a2);
        double denom = (b2ma2sq-b2ma2b1ma1sq/b1ma1sq);
        if (fabs(denom) < 1.0e-9) {  // edges are parallel
	    double min = e.distance(a1, tdum);
            d = e.distance(b1, tdum);
            if (d < min) min = d;
            d2 = min*min;
	}
	else{ 
	    t2 = (a2ma1b1ma1*b2ma2b1ma1/b1ma1sq-a2ma1b2ma2)/denom;
	    t1 = (a2ma1b1ma1 + t2*b2ma2b1ma1)/b1ma1sq;
	    if (t1 > 1.0) t1 = 1.0;
	    if (t1 < 0.0) t1 = 0.0;
	    if (t2 > 1.0) t2 = 1.0;
	    if (t2 < 0.0) t2 = 0.0;
	    d2 = (this->point(t1)).dist2(e.point(t2));
	}
        return sqrt(d2); 
    }
};



#endif

