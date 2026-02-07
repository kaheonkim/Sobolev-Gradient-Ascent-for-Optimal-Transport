#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;



class convex_hull{
public:
    int* indices;
    int  hullCount;

    convex_hull(int n){
        indices=new int[n];
        hullCount=0;
    }

    ~convex_hull(){
        delete[] indices;
    }
};



// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---



class BFM3D2{
public:
int n1;
int n2;
int n3;

double totalMass;

double *xMap;
double *yMap;
double *zMap;

double *rho;

int *argmin;
double *temp;
    
convex_hull* hull;

    BFM3D2(int n1, int n2, int n3, py::array_t<double> & mu_np){

        py::buffer_info mu_buf = mu_np.request();
        double *mu = static_cast<double *>(mu_buf.ptr);

        this->n1 = n1;
        this->n2 = n2;
        this->n3 = n3;

        int n = fmax(fmax(n1,n2),n3);
        hull   = new convex_hull(n);    
        argmin = new int[n];
        temp   = new double[n1*n2*n3];

        xMap=new double[(n1+1)*(n2+1)*(n3+1)];
        yMap=new double[(n1+1)*(n2+1)*(n3+1)];
        zMap=new double[(n1+1)*(n2+1)*(n3+1)];
        for (int i=0;i<n3+1;i++){
            for(int j=0;j<n2+1;j++){
                for(int k=0;k<n1+1;k++){
                    double x=k/(n1*1.0);
                    double y=j/(n2*1.0);
                    double z=i/(n3*1.0);
                    
                    xMap[i * (n1 + 1) * (n2 + 1) + j * (n1 + 1) + k] = x;
                    yMap[i * (n1 + 1) * (n2 + 1) + j * (n1 + 1) + k] = y;
                    zMap[i * (n1 + 1) * (n2 + 1) + j * (n1 + 1) + k] = z;
                }
            }
        }
        
        rho=new double[n1*n2*n3];
        memcpy(rho,mu,n1*n2*n3*sizeof(double));

        totalMass = 0;
        for(int i=0;i<n1*n2*n3;++i){
            totalMass += mu[i];
        }
        totalMass /= n1*n2*n3;

    }

    ~BFM3D2(){
        delete [] xMap;
        delete [] yMap;
        delete [] zMap;
        delete [] rho;
        delete hull;
    }

    void ctransform(py::array_t<double> & dual_np, py::array_t<double> & phi_np){

        py::buffer_info phi_buf  = phi_np.request();
        py::buffer_info dual_buf = dual_np.request();

        double *phi  = static_cast<double *> (phi_buf.ptr);
        double *dual = static_cast<double *> (dual_buf.ptr);

        compute_3d_dual_inside(dual, phi, hull, n1, n2 ,n3);
    }

    void pushforward(py::array_t<double> & rho_np, py::array_t<double> & phi_np, py::array_t<double> & nu_np){

        py::buffer_info phi_buf  = phi_np.request();
        py::buffer_info nu_buf   = nu_np.request();

        double *phi = static_cast<double *> (phi_buf.ptr);
        double *nu  = static_cast<double *> (nu_buf.ptr);

        calc_pushforward_map(phi);
        sampling_pushforward(nu);

        py::buffer_info rho_buf  = rho_np.request();
        memcpy(static_cast<double *> (rho_buf.ptr), rho, n1*n2*n3*sizeof(double));
    }


    double compute_w2(py::array_t<double> & phi_np, py::array_t<double> & dual_np, py::array_t<double> & mu_np, py::array_t<double> & nu_np){

        py::buffer_info phi_buf  = phi_np.request();
        py::buffer_info dual_buf = dual_np.request();
        py::buffer_info mu_buf   = mu_np.request();
        py::buffer_info nu_buf   = nu_np.request();

        double *phi  = static_cast<double *> (phi_buf.ptr);
        double *dual = static_cast<double *> (dual_buf.ptr);
        double *mu   = static_cast<double *> (mu_buf.ptr);
        double *nu   = static_cast<double *> (nu_buf.ptr);
        
        int pcount=n1*n2*n3;
        
        double value=0;
        
        for(int i=0;i<n3;i++){
            for(int j=0;j<n2;j++){
                for(int k=0;k<n1;k++){
                    double x=(k+.5)/(n1*1.0);
                    double y=(j+.5)/(n2*1.0);
                    double z=(i+.5)/(n3*1.0);
                    
                    value+=.5*(x*x+y*y+z*z)*(mu[i*n1*n2+j*n1+k]+nu[i*n1*n2+j*n1+k])-nu[i*n1*n2+j*n1+k]*phi[i*n1*n2+j*n1+k]-mu[i*n1*n2+j*n1+k]*dual[i*n1*n2+j*n1+k];
                }
            }
        }

        value/=pcount;
        
        return value;
    }








    void compute_2d_dual(double *dual, double *u, double *temp, convex_hull *hull, int *argmin, int n1, int n2){
    
        int pcount=n1*n2;
        
        
        memcpy(temp, u, pcount*sizeof(double));
        
        
        
        for(int i=0;i<n2;i++){
            compute_dual(&dual[i*n1], &temp[i*n1], argmin, hull, n1);
            
        }
        transpose_doubles_2d(temp, dual, n1, n2);
        for(int i=0;i<n1*n2;i++){
            dual[i]=-temp[i];
        }
        for(int j=0;j<n1;j++){
            compute_dual(&temp[j*n2], &dual[j*n2], argmin, hull, n2);
            
        }
        transpose_doubles_2d(dual, temp, n2, n1);
    
    
    }

    

    void compute_3d_dual_inside(double *dual, double *u, convex_hull *hull, int n1, int n2, int n3){
        
        int pcount=n1*n2*n3;
        
        int n=fmax(fmax(n1,n2),n3);;
        
        
        memcpy(temp, u, pcount*sizeof(double));
        
        
        for(int k=0;k<n3;k++){
            compute_2d_dual(&dual[k*n1*n2], &u[k*n1*n2], &temp[k*n1*n2], hull, argmin, n1, n2);
        }
        
        transpose_doubles_3d(temp, dual, n1, n2, n3);
        
        for(int i=0;i<n1*n2*n3;i++){
            dual[i]=-temp[i];
        }
        
        for(int i=0;i<n1*n2;i++){
            compute_dual(&temp[i*n3], &dual[i*n3], argmin, hull, n3);
        }
        
        transpose_doubles_3d(dual, temp, n3, n2, n1);

    }



    void compute_dual(double *dual, double *u, int *dualIndicies, convex_hull *hull, int n){
        // *dual, *u pointer and indicates n contiguous section of memory
        
        get_convex_hull(u, hull, n);
       
        
        compute_dual_indices(dualIndicies, u, hull, n);
        
        for(int i=0;i<n;i++){
            double s=(i+.5)/(n*1.0);
            int index=dualIndicies[i];
            double x=(index+.5)/(n*1.0);
            double v1=s*x-u[dualIndicies[i]];
            double v2=s*(n-.5)/(n*1.0)-u[n-1];
            if(v1>v2){
                dual[i]=v1;
            }else{
                dualIndicies[i]=n-1;
                dual[i]=v2;
            }
            
        }
        
    }


    int sgn(double x){
        
        int truth=(x>0)-(x<0);
        return truth;
        
    }


    void transpose_doubles_2d(double *transpose, double *data, int n1, int n2){
    
    for(int i=0;i<n2;i++){
        for(int j=0;j<n1;j++){
            transpose[j*n2+i]=data[i*n1+j];
            }
        }
    }


    void transpose_doubles_3d(double *transpose, double *data, int n1, int n2, int n3){
        
        for(int k=0;k<n3;k++){
            for(int i=0;i<n2;i++){
                for(int j=0;j<n1;j++){
                    
                    transpose[j*n3*n2+i*n3+k]=data[k*n1*n2+i*n1+j];
                }
            }
        }
        
    }
    void get_convex_hull(double *u, convex_hull *hull, int n){
        
        hull->indices[0]=0;
        hull->indices[1]=1;
        hull->hullCount=2;
        
        for(int i=2;i<n;i++){
            add_point(u, hull, i);
        }
    }


    void add_point(double *u, convex_hull *hull, int i){
        
        
        if(hull->hullCount<2){
            hull->indices[1]=i;
            hull->hullCount++;
        }else{
            int hc=hull->hullCount;
            int ic1=hull->indices[hc-1];
            int ic2=hull->indices[hc-2];
            
            double oldSlope=(u[ic1]-u[ic2])/(ic1-ic2);
            double slope=(u[i]-u[ic1])/(i-ic1);
            
            if(slope>=oldSlope){
                int hc=hull->hullCount;
                hull->indices[hc]=i;
                hull->hullCount++;
            }else{
                hull->hullCount--;
                add_point(u, hull, i);
            }
        }
    }

    
    double interpolate_function(double *function, double x, double y, double z, int n1, int n2, int n3){
        
        int xIndex=fmin(fmax(x*n1-.5 ,0),n1-1);
        int yIndex=fmin(fmax(y*n2-.5 ,0),n2-1);
        int zIndex=fmin(fmax(z*n3-.5 ,0),n3-1);
        
        double xfrac=x*n1-xIndex-.5;
        double yfrac=y*n2-yIndex-.5;
        double zfrac=z*n3-zIndex-.5;
        
        int xOther=xIndex+sgn(xfrac);
        int yOther=yIndex+sgn(yfrac);
        int zOther=zIndex+sgn(zfrac);
        
        xOther=fmax(fmin(xOther, n1-1),0);
        yOther=fmax(fmin(yOther, n2-1),0);
        zOther=fmax(fmin(zOther, n3-1),0);


        // Trilinear Interpolation
        double v1=(1-fabs(xfrac))*(1-fabs(yfrac))*(1-fabs(zfrac))*function[zIndex*n1*n2+yIndex*n1+xIndex];
        double v2=fabs(xfrac)*(1-fabs(yfrac))*(1-fabs(zfrac))*function[zIndex*n1*n2+yIndex*n1+xOther];
        double v3=(1-fabs(xfrac))*fabs(yfrac)*(1-fabs(zfrac))*function[zIndex*n1*n2+yOther*n1+xIndex];
        double v4=fabs(xfrac)*fabs(yfrac)*(1-fabs(zfrac))*function[zIndex*n1*n2+yOther*n1+xOther];

        double v5= (1-fabs(xfrac))*(1-fabs(yfrac))*(fabs(zfrac))*function[zOther*n1*n2+yIndex*n1+xIndex];
        double v6= fabs(xfrac)*(1-fabs(yfrac))*(fabs(zfrac))*function[zOther*n1*n2+yIndex*n1+xOther];
        double v7= (1-fabs(xfrac))*fabs(yfrac)*(fabs(zfrac))*function[zOther*n1*n2+yOther*n1+xIndex];
        double v8= fabs(xfrac)*fabs(yfrac)*(fabs(zfrac))*function[zOther*n1*n2+yOther*n1+xOther];
        
        double v=v1+v2+v3+v4+v5+v6+v7+v8;
        
        return v;
        
    }




    void compute_dual_indices(int *dualIndicies, double *u, convex_hull *hull, int n){
        
        int counter=1;
        int hc=hull->hullCount;
        
        for(int i=0;i<n;i++){
           
            double s=(i+.5)/(n*1.0);
            int ic1=hull->indices[counter];
            int ic2=hull->indices[counter-1];
            
            double slope=n*(u[ic1]-u[ic2])/(ic1-ic2);
            while(s>slope&&counter<hc-1){
                counter++;
                ic1=hull->indices[counter];
                ic2=hull->indices[counter-1];
                slope=n*(u[ic1]-u[ic2])/(ic1-ic2);
            }
            dualIndicies[i]=hull->indices[counter-1];
            
        }
    }


    void calc_pushforward_map(double *dual){
        
        
        double xStep=1.0/n1;
        double yStep=1.0/n2;
        double zStep=1.0/n3;
        
        for(int i=0;i<n3+1;i++){
            for(int j=0;j<n2+1;j++){
                for(int k=0;k<n1+1;k++){
                    double x=k/(n1*1.0);
                    double y=j/(n2*1.0);
                    double z=i/(n3*1.0);
                    
                    double dualxp=interpolate_function(dual, x+xStep, y, z, n1, n2,n3);
                    double dualxm=interpolate_function(dual, x-xStep, y, z, n1, n2,n3);
                    
                    double dualyp=interpolate_function(dual, x, y+yStep,z, n1, n2, n3);
                    double dualym=interpolate_function(dual, x, y-yStep,z, n1, n2, n3);
                    
                    double dualzp=interpolate_function(dual, x, y, z+zStep, n1,n2,n3);
                    double dualzm=interpolate_function(dual, x, y, z-zStep, n1,n2,n3);
                    
                    xMap[i * (n1 + 1) * (n2 + 1) + j * (n1 + 1) + k]=.5*n1*(dualxp-dualxm);
                    yMap[i * (n1 + 1) * (n2 + 1) + j * (n1 + 1) + k]=.5*n2*(dualyp-dualym);
                    zMap[i * (n1 + 1) * (n2 + 1) + j * (n1 + 1) + k]=.5*n3*(dualzp-dualzm);
                    
                    
                }
            } 
        }   
    }
    void sampling_pushforward(double *mu){
        
        int pcount=n1*n2*n3;
        
        memset(rho,0,pcount*sizeof(double));
        
        for(int k=0;k<n3;k++){
            for(int i=0;i<n2;i++){
                for(int j=0;j<n1;j++){
                    
                    double mass=mu[k*n1*n2+i*n1+j];
                    
                    if(mass>0){
                        
                        double xStretch=0, yStretch=0, zStretch=0;
                        
                        for(int r=0;r<4;r++){
                            
                            int step1=r%2;
                            int step2=r/2;
                            
                            xStretch=fmax(xStretch, fabs(xMap[(k+step1)*(n1+1)*(n2+1)+(i+step2)*(n1+1)+j+1]-xMap[(k+step1)*(n1+1)*(n2+1)+(i+step2)*(n1+1)+j]));
                            
                            yStretch=fmax(yStretch, fabs(yMap[(k+step1)*(n1+1)*(n2+1)+(i+1)*(n1+1)+j+step2]-yMap[(k+step1)*(n1+1)*(n2+1)+i*(n1+1)+j+step2]));
                            
                            zStretch=fmax(zStretch, fabs(zMap[(k+1)*(n1+1)*(n2+1)+(i+step2)*(n1+1)+j+step1]-zMap[k*(n1+1)*(n2+1)+(i+step2)*(n1+1)+j+step1]));
                            
                            
                        }
                        
                        
                        int xSamples=fmax(n1*xStretch,1);
                        int ySamples=fmax(n2*yStretch,1);
                        int zSamples=fmax(n3*zStretch,1);
                        
                        
                        
                        
                        
                        double factor=1/(xSamples*ySamples*zSamples*1.0);
                        
                        for(int r=0;r<zSamples;r++){
                            for(int l=0;l<ySamples;l++){
                                for(int s=0;s<xSamples;s++){
                                    
                                    double a=(s+.5)/(xSamples*1.0);
                                    double b=(l+.5)/(ySamples*1.0);
                                    double c=(r+.5)/(zSamples*1.0);
                                    
                                    double xPoint=(1-c)*((1-b)*(1-a)*xMap[k*(n1+1)*(n2+1)+i*(n1+1)+j]+(1-b)*a*xMap[k*(n1+1)*(n2+1)+i*(n1+1)+j+1]+b*(1-a)*xMap[k*(n1+1)*(n2+1)+(i+1)*(n1+1)+j]+a*b*xMap[k*(n1+1)*(n2+1)+i*(n1+1)+j]);
                                    
                                    xPoint+=c*((1-b)*(1-a)*xMap[(k+1)*(n1+1)*(n2+1)+i*(n1+1)+j]+(1-b)*a*xMap[(k+1)*(n1+1)*(n2+1)+i*(n1+1)+j+1]+b*(1-a)*xMap[(k+1)*(n1+1)*(n2+1)+(i+1)*(n1+1)+j]+a*b*xMap[(k+1)*(n1+1)*(n2+1)+i*(n1+1)+j]);
                                    
                                    
                                    double yPoint=(1-c)*((1-b)*(1-a)*yMap[k*(n1+1)*(n2+1)+i*(n1+1)+j]+(1-b)*a*yMap[k*(n1+1)*(n2+1)+i*(n1+1)+j+1]+b*(1-a)*yMap[k*(n1+1)*(n2+1)+(i+1)*(n1+1)+j]+a*b*yMap[k*(n1+1)*(n2+1)+i*(n1+1)+j]);
                                    
                                    yPoint+=c*((1-b)*(1-a)*yMap[(k+1)*(n1+1)*(n2+1)+i*(n1+1)+j]+(1-b)*a*yMap[(k+1)*(n1+1)*(n2+1)+i*(n1+1)+j+1]+b*(1-a)*yMap[(k+1)*(n1+1)*(n2+1)+(i+1)*(n1+1)+j]+a*b*yMap[(k+1)*(n1+1)*(n2+1)+i*(n1+1)+j]);
                                    
                                    double zPoint=(1-c)*((1-b)*(1-a)*zMap[k*(n1+1)*(n2+1)+i*(n1+1)+j]+(1-b)*a*zMap[k*(n1+1)*(n2+1)+i*(n1+1)+j+1]+b*(1-a)*zMap[k*(n1+1)*(n2+1)+(i+1)*(n1+1)+j]+a*b*zMap[k*(n1+1)*(n2+1)+i*(n1+1)+j]);
                                    
                                    zPoint+=c*((1-b)*(1-a)*zMap[(k+1)*(n1+1)*(n2+1)+i*(n1+1)+j]+(1-b)*a*zMap[(k+1)*(n1+1)*(n2+1)+i*(n1+1)+j+1]+b*(1-a)*zMap[(k+1)*(n1+1)*(n2+1)+(i+1)*(n1+1)+j]+a*b*zMap[(k+1)*(n1+1)*(n2+1)+i*(n1+1)+j]);
                                    
                                    
                                    
                                    double X=xPoint*n1;
                                    double Y=yPoint*n2;
                                    double Z=zPoint*n3;
                                    
                                    int xIndex=X-.5;
                                    int yIndex=Y-.5;
                                    int zIndex=Z-.5;
                                    
                                    double xFrac=X-xIndex;
                                    double yFrac=Y-yIndex;
                                    double zFrac=Z-zIndex;
                                    
                                    for(int r=0;r<8;r++){
                                        
                                        int xOffset=r%2;
                                        int yOffset=(r/2)%2;
                                        int zOffset=(r/4)%2;
                                        
                                        int xidx=xIndex+xOffset;
                                        int yidx=yIndex+yOffset;
                                        int zidx=zIndex+zOffset;
                                        
                                        double xWeight=xFrac*xOffset+(1-xFrac)*(1-xOffset);
                                        double yWeight=yFrac*yOffset+(1-yFrac)*(1-yOffset);
                                        double zWeight=zFrac*zOffset+(1-zFrac)*(1-zOffset);
                                        
                                        xidx=fmin(fmax(xidx,0),n1-1);
                                        yidx=fmin(fmax(yidx,0),n2-1);
                                        zidx=fmin(fmax(zidx,0),n3-1);
                                        
                                        rho[zidx*n1*n2+yidx*n1+xidx]+=mass*factor*xWeight*yWeight*zWeight;
                                        
                                    }
                                    
                                    
                                    
                                }
                            }
                        }
                        
                        
                    }
                    
                }
            }
        }
        
        
        
        double sum=0;
        for(int i=0;i<pcount;i++){
            sum+=rho[i]/pcount;
        }
        for(int i=0;i<pcount;i++){
            rho[i]/=sum;
        }

        
    }

};

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---



PYBIND11_MODULE(w23d_, m) {
    // optional module docstring
    m.doc() = "pybind11 for w2 code";

    py::class_<BFM3D2>(m, "BFM3D2")
        .def(py::init<int, int, int, py::array_t<double> &>())
        .def("ctransform", &BFM3D2::ctransform)
        .def("pushforward", &BFM3D2::pushforward);
}
// Type the command below to form w23d.so file into the terminal
//  g++ `python3 -m pybind11 --includes` w23d_.cpp -o w23d_.so -shared -fPIC -undefined dynamic_lookup