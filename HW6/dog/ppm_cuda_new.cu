// nvcc -O2 ppm-cuda.cu -o ppm-cuda

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <algorithm>

/********************************************
 * UTILITY: GPU ERROR CHECKING
 ********************************************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if(code != cudaSuccess){
        fprintf(stderr,"GPUassert: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        if(abort) exit(code);
    }
}

/********************************************
 * PPM IMAGE LOADING
 ********************************************/
char* data;

int read_ppm(std::string filename,
             int& width, int& height,
             std::vector<float>& r,
             std::vector<float>& g,
             std::vector<float>& b)
{
    std::ifstream in(filename.c_str(), std::ios::binary);
    if(!in.is_open()){
        std::cerr<<"Cannot open input "<<filename<<"\n";
        return 0;
    }

    std::string magic;
    in >> magic;
    if(magic != "P6"){
        std::cerr<<"Not a P6 PPM file\n";
        return 0;
    }

    // skip comments
    char c;
    in.get(c);
    while(c=='#'){
        while(in.get(c) && c!='\n');
        in.get(c);
    }
    in.unget();

    int maxcol;
    in >> width >> height >> maxcol;
    in.get(); // skip newline

    data = new char[width*height*3];
    in.read(data, width*height*3);
    in.close();

    r.resize(width*height);
    g.resize(width*height);
    b.resize(width*height);

    for(int i=0;i<width*height;i++){
        r[i] = ((unsigned char)data[3*i+0])/255.0f;
        g[i] = ((unsigned char)data[3*i+1])/255.0f;
        b[i] = ((unsigned char)data[3*i+2])/255.0f;
    }
    delete [] data;
    return 1;
}

/********************************************
 * PPM IMAGE SAVING
 ********************************************/
int write_ppm(std::string filename,
              int width,int height,
              const std::vector<float>& r,
              const std::vector<float>& g,
              const std::vector<float>& b)
{
    std::ofstream out(filename.c_str(), std::ios::binary);
    if(!out.is_open()){
        std::cerr<<"Cannot open output "<<filename<<"\n";
        return 0;
    }

    out << "P6\n# CUDA OUTPUT\n" << width << " " << height << "\n255\n";
    for(int i=0;i<width*height;i++){
        out << (unsigned char)(fminf(r[i],1.0f)*255)
            << (unsigned char)(fminf(g[i],1.0f)*255)
            << (unsigned char)(fminf(b[i],1.0f)*255);
    }
    out.close();
    return 1;
}

/********************************************
 * CUDA KERNELS
 ********************************************/

/************ Gaussian Blur kernel ************/
__global__ void blur_kernel(int w, int h, float weight,
                            const float* r_in,const float* g_in,const float* b_in,
                            float* r_out,float* g_out,float* b_out)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x>=w || y>=h) return;

    int idx = y*w + x;

    float wc = weight;
    float wn = (1.0f - weight)/4.0f;

    int xm1 = max(x-1,0);
    int xp1 = min(x+1,w-1);
    int ym1 = max(y-1,0);
    int yp1 = min(y+1,h-1);

    r_out[idx] = wc*r_in[idx] + wn*(r_in[y*w+xm1] + r_in[y*w+xp1] + r_in[ym1*w+x] + r_in[yp1*w+x]);
    g_out[idx] = wc*g_in[idx] + wn*(g_in[y*w+xm1] + g_in[y*w+xp1] + g_in[ym1*w+x] + g_in[yp1*w+x]);
    b_out[idx] = wc*b_in[idx] + wn*(b_in[y*w+xm1] + b_in[y*w+xp1] + b_in[ym1*w+x] + b_in[yp1*w+x]);
}

/************ Edge Detection Kernel ************/
__global__ void edge_kernel(int w,int h,float threshold,
                            const float* r_in,
                            float* r_out, float* g_out, float* b_out)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if(x>=w || y>=h) return;

    int xm1 = max(x-1,0);
    int xp1 = min(x+1,w-1);
    int ym1 = max(y-1,0);
    int yp1 = min(y+1,h-1);

    float gx = 0.5f*(r_in[y*w+xp1] - r_in[y*w+xm1]);
    float gy = 0.5f*(r_in[yp1*w+x] - r_in[ym1*w+x]);
    float mag = sqrtf(gx*gx + gy*gy);

    int idx = y*w + x;

    if(mag > threshold){
        r_out[idx]=g_out[idx]=b_out[idx]=1.0f;
    }else{
        r_out[idx]=g_out[idx]=b_out[idx]=0.0f;
    }
}

/************ Sine Wave Image Kernel ************/
__global__ void sine_kernel(int w,int h,float amplitude,float period,
                            float* r,float* g,float* b)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x>=w || y>=h) return;

    float yf = 0.5f*h + amplitude * sinf(2*3.1415926f*(x/period));
    int idx = y*w + x;

    if(fabsf(y-yf) < 2.0f){
        r[idx]=0; g[idx]=1; b[idx]=0;
    } else {
        r[idx]=0; g[idx]=0; b[idx]=0;
    }
}

/************ Vignette Custom Kernel ************/
__global__ void vignette_kernel(int w,int h,
                                float* r,float* g,float* b)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if(x>=w || y>=h) return;

    float cx = w/2.0f;
    float cy = h/2.0f;

    float dx = (x - cx) / (w*0.5f);
    float dy = (y - cy) / (h*0.5f);

    float d = sqrtf(dx*dx + dy*dy);
    float factor = fmaxf(1.0f - d, 0.0f);

    int idx = y*w + x;
    r[idx] *= factor;
    g[idx] *= factor;
    b[idx] *= factor;
}

/********************************************
 * OPERATIONS
 ********************************************/

void gaussian_blur_op()
{
    int w,h;
    std::vector<float> r,g,b;
    read_ppm("input.ppm",w,h,r,g,b);

    float *drA,*dgA,*dbA, *drB,*dgB,*dbB;
    size_t N = w*h*sizeof(float);

    cudaMalloc(&drA,N); cudaMalloc(&dgA,N); cudaMalloc(&dbA,N);
    cudaMalloc(&drB,N); cudaMalloc(&dgB,N); cudaMalloc(&dbB,N);

    cudaMemcpy(drA,r.data(),N,cudaMemcpyHostToDevice);
    cudaMemcpy(dgA,g.data(),N,cudaMemcpyHostToDevice);
    cudaMemcpy(dbA,b.data(),N,cudaMemcpyHostToDevice);

    dim3 tpb(16,16), bpg((w+15)/16,(h+15)/16);

    float weight = 0.2f;
    int iterations = 40;

    for(int i=0;i<iterations;i++){
        blur_kernel<<<bpg,tpb>>>(w,h,weight, drA,dgA,dbA, drB,dgB,dbB);
        cudaDeviceSynchronize();
        std::swap(drA,drB);
        std::swap(dgA,dgB);
        std::swap(dbA,dbB);
    }

    cudaMemcpy(r.data(),drA,N,cudaMemcpyDeviceToHost);
    cudaMemcpy(g.data(),dgA,N,cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(),dbA,N,cudaMemcpyDeviceToHost);

    write_ppm("blur.ppm",w,h,r,g,b);

    cudaFree(drA); cudaFree(dgA); cudaFree(dbA);
    cudaFree(drB); cudaFree(dgB); cudaFree(dbB);
}

void edge_detection_op()
{
    int w,h;
    std::vector<float> r,g,b;
    read_ppm("input.ppm",w,h,r,g,b);

    float *dr,*dg,*db,*dr2,*dg2,*db2;
    size_t N = w*h*sizeof(float);

    cudaMalloc(&dr,N); cudaMalloc(&dg,N); cudaMalloc(&db,N);
    cudaMalloc(&dr2,N); cudaMalloc(&dg2,N); cudaMalloc(&db2,N);

    cudaMemcpy(dr,r.data(),N,cudaMemcpyHostToDevice);

    dim3 tpb(16,16), bpg((w+15)/16,(h+15)/16);

    edge_kernel<<<bpg,tpb>>>(w,h,0.15f, dr, dr2,dg2,db2);

    cudaMemcpy(r.data(),dr2,N,cudaMemcpyDeviceToHost);
    cudaMemcpy(g.data(),dg2,N,cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(),db2,N,cudaMemcpyDeviceToHost);

    write_ppm("edge.ppm",w,h,r,g,b);

    cudaFree(dr); cudaFree(dg); cudaFree(db);
    cudaFree(dr2); cudaFree(dg2); cudaFree(db2);
}

void sine_wave_op()
{
    int w, h;
    std::vector<float> r, g, b;

    // Read input image to get dimensions
    read_ppm("input.ppm", w, h, r, g, b);

    size_t N = w * h * sizeof(float);

    float *dr, *dg, *db;
    cudaMalloc(&dr, N);
    cudaMalloc(&dg, N);
    cudaMalloc(&db, N);

    dim3 tpb(16,16);
    dim3 bpg((w + 15) / 16, (h + 15) / 16);

    // Automatically scale sine parameters based on image size
    float amplitude = 0.20f * h;   // 20% of image height
    float period    = 0.25f * w;   // 25% of image width

    // Launch kernel
    sine_kernel<<<bpg, tpb>>>(w, h, amplitude, period, dr, dg, db);

    // Copy result back
    cudaMemcpy(r.data(), dr, N, cudaMemcpyDeviceToHost);
    cudaMemcpy(g.data(), dg, N, cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(), db, N, cudaMemcpyDeviceToHost);

    // Save image
    write_ppm("sine.ppm", w, h, r, g, b);

    cudaFree(dr);
    cudaFree(dg);
    cudaFree(db);
}


void custom_effect_op()
{
    int w,h;
    std::vector<float> r,g,b;
    read_ppm("input.ppm",w,h,r,g,b);

    size_t N = w*h*sizeof(float);

    float *dr,*dg,*db;
    cudaMalloc(&dr,N); cudaMalloc(&dg,N); cudaMalloc(&db,N);

    cudaMemcpy(dr,r.data(),N,cudaMemcpyHostToDevice);
    cudaMemcpy(dg,g.data(),N,cudaMemcpyHostToDevice);
    cudaMemcpy(db,b.data(),N,cudaMemcpyHostToDevice);

    dim3 tpb(16,16), bpg((w+15)/16,(h+15)/16);

    vignette_kernel<<<bpg,tpb>>>(w,h, dr,dg,db);

    cudaMemcpy(r.data(),dr,N,cudaMemcpyDeviceToHost);
    cudaMemcpy(g.data(),dg,N,cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(),db,N,cudaMemcpyDeviceToHost);

    write_ppm("vignette.ppm",w,h,r,g,b);

    cudaFree(dr); cudaFree(dg); cudaFree(db);
}

/********************************************
 * MAIN DISPATCH FUNCTION
 ********************************************/
int main(int argc,char** argv)
{
    if(argc<2){
        std::cerr<<"Usage: ./ppm-cuda <blur|edge|sine|custom>\n";
        return 1;
    }

    std::string op = argv[1];
    if(op == "blur")      gaussian_blur_op();
    else if(op == "edge") edge_detection_op();
    else if(op == "sine") sine_wave_op();
    else if(op == "custom") custom_effect_op();
    else {
        std::cerr << "Unknown operation: " << op << "\n";
        return 1;
    }
    return 0;
}

