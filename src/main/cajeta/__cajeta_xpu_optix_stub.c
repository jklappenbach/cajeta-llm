#include <stdint.h>
#define W __attribute__((weak))
W int      cajeta_xpu_optix_available(void){return 0;}
W void*    cajeta_xpu_optix_context(void){return 0;}
W void*    cajeta_xpu_optix_cuda_context(void){return 0;}
W int64_t  cajeta_xpu_optix_accel_build_aabbs(const float*a,uint32_t b){(void)a;(void)b;return 0;}
W int64_t  cajeta_xpu_optix_accel_build_triangles(const float*a,uint32_t b,uint32_t c){(void)a;(void)b;(void)c;return 0;}
W uint64_t cajeta_xpu_optix_traversable(int64_t a){(void)a;return 0;}
W uint64_t cajeta_xpu_optix_accel_boxes(int64_t a){(void)a;return 0;}
W void     cajeta_xpu_optix_accel_free(int64_t a){(void)a;}
W int      cajeta_xpu_optix_launch(const char*a,uint64_t b,const char*c,const char*d,const char*e,const char*f,const void*g,uint64_t h,uint32_t i){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;(void)i;return -1;}
W int      cajeta_xpu_optix_launch_tri(const char*a,uint64_t b,const char*c,const char*d,const char*e,const char*f,const void*g,uint64_t h,uint32_t i){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;(void)i;return -1;}
