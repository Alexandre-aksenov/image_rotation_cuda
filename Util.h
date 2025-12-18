
#define SafeCall(err)     cudaSafeCall(err, __FILE__, __LINE__)
#define LastError(err)     GetLastError(err, __FILE__, __LINE__)

//////////////////////////////////////////////////////////////////////////////////////
inline void cudaSafeCall( cudaError err, const char *file, const int line )
{
    if( cudaSuccess != err) {
		fprintf(stderr, "%s(%i) : cudaSafeCall() Runtime API error %d: %s.\n",\
                file, line, (int)err, cudaGetErrorString( err) );
        exit(-1);
    }
}

inline void GetLastError( const char *errorMessage, const char *file, const int line )
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
        fprintf(stderr, "%s(%i) : CheckMsg() CUDA error : %s : (%d) %s.\n",\
                file, line, errorMessage, (int)err, cudaGetErrorString( err ) );
        exit(-1);
    }
}

// binary file > float array

void load_image(char *name,float** h,int x,int y) {
    unsigned int s= x * y;

	*h = (float*) malloc(s*sizeof(float));	// allocation
	FILE *fp;
	unsigned char * lect=NULL;
	lect = (unsigned char *) malloc(s);

	fp = fopen(name, "rb");
	if (!fp) {
		fprintf(stderr, "Erreur: impossible d'ouvrir %s.\n",name);
		exit(1);
	}	
    if (!fread((void *) lect, s, 1, fp)) {
        printf("Unable to read source image file: %s\n", name);
        exit(-1);
    }
	fclose(fp);	
	for (int j=0; j<s; j++) (*h)[j]=(float)lect[j]/256.0;
    printf("Loaded '%s', %d x %d pixels\n", name, x, y);
	free(lect);
}

// binary file > char array

void load_image_c(char *name,unsigned char** h,int x,int y) {
    unsigned int s= x * y;

	*h = (unsigned char*) malloc(s);
	FILE *fp;
	fp = fopen(name, "rb");
	if (!fp) {
		fprintf(stderr, "Erreur: impossible d'ouvrir %s.\n",name);
		exit(1);
	}	
    if (!fread((void *) (*h), s, 1, fp)) {
        printf("Unable to read source image file: %s\n", name);
        exit(-1);
    }
	fclose(fp);	
    printf("Loaded '%s', %d x %d pixels\n", name, x, y);
}

// float array -> output file

void write_image(char *name,float* h,int x,int y){

    char* pgm_header="P5%c%d%c%d%c255%c\0";
    char buf[512];

    sprintf( buf, pgm_header, 0x0a, x, 0x0a, y, 0x0a, 0x0a );

    unsigned int s= x * y;
    FILE *fw;
    fw = fopen(name, "wb");

    unsigned char * lect = (unsigned char *) malloc(s);

    for (int j=0; j<s; j++) lect[j]=(unsigned char)(h[j]*256.0);

    if (!fw) {
        fprintf(stderr, "Erreur: impossible d'ouvrir %s.\n",name);
        exit(1);
    }    

    fwrite((void*)buf, strlen( buf ), 1, fw );

    if (!fwrite((void *) lect, s, 1, fw)) {
        printf("Unable to write res image file: %s\n", name);
        exit(-1);
    }
    fclose(fw);
    printf("Saved '%s', %d x %d pixels\n", name, x, y);
    free(lect);
}
