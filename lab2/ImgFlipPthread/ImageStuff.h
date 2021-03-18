struct ImgProp {
	int Hpixels;   // width
	int Vpixels;   // height
	unsigned char HeaderInfo[54];
	unsigned long int Hbytes; // num of horiz. bytes
};

struct Pixel {
	unsigned char R;
	unsigned char G;
	unsigned char B;
};

typedef unsigned char pel;    // pixel element

pel** ReadBMP(char*);         // Load a BMP image
void WriteBMP(pel**, char*);  // Store a BMP image

extern struct ImgProp ip;
