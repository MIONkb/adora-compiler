        
#include <stdint.h>

void gray(uint32_t* src, uint32_t* dst) { 
    #pragma scop
	for (int i = 0; i < 1280 * 720; i++) {
		uint32_t pixel = src[i];
		// Extract components (Little Endian: R is LSB)
		uint32_t r = pixel & 0xFF;
		uint32_t g = (pixel >> 8) & 0xFF;
		uint32_t b = (pixel >> 16) & 0xFF;
		uint32_t a = (pixel >> 24) & 0xFF;
		// Calculate Grayscale using luminosity formula
		// Gray = 0.299*R + 0.587*G + 0.114*B
		// Integer approximation: (R*77 + G*150 + B*29) >> 8
		uint32_t gray = (r * 77 + g * 150 + b * 29) >> 8;
		// Reconstruct pixel (Keep Alpha, set R=G=B=Gray)
		dst[i] = (a << 24) | (gray << 16) | (gray << 8) | gray;
	}
    #pragma endscop
}