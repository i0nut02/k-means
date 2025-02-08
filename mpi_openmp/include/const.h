#ifndef CONST_H
#define CONST_H


#define MAXLINE 200000
#define DEFAULT_CLASS -1
#define SEED 0

// Error codes
#define INPUT_ERR -1
#define MEMORY_ALLOCATION_ERR -2
#define TOO_MUCH_COLUMNS_ERR -3
#define READ_ERR -4
#define WRITE_ERR -5

#define CACHE_LINE_SIZE 64
#define PAD_INT (CACHE_LINE_SIZE / sizeof(int))

#endif